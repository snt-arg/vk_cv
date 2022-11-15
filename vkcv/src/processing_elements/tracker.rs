use half::f16;
use std::sync::Arc;
use vulkano::{
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    format::Format,
    image::{view::ImageView, ImageAccess, StorageImage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sampler::{Sampler, SamplerCreateInfo},
};

use crate::{
    endpoints::image_download::TransferredImage,
    utils::{self, ImageInfo},
    vk_init::VkContext,
};

use super::{AutoCommandBufferBuilder, Io, IoFragment, ProcessingElement};

// 0th pass: canvas (power of two)
mod cs_canvas {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/tracker_canvas.comp.glsl",
    }
}

// 1st pass: coordinate mask
mod cs_cm {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/tracker_cm.comp.glsl",
    }
}

// subsequent passes: scale down 2x
mod cs_pool2 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/mean_pooling_2.comp.glsl",
    }
}

mod cs_pool2_sampler {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/mean_pooling_2.sampler.comp.glsl",
    }
}

// subsequent passes: scale down 4x
mod cs_pool4 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/mean_pooling_4.comp.glsl",
    }
}

mod cs_pool4_sampler {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/mean_pooling_4.sampler.comp.glsl",
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PoolingStrategy {
    Pooling4,
    Pooling2,
    SampledPooling4,
    SampledPooling2,
}

#[derive(Clone, Copy, Debug)]
pub enum Canvas {
    Pad,
    Crop,
}

pub struct Tracker {
    pooling: PoolingStrategy,
    canvas: Canvas,
}

impl Tracker {
    pub fn new(pooling: PoolingStrategy, canvas: Canvas) -> Self {
        Self { pooling, canvas }
    }

    fn canvas(
        ctx: &VkContext,
        builder: &mut AutoCommandBufferBuilder,
        input_img: Arc<StorageImage>,
        canvas: Canvas,
    ) -> Arc<StorageImage> {
        let pipeline = {
            let shader = cs_canvas::load(ctx.device.clone()).unwrap();
            ComputePipeline::new(
                ctx.device.clone(),
                shader.entry_point("main").unwrap(),
                &cs_canvas::SpecializationConstants {
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        };

        // find closest power of two size
        let stride = input_img
            .dimensions()
            .width()
            .max(input_img.dimensions().height());

        let pot = match canvas {
            Canvas::Crop => 2u32.pow((stride as f32).log2().floor() as u32),
            Canvas::Pad => 2u32.pow((stride as f32).log2().ceil() as u32),
        };

        // skip this pass if image is already power of two
        if input_img.dimensions().width_height() == [pot, pot] {
            return input_img;
        }

        // output image
        let output_img = utils::create_storage_image(
            ctx,
            &ImageInfo {
                format: Format::R8_UNORM,
                height: pot,
                width: pot,
            },
        );

        // setup layout
        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let input_img_view = ImageView::new_default(input_img.clone()).unwrap();
        let output_img_view = ImageView::new_default(output_img.clone()).unwrap();

        let set = PersistentDescriptorSet::new(
            &ctx.memory.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, input_img_view),
                WriteDescriptorSet::image_view(1, output_img_view),
            ],
        )
        .unwrap();

        let workgroups = utils::workgroups(&output_img.dimensions().width_height(), &[16, 16]);

        // build command buffer
        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set,
            )
            .dispatch(workgroups)
            .unwrap();

        output_img
    }

    fn coordinate_mask(
        ctx: &VkContext,
        builder: &mut AutoCommandBufferBuilder,
        input_img: Arc<StorageImage>,
        sub_dims: &[u32; 2],
    ) -> Arc<StorageImage> {
        // ref: https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-26-object-detection-color-using-gpu-real-time-video
        // pipeline
        let pipeline = {
            let shader = cs_cm::load(ctx.device.clone()).unwrap();
            ComputePipeline::new(
                ctx.device.clone(),
                shader.entry_point("main").unwrap(),
                &cs_cm::SpecializationConstants {
                    inv_width: 1.0 / sub_dims[0] as f32,
                    inv_height: 1.0 / sub_dims[1] as f32,
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        };

        // output image
        let output_img = utils::create_storage_image(
            ctx,
            &ImageInfo::from_image(&input_img, Format::R16G16B16A16_SFLOAT),
        );

        // setup layout
        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let input_img_view = ImageView::new_default(input_img.clone()).unwrap();
        let output_img_view = ImageView::new_default(output_img.clone()).unwrap();

        let set = PersistentDescriptorSet::new(
            &ctx.memory.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, input_img_view),
                WriteDescriptorSet::image_view(1, output_img_view),
            ],
        )
        .unwrap();

        let workgroups = utils::workgroups(&output_img.dimensions().width_height(), &[16, 16]);

        // build command buffer
        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .dispatch(workgroups)
            .unwrap();

        output_img
    }

    fn pooling(
        ctx: &VkContext,
        builder: &mut AutoCommandBufferBuilder,
        mut input_img: Arc<StorageImage>,
        pooling_strategy: PoolingStrategy,
    ) -> Arc<StorageImage> {
        let size = input_img.dimensions().width_height();
        assert_eq!(size[0], size[1]);

        let divs_by_2 = (size[0] as f32).log2().ceil() as u32;
        let divs_by_4 = (divs_by_2 as f32 / 2.0).floor() as u32;
        let remaining_divs_by_2 = divs_by_2 - (divs_by_4 * 2);

        match pooling_strategy {
            PoolingStrategy::Pooling4 => {
                for _ in 0..divs_by_4 {
                    input_img = Self::pooling4(ctx, builder, input_img, false);
                }

                for _ in 0..remaining_divs_by_2 {
                    input_img = Self::pooling2(ctx, builder, input_img, false);
                }
            }
            PoolingStrategy::SampledPooling4 => {
                for _ in 0..divs_by_4 {
                    input_img = Self::pooling4(ctx, builder, input_img, true);
                }

                for _ in 0..remaining_divs_by_2 {
                    input_img = Self::pooling2(ctx, builder, input_img, true);
                }
            }
            PoolingStrategy::Pooling2 => {
                for _ in 0..divs_by_2 {
                    input_img = Self::pooling2(ctx, builder, input_img, false);
                }
            }
            PoolingStrategy::SampledPooling2 => {
                for _ in 0..divs_by_2 {
                    input_img = Self::pooling2(ctx, builder, input_img, true);
                }
            }
        }

        input_img
    }

    fn pooling2(
        ctx: &VkContext,
        builder: &mut AutoCommandBufferBuilder,
        input_img: Arc<StorageImage>,
        use_sampler: bool,
    ) -> Arc<StorageImage> {
        let in_size = input_img.dimensions().width();
        let out_size = in_size / 2;

        let local_size = [out_size.min(16), out_size.min(16)];

        let pipeline = if use_sampler {
            let shader = cs_pool2_sampler::load(ctx.device.clone()).unwrap();
            ComputePipeline::new(
                ctx.device.clone(),
                shader.entry_point("main").unwrap(),
                &cs_pool2_sampler::SpecializationConstants {
                    constant_0: local_size[0],
                    constant_1: local_size[1],
                    inv_size: 1.0 / (out_size as f32),
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        } else {
            let shader = cs_pool2::load(ctx.device.clone()).unwrap();
            ComputePipeline::new(
                ctx.device.clone(),
                shader.entry_point("main").unwrap(),
                &cs_pool2::SpecializationConstants {
                    constant_0: local_size[0],
                    constant_1: local_size[1],
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        };

        // output image
        let output_img = utils::create_storage_image(
            ctx,
            &ImageInfo {
                format: Format::R16G16B16A16_SFLOAT,
                height: out_size,
                width: out_size,
            },
        );

        // setup layout
        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let input_img_view = ImageView::new_default(input_img.clone()).unwrap();
        let output_img_view = ImageView::new_default(output_img.clone()).unwrap();

        let set = if use_sampler {
            let sampler = Sampler::new(
                ctx.device.clone(),
                SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
            )
            .unwrap();
            PersistentDescriptorSet::new(
                &ctx.memory.descriptor_set_allocator,
                layout.clone(),
                [
                    WriteDescriptorSet::image_view_sampler(0, input_img_view, sampler),
                    WriteDescriptorSet::image_view(1, output_img_view),
                ],
            )
            .unwrap()
        } else {
            PersistentDescriptorSet::new(
                &ctx.memory.descriptor_set_allocator,
                layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, input_img_view),
                    WriteDescriptorSet::image_view(1, output_img_view),
                ],
            )
            .unwrap()
        };

        // workgroups
        let workgroups = utils::workgroups(&output_img.dimensions().width_height(), &local_size);

        // build command buffer
        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .dispatch(workgroups)
            .unwrap();

        output_img
    }

    fn pooling4(
        ctx: &VkContext,
        builder: &mut AutoCommandBufferBuilder,
        input_img: Arc<StorageImage>,
        use_sampler: bool,
    ) -> Arc<StorageImage> {
        let in_size = input_img.dimensions().width();
        let out_size = in_size / 4;

        let local_size = [out_size.min(16), out_size.min(16)];

        let pipeline = if use_sampler {
            let shader = cs_pool4_sampler::load(ctx.device.clone()).unwrap();
            ComputePipeline::new(
                ctx.device.clone(),
                shader.entry_point("main").unwrap(),
                &cs_pool4_sampler::SpecializationConstants {
                    constant_0: local_size[0],
                    constant_1: local_size[1],
                    inv_size: 1.0 / (out_size as f32),
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        } else {
            let shader = cs_pool4::load(ctx.device.clone()).unwrap();
            ComputePipeline::new(
                ctx.device.clone(),
                shader.entry_point("main").unwrap(),
                &cs_pool4::SpecializationConstants {
                    constant_0: local_size[0],
                    constant_1: local_size[1],
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        };

        // output image
        let output_img = utils::create_storage_image(
            ctx,
            &ImageInfo {
                format: Format::R16G16B16A16_SFLOAT,
                height: out_size,
                width: out_size,
            },
        );

        // setup layout
        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let input_img_view = ImageView::new_default(input_img.clone()).unwrap();
        let output_img_view = ImageView::new_default(output_img.clone()).unwrap();

        let set = if use_sampler {
            let sampler = Sampler::new(
                ctx.device.clone(),
                SamplerCreateInfo::simple_repeat_linear_no_mipmap(),
            )
            .unwrap();
            PersistentDescriptorSet::new(
                &ctx.memory.descriptor_set_allocator,
                layout.clone(),
                [
                    WriteDescriptorSet::image_view_sampler(0, input_img_view, sampler),
                    WriteDescriptorSet::image_view(1, output_img_view),
                ],
            )
            .unwrap()
        } else {
            PersistentDescriptorSet::new(
                &ctx.memory.descriptor_set_allocator,
                layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, input_img_view),
                    WriteDescriptorSet::image_view(1, output_img_view),
                ],
            )
            .unwrap()
        };

        // workgroups
        let workgroups = utils::workgroups(&output_img.dimensions().width_height(), &local_size);

        // build command buffer
        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .dispatch(workgroups)
            .unwrap();

        output_img
    }
}

impl ProcessingElement for Tracker {
    fn build(
        &self,
        ctx: &VkContext,
        builder: &mut AutoCommandBufferBuilder,
        input: &IoFragment,
    ) -> IoFragment {
        // input image
        let input_img = input.output_image().unwrap();

        // canvas the input image to be a power of two
        // this is skipped if the input image is already a POT
        let output_img_canvas = Self::canvas(ctx, builder, input_img.clone(), self.canvas);

        // coordinate mask
        let output_img_cm = Self::coordinate_mask(
            ctx,
            builder,
            output_img_canvas.clone(),
            &input_img.dimensions().width_height(),
        );

        // scale down to 1x1 px
        let output_img = Self::pooling(ctx, builder, output_img_cm.clone(), self.pooling);

        // create a descriptive label
        let label = format!(
            "Tracker\n\t- {}\n\t- {}\n\t- {}\n\t- {}\n\t",
            utils::basic_label("Input", &input_img),
            utils::basic_label("Canvas", &output_img_canvas),
            utils::basic_label("Coordinate Mask", &output_img_cm),
            utils::basic_label("Downscale", &output_img),
        );

        IoFragment {
            input: Io::Image(input_img),
            output: Io::Image(output_img.clone()),
            label,
        }
    }
}

pub fn centroid(tf_img: &TransferredImage) -> ([f32; 2], f32) {
    assert_eq!(tf_img.info().width, 1);
    assert_eq!(tf_img.info().height, 1);

    let buffer = tf_img.buffer_content();

    match tf_img.info().format {
        Format::R16G16B16A16_SFLOAT => {
            let x = f16::from_le_bytes([buffer[0], buffer[1]]).to_f32();
            let y = f16::from_le_bytes([buffer[2], buffer[3]]).to_f32();
            let z = f16::from_le_bytes([buffer[4], buffer[5]]).to_f32();

            ([x / z, y / z], z)
        }
        Format::R32G32B32A32_SFLOAT => {
            let x = f32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
            let y = f32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);
            let z = f32::from_le_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]);
            ([x / z, y / z], z)
        }
        _ => ([f32::NAN, f32::NAN], f32::NAN),
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn reduce_4x() {
        let size = 1024;

        let a = (size as f32).log2().ceil();
        let divs_by_4 = (a / 2.0).floor() as u32;
        let divs_by_2 = a as u32 - (divs_by_4 * 2);

        dbg!(size);
        dbg!(divs_by_4);
        dbg!(divs_by_2);
    }
}
