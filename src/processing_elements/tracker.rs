use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageAccess, StorageImage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
};

use crate::utils::{self, ImageInfo};

use super::{Io, IoFragment, ProcessingElement};

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

pub struct Tracker {
    pooling_strategy: PoolingStrategy,
    crop: bool,
}

impl Tracker {
    pub fn new(pooling_strategy: PoolingStrategy, crop: bool) -> Self {
        Self {
            pooling_strategy,
            crop,
        }
    }

    fn canvas(
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input_img: Arc<StorageImage>,
        crop: bool,
    ) -> Arc<StorageImage> {
        let pipeline = {
            let shader = cs_canvas::load(device.clone()).unwrap();
            ComputePipeline::new(
                device.clone(),
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

        let pot = if crop {
            2u32.pow((stride as f32).log2().floor() as u32)
        } else {
            2u32.pow((stride as f32).log2().ceil() as u32)
        };

        // skip this pass if image is already power of two
        if input_img.dimensions().width_height() == [pot, pot] {
            return input_img;
        }

        // output image
        let output_img = utils::create_storage_image(
            device.clone(),
            queue.clone(),
            &ImageInfo {
                format: Format::R8_UNORM,
                height: pot,
                width: pot,
            },
        );

        // setup layout
        let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        let input_img_view = ImageView::new(input_img.clone()).unwrap();
        let output_img_view = ImageView::new(output_img.clone()).unwrap();

        set_builder.add_image(input_img_view).unwrap();
        set_builder.add_image(output_img_view).unwrap();

        let set = set_builder.build().unwrap();

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

    fn coordinate_mask(
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input_img: Arc<StorageImage>,
        sub_dims: &[u32; 2],
    ) -> Arc<StorageImage> {
        // ref: https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-26-object-detection-color-using-gpu-real-time-video
        // pipeline
        let pipeline = {
            let shader = cs_cm::load(device.clone()).unwrap();
            ComputePipeline::new(
                device.clone(),
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
            device.clone(),
            queue.clone(),
            &ImageInfo::from_image(&input_img, Format::R32G32B32A32_SFLOAT),
        );

        // setup layout
        let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        let input_img_view = ImageView::new(input_img.clone()).unwrap();
        let output_img_view = ImageView::new(output_img.clone()).unwrap();

        set_builder.add_image(input_img_view).unwrap();
        set_builder.add_image(output_img_view).unwrap();

        let set = set_builder.build().unwrap();

        let workgroups = utils::workgroups(&sub_dims, &[16, 16]);

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
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
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
                    input_img =
                        Self::pooling4(device.clone(), queue.clone(), builder, input_img, false);
                }

                for _ in 0..remaining_divs_by_2 {
                    input_img =
                        Self::pooling2(device.clone(), queue.clone(), builder, input_img, false);
                }
            }
            PoolingStrategy::SampledPooling4 => {
                for _ in 0..divs_by_4 {
                    input_img =
                        Self::pooling4(device.clone(), queue.clone(), builder, input_img, true);
                }

                for _ in 0..remaining_divs_by_2 {
                    input_img =
                        Self::pooling2(device.clone(), queue.clone(), builder, input_img, true);
                }
            }
            PoolingStrategy::Pooling2 => {
                for _ in 0..divs_by_2 {
                    input_img =
                        Self::pooling2(device.clone(), queue.clone(), builder, input_img, false);
                }
            }
            PoolingStrategy::SampledPooling2 => {
                for _ in 0..divs_by_2 {
                    input_img =
                        Self::pooling2(device.clone(), queue.clone(), builder, input_img, true);
                }
            }
        }

        input_img
    }

    fn pooling2(
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input_img: Arc<StorageImage>,
        use_sampler: bool,
    ) -> Arc<StorageImage> {
        let in_size = input_img.dimensions().width();
        let out_size = in_size / 2;

        let local_size = [out_size.min(16), out_size.min(16)];

        let pipeline = if use_sampler {
            let shader = cs_pool2_sampler::load(device.clone()).unwrap();
            ComputePipeline::new(
                device.clone(),
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
            let shader = cs_pool2::load(device.clone()).unwrap();
            ComputePipeline::new(
                device.clone(),
                shader.entry_point("main").unwrap(),
                &cs_pool2::SpecializationConstants {
                    constant_0: local_size[0],
                    constant_1: local_size[1],
                    inv_size: 1.0 / (out_size as f32),
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        };

        // output image
        let output_img = utils::create_storage_image(
            device.clone(),
            queue.clone(),
            &ImageInfo {
                format: Format::R32G32B32A32_SFLOAT,
                height: out_size,
                width: out_size,
            },
        );

        // setup layout
        let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        let input_img_view = ImageView::new(input_img.clone()).unwrap();
        let output_img_view = ImageView::new(output_img.clone()).unwrap();

        if use_sampler {
            let sampler = Sampler::new(
                device.clone(),
                Filter::Linear,
                Filter::Linear,
                MipmapMode::Nearest,
                SamplerAddressMode::Repeat,
                SamplerAddressMode::Repeat,
                SamplerAddressMode::Repeat,
                0.0,
                1.0,
                0.0,
                0.0,
            )
            .unwrap();

            set_builder
                .add_sampled_image(input_img_view, sampler)
                .unwrap();
        } else {
            set_builder.add_image(input_img_view).unwrap();
        }

        set_builder.add_image(output_img_view).unwrap();

        let set = set_builder.build().unwrap();

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
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input_img: Arc<StorageImage>,
        use_sampler: bool,
    ) -> Arc<StorageImage> {
        let in_size = input_img.dimensions().width();
        let out_size = in_size / 4;

        let local_size = [out_size.min(16), out_size.min(16)];

        let pipeline = if use_sampler {
            let shader = cs_pool4_sampler::load(device.clone()).unwrap();
            ComputePipeline::new(
                device.clone(),
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
            let shader = cs_pool4::load(device.clone()).unwrap();
            ComputePipeline::new(
                device.clone(),
                shader.entry_point("main").unwrap(),
                &cs_pool4::SpecializationConstants {
                    constant_0: local_size[0],
                    constant_1: local_size[1],
                    inv_size: 1.0 / (out_size as f32),
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        };

        // output image
        let output_img = utils::create_storage_image(
            device.clone(),
            queue.clone(),
            &ImageInfo {
                format: Format::R32G32B32A32_SFLOAT,
                height: out_size,
                width: out_size,
            },
        );

        // setup layout
        let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        let input_img_view = ImageView::new(input_img.clone()).unwrap();
        let output_img_view = ImageView::new(output_img.clone()).unwrap();

        if use_sampler {
            let sampler = Sampler::new(
                device.clone(),
                Filter::Linear,
                Filter::Linear,
                MipmapMode::Nearest,
                SamplerAddressMode::Repeat,
                SamplerAddressMode::Repeat,
                SamplerAddressMode::Repeat,
                0.0,
                1.0,
                0.0,
                0.0,
            )
            .unwrap();

            set_builder
                .add_sampled_image(input_img_view, sampler)
                .unwrap();
        } else {
            set_builder.add_image(input_img_view).unwrap();
        }

        set_builder.add_image(output_img_view).unwrap();

        let set = set_builder.build().unwrap();

        // let workgroups =  (out_size as f32 / local_size as f32).ceil() as u32;
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
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input: &IoFragment,
    ) -> IoFragment {
        // input image
        let input_img = input.output_image().unwrap();

        // canvas the input image to be a power of two
        // this is skipped if the input image is already a POT
        let output_img = Self::canvas(
            device.clone(),
            queue.clone(),
            builder,
            input_img.clone(),
            self.crop,
        );

        // coordinate mask
        let output_img = Self::coordinate_mask(
            device.clone(),
            queue.clone(),
            builder,
            output_img.clone(),
            &input_img.dimensions().width_height(),
        );

        // scale down to 1x1 px
        let output_img = Self::pooling(
            device,
            queue,
            builder,
            output_img.clone(),
            self.pooling_strategy,
        );

        IoFragment {
            input: Io::Image(input_img),
            output: Io::Image(output_img),
            label: "Tracker".to_string(),
        }
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
