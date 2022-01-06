use std::{ops::Deref, sync::Arc};
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageAccess, StorageImage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode},
};

use crate::utils::{create_storage_image, ImageInfo};

use super::ProcessingElement;

// 1st pass: coordinate mask
mod cs1 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/tracker_cm.comp.glsl",
    }
}

// subsequent passes: scale down
mod cs2 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/tracker_2x.comp.glsl",
    }
}

pub struct Tracker {
    input_img: Arc<StorageImage>,
    output_img: Arc<StorageImage>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

impl Tracker {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, input: &dyn ProcessingElement) -> Self {
        let local_size = (1, 8);

        // input image
        let input_img = input.output_image().unwrap();

        // pipeline
        let pipeline = {
            let shader = cs1::load(device.clone()).unwrap();
            ComputePipeline::new(
                device.clone(),
                shader.entry_point("main").unwrap(),
                &cs1::SpecializationConstants {
                    inv_width: 1.0 / input_img.dimensions().width() as f32,
                    inv_height: 1.0 / input_img.dimensions().height() as f32,
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        };

        // output image
        let output_img = create_storage_image(
            device.clone(),
            queue.clone(),
            &ImageInfo::from_image(&input_img, Format::R32G32B32A32_SFLOAT),
        );
        // https://developer.nvidia.com/gpugems/gpugems3/part-iv-image-effects/chapter-26-object-detection-color-using-gpu-real-time-video
        // let samples = Sampler::new(
        //     device.clone(),
        //     Filter::Linear,
        //     Filter::Linear,
        //     MipmapMode::Nearest,
        //     SamplerAddressMode::Repeat,
        //     SamplerAddressMode::Repeat,
        //     SamplerAddressMode::Repeat,
        //     0.0,
        //     1.0,
        //     0.0,
        //     0.0,
        // );

        // setup layout
        let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        let input_img_view = ImageView::new(input_img.clone()).unwrap();
        let output_img_view = ImageView::new(output_img.clone()).unwrap();

        set_builder.add_image(input_img_view).unwrap();
        set_builder.add_image(output_img_view).unwrap();
        // set_builder.add_sampled_image(input_img_view, sampler)

        let set = set_builder.build().unwrap();

        // build command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            // .push_constants(pipeline.layout().clone(), 0, push_constants)
            .dispatch([
                input_img.dimensions().width() / 16,
                input_img.dimensions().height() / 16,
                1,
            ])
            .unwrap();

        let output_img = Self::reduce(device, queue, &mut builder, output_img.clone());

        let command_buffer = Arc::new(builder.build().unwrap());

        Self {
            input_img,
            output_img,
            command_buffer,
        }
    }

    fn reduce(
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        mut input_img: Arc<StorageImage>,
    ) -> Arc<StorageImage> {
        let size = input_img.dimensions().width_height();
        assert_eq!(size[0], size[1]);

        let steps = (size[0] as f32).log2() as u32;
        for _ in 0..steps {
            let in_size = input_img.dimensions().width();
            let out_size = in_size / 2;

            let pipeline = {
                let shader = cs2::load(device.clone()).unwrap();
                ComputePipeline::new(
                    device.clone(),
                    shader.entry_point("main").unwrap(),
                    &cs1::SpecializationConstants {
                        ..Default::default()
                    },
                    None,
                    |_| {},
                )
                .unwrap()
            };

            // output image
            let output_img = create_storage_image(
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

            set_builder.add_image(input_img_view).unwrap();
            set_builder.add_image(output_img_view).unwrap();

            let set = set_builder.build().unwrap();

            let workgroups = (out_size as f32 / 16.0).ceil() as u32;

            // build command buffer
            builder
                .bind_pipeline_compute(pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    pipeline.layout().clone(),
                    0,
                    set.clone(),
                )
                .dispatch([workgroups, workgroups, 1])
                .unwrap();

            // swap
            input_img = output_img.clone();
        }

        input_img
    }
}

impl ProcessingElement for Tracker {
    fn command_buffer(&self) -> Arc<PrimaryAutoCommandBuffer> {
        self.command_buffer.clone()
    }

    fn input_image(&self) -> Option<Arc<StorageImage>> {
        Some(self.input_img.clone())
    }

    fn output_image(&self) -> Option<Arc<StorageImage>> {
        Some(self.output_img.clone())
    }
}
