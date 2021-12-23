use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    device::{Device, Queue},
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage,
    },
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

use super::ProcessingElement;

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/color_filter.comp.glsl",
    }
}

pub struct ColorFilter {
    input_img: Arc<StorageImage>,
    output_img: Arc<StorageImage>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

impl ColorFilter {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, input_img: Arc<StorageImage>) -> Self {
        let pipeline = {
            let shader = cs::load(device.clone()).unwrap();
            ComputePipeline::new(
                device.clone(),
                shader.entry_point("main").unwrap(),
                &cs::SpecializationConstants {},
                None,
                |_| {},
            )
            .unwrap()
        };

        let usage = ImageUsage {
            transfer_source: true,
            transfer_destination: true,
            storage: true,
            ..ImageUsage::none()
        };
        let flags = ImageCreateFlags::none();

        let output_img = StorageImage::with_usage(
            device.clone(),
            ImageDimensions::Dim2d {
                width: input_img.dimensions().width(),
                height: input_img.dimensions().height(),
                array_layers: 1,
            },
            Format::R8_UNORM,
            usage,
            flags,
            Some(queue.family()),
        )
        .unwrap();

        // setup layout
        let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        let input_img_view = ImageView::new(input_img.clone()).unwrap();
        let output_img_view = ImageView::new(output_img.clone()).unwrap();

        set_builder.add_image(input_img_view).unwrap();
        set_builder.add_image(output_img_view).unwrap();

        let set = set_builder.build().unwrap();

        // build command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        let push_constants = cs::ty::PushConstants {
            rgb_min: [0.0, 0.0, 0.0],
            rgb_max: [0.5, 1.0, 0.5],
            _dummy0: [0, 0, 0, 0],
        };

        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .push_constants(pipeline.layout().clone(), 0, push_constants)
            .dispatch([
                input_img.dimensions().width() / 16,
                input_img.dimensions().height() / 16,
                1,
            ])
            .unwrap();

        let command_buffer = Arc::new(builder.build().unwrap());

        Self {
            input_img,
            output_img,
            command_buffer,
        }
    }

    pub fn output_image(&self) -> Arc<StorageImage> {
        self.output_img.clone()
    }
}

impl ProcessingElement for ColorFilter {
    fn command_buffer(&self) -> Arc<PrimaryAutoCommandBuffer> {
        self.command_buffer.clone()
    }
}
