use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
        PrimaryCommandBuffer,
    },
    descriptor_set::PersistentDescriptorSet,
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageAccess, ImageDimensions, StorageImage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
};

use super::ProcessingElement;

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/convelution.comp.glsl",
    }
}

pub struct Convolution {
    input_img: Arc<StorageImage>,
    output_img: Arc<StorageImage>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

impl Convolution {
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

        let output_img = StorageImage::new(
            device.clone(),
            ImageDimensions::Dim2d {
                width: input_img.dimensions().width(),
                height: input_img.dimensions().height(),
                array_layers: 1,
            },
            Format::R8G8B8A8_UNORM,
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
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let push_constants = cs::ty::PushConstants {
            kernel: [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0],
            offset: 1.0,
            denom: 0.5,
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

impl ProcessingElement for Convolution {
    fn command_buffer(&self) -> Arc<PrimaryAutoCommandBuffer> {
        self.command_buffer.clone()
    }
}
