use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::PersistentDescriptorSet,
    device::Device,
    format::Format,
    image::{view::ImageView, ImageDimensions, StorageImage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/convelution.comp.glsl",
    }
}

pub struct Convolution {}

impl Convolution {
    pub fn new(device: Arc<Device>) -> Self {
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

        Self {}
    }
}
