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

use crate::utils::{self, ImageInfo};

use super::ProcessingElement;

pub struct Input {
    output_img: Arc<StorageImage>,
    input_buffer: Arc<CpuAccessibleBuffer<[u8]>>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

impl Input {
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        input_data: Vec<u8>,
        input_format: ImageInfo,
    ) -> Self {
        let output_img = StorageImage::new(
            device.clone(),
            ImageDimensions::Dim2d {
                width: input_format.width,
                height: input_format.height,
                array_layers: 1,
            },
            Format::R8G8B8A8_UNORM,
            Some(queue.family()),
        )
        .unwrap();

        let count = input_format.width * input_format.height * 4;

        let input_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, input_data)
                .unwrap();

        // build command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .copy_buffer_to_image(input_buffer.clone(), output_img.clone())
            .unwrap();

        let command_buffer = Arc::new(builder.build().unwrap());

        Self {
            output_img,
            input_buffer,
            command_buffer,
        }
    }

    pub fn output_image(&self) -> Arc<StorageImage> {
        self.output_img.clone()
    }
}

impl ProcessingElement for Input {
    fn command_buffer(&self) -> Arc<PrimaryAutoCommandBuffer> {
        self.command_buffer.clone()
    }
}
