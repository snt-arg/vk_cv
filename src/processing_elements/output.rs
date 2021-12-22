use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    device::{Device, Queue},
    image::{ImageAccess, StorageImage},
};

use crate::utils::{self, ImageInfo};

use super::ProcessingElement;

pub struct Output {
    input_img: Arc<StorageImage>,
    output_buffer: Arc<CpuAccessibleBuffer<[u8]>>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

impl Output {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, input_img: Arc<StorageImage>) -> Self {
        let count = input_img.dimensions().width()
            * input_img.dimensions().height()
            * input_img.dimensions().depth()
            * 4;
        let output_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            (0..count).map(|_| 0u8),
        )
        .unwrap();

        // build command buffer
        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        builder
            .copy_image_to_buffer(input_img.clone(), output_buffer.clone())
            .unwrap();

        let command_buffer = Arc::new(builder.build().unwrap());

        Self {
            input_img,
            output_buffer,
            command_buffer,
        }
    }

    pub fn save_output_buffer(&self, filename: &str) {
        let buffer_content = self.output_buffer.read().unwrap();

        let info = ImageInfo {
            width: self.input_img.dimensions().width(),
            height: self.input_img.dimensions().height(),
            depth: 4,
        };

        utils::write_image(filename, &buffer_content, info);
    }
}

impl ProcessingElement for Output {
    fn command_buffer(&self) -> Arc<PrimaryAutoCommandBuffer> {
        self.command_buffer.clone()
    }
}
