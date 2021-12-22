use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    device::{Device, Queue},
    format::Format,
    image::{ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage},
};

use crate::utils::ImageInfo;

use super::ProcessingElement;

pub struct Input {
    output_img: Arc<StorageImage>,
    input_buffer: Arc<CpuAccessibleBuffer<[u8]>>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

impl Input {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, input_format: ImageInfo) -> Self {
        let usage = ImageUsage {
            transfer_destination: true,
            storage: true,
            ..ImageUsage::none()
        };
        let flags = ImageCreateFlags::none();

        let output_img = StorageImage::with_usage(
            device.clone(),
            ImageDimensions::Dim2d {
                width: input_format.width,
                height: input_format.height,
                array_layers: 1,
            },
            Format::R8G8B8A8_UNORM,
            usage,
            flags,
            Some(queue.family()),
        )
        .unwrap();

        let count = input_format.bytes_count();
        let input_buffer = CpuAccessibleBuffer::from_iter(
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

    pub fn copy_input_data(&mut self, data: &[u8]) {
        if let Ok(mut lock) = self.input_buffer.write() {
            let len = lock.len();
            if len < data.len() {
                lock.copy_from_slice(&data[0..len]);
            } else {
                lock.copy_from_slice(data);
            }
        }
    }
}

impl ProcessingElement for Input {
    fn command_buffer(&self) -> Arc<PrimaryAutoCommandBuffer> {
        self.command_buffer.clone()
    }
}
