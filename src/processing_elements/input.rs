use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    device::{Device, Queue},
    image::{ImageCreateFlags, ImageUsage, StorageImage},
};

use crate::utils::{create_storage_image, ImageInfo};

use super::{PipeInput, ProcessingElement};

pub struct Input {
    output_img: Arc<StorageImage>,
    input_buffer: Arc<CpuAccessibleBuffer<[u8]>>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

impl Input {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, input_format: &ImageInfo) -> Self {
        // output image
        let output_img = create_storage_image(device.clone(), queue.clone(), input_format);

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

    fn input_image(&self) -> Option<Arc<StorageImage>> {
        None
    }

    fn output_image(&self) -> Option<Arc<StorageImage>> {
        Some(self.output_img.clone())
    }
}

impl PipeInput for Input {}
