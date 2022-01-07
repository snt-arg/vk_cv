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
    output_img: Option<Arc<StorageImage>>,
    input_buffer: Option<Arc<CpuAccessibleBuffer<[u8]>>>,
    input_format: ImageInfo,
}

impl Input {
    pub fn new(input_format: ImageInfo) -> Self {
        Self {
            input_buffer: None,
            output_img: None,
            input_format,
        }
    }

    pub fn copy_input_data(&mut self, data: &[u8]) {
        if let Ok(mut lock) = self.input_buffer.as_mut().unwrap().write() {
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
    fn input_image(&self) -> Option<Arc<StorageImage>> {
        None
    }

    fn output_image(&self) -> Option<Arc<StorageImage>> {
        self.output_img.clone()
    }

    fn build(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input: &dyn ProcessingElement,
    ) {
        // output image
        let output_img = create_storage_image(device.clone(), queue.clone(), &self.input_format);

        let count = self.input_format.bytes_count();
        let input_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            (0..count).map(|_| 0u8),
        )
        .unwrap();

        // build command buffer
        builder
            .copy_buffer_to_image(input_buffer.clone(), output_img.clone())
            .unwrap();

        self.input_buffer = Some(input_buffer);
        self.output_img = Some(output_img);
    }
}

impl PipeInput for Input {}
