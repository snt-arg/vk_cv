use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::{Device, Queue},
};

use crate::utils::{create_storage_image, ImageInfo};

use super::{Io, IoElement, PipeInput, ProcessingElement};

pub struct Input {
    input_format: ImageInfo,
}

impl Input {
    pub fn new(input_format: ImageInfo) -> Self {
        Self { input_format }
    }
}

impl ProcessingElement for Input {
    fn build(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        _input: &IoElement,
    ) -> IoElement {
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

        IoElement {
            input: Io::Buffer(input_buffer),
            output: Io::Image(output_img),
        }
    }
}

impl PipeInput for Input {}
