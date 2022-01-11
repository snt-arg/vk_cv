use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::{Device, Queue},
    image::{ImageAccess, StorageImage},
};

use crate::utils::{self, ImageInfo};

use super::{Io, IoElement, PipeOutput, ProcessingElement};

pub struct Output {}

impl Output {
    pub fn new() -> Self {
        Self {}
    }
}

impl ProcessingElement for Output {
    fn build(
        &mut self,
        device: Arc<Device>,
        _queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input: &IoElement,
    ) -> IoElement {
        // input image
        let input_img = input.output_image().unwrap();

        // output buffer (cpu accessible)
        let depth = input_img.format().size().unwrap() as u32;
        let count = input_img.dimensions().width()
            * input_img.dimensions().height()
            * input_img.dimensions().depth()
            * depth;
        let output_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            (0..count).map(|_| 0u8),
        )
        .unwrap();

        // build command buffer
        builder
            .copy_image_to_buffer(input_img.clone(), output_buffer.clone())
            .unwrap();

        IoElement {
            input: Io::Image(input_img),
            output: Io::Buffer(output_buffer),
        }
    }
}

impl PipeOutput for Output {}
