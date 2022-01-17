use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::{Device, Queue},
    image::ImageAccess,
};

use super::{Io, IoFragment, PipeOutput, ProcessingElement};

pub struct Output {}

impl Output {
    pub fn new() -> Self {
        Self {}
    }
}

impl ProcessingElement for Output {
    fn build(
        &self,
        device: Arc<Device>,
        _queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input: &IoFragment,
    ) -> IoFragment {
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

        IoFragment {
            input: Io::Image(input_img),
            output: Io::Buffer(output_buffer),
            label: "Output".to_string(),
        }
    }
}

impl PipeOutput for Output {}
