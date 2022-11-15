use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::CopyImageToBufferInfo,
    image::ImageAccess,
};

use crate::vk_init::VkContext;

use super::{AutoCommandBufferBuilder, Io, IoFragment, PipeOutput, ProcessingElement};

pub struct Output {}

impl Output {
    pub fn new() -> Self {
        Self {}
    }
}

impl ProcessingElement for Output {
    fn build(
        &self,
        ctx: &VkContext,
        builder: &mut AutoCommandBufferBuilder,
        input: &IoFragment,
    ) -> IoFragment {
        // input image
        if let Some(input_img) = input.output_image() {
            // output buffer (cpu accessible)
            let depth = (input_img.format().components().iter().sum::<u8>() / 8) as u32;
            let count = input_img.dimensions().width()
                * input_img.dimensions().height()
                * input_img.dimensions().depth()
                * depth;
            let output_buffer = CpuAccessibleBuffer::from_iter(
                &ctx.memory.allocator,
                BufferUsage {
                    transfer_src: true,
                    transfer_dst: true,
                    uniform_buffer: true,
                    storage_buffer: true,
                    ..Default::default()
                },
                false,
                (0..count).map(|_| 0u8),
            )
            .unwrap();

            // build command buffer
            builder
                .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                    input_img.clone(),
                    output_buffer.clone(),
                ))
                .unwrap();

            return IoFragment {
                input: Io::Image(input_img),
                output: Io::Buffer(output_buffer),
                label: "Output".to_string(),
            };
        }
        IoFragment::none()
    }
}

impl PipeOutput for Output {}
