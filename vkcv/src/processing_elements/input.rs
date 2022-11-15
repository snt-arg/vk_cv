use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::CopyBufferToImageInfo,
};

use crate::{
    utils::{create_storage_image, ImageInfo},
    vk_init::VkContext,
};

use super::{AutoCommandBufferBuilder, Io, IoFragment, PipeInput, ProcessingElement};

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
        &self,
        ctx: &VkContext,
        builder: &mut AutoCommandBufferBuilder,
        _input: &IoFragment,
    ) -> IoFragment {
        // output image
        let output_img = create_storage_image(ctx, &self.input_format);

        let count = self.input_format.bytes_count();
        let input_buffer = CpuAccessibleBuffer::from_iter(
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
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                input_buffer.clone(),
                output_img.clone(),
            ))
            .unwrap();

        IoFragment {
            input: Io::Buffer(input_buffer),
            output: Io::Image(output_img),
            label: "Input".to_string(),
        }
    }
}

impl PipeInput for Input {}
