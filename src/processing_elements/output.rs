use std::sync::Arc;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::{Device, Queue},
    image::{ImageAccess, StorageImage},
};

use crate::utils::{self, ImageInfo};

use super::{PipeOutput, ProcessingElement};

pub struct Output {
    input_img: Option<Arc<StorageImage>>,
    output_buffer: Option<Arc<CpuAccessibleBuffer<[u8]>>>,
}

impl Output {
    pub fn new() -> Self {
        Self {
            input_img: None,
            output_buffer: None,
        }
    }

    pub fn save_output_buffer(&self, filename: &str) {
        let buffer_content = self.output_buffer.as_ref().unwrap().read().unwrap();
        let input_img = self.input_img.as_ref().unwrap();

        let info = ImageInfo {
            width: input_img.dimensions().width(),
            height: input_img.dimensions().height(),
            format: input_img.format(),
        };

        utils::write_image(filename, &buffer_content, info);
    }

    pub fn centroid(&self) -> [f32; 2] {
        let data = self.output_buffer.as_ref().unwrap().read().unwrap();

        let x = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        // dbg!(x);
        let y = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        // dbg!(y);
        let z = f32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        // dbg!(z);

        // dbg!(x / z, y / z);

        [x / z, y / z]
    }
}

impl ProcessingElement for Output {
    fn input_image(&self) -> Option<Arc<StorageImage>> {
        self.input_img.clone()
    }

    fn output_image(&self) -> Option<Arc<StorageImage>> {
        None
    }

    fn build(
        &mut self,
        device: Arc<Device>,
        _queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input: &dyn ProcessingElement,
    ) {
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

        self.input_img = Some(input_img);
        self.output_buffer = Some(output_buffer);
    }
}

impl PipeOutput for Output {}
