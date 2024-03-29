pub mod color_filter;
pub mod convolution;
pub mod convolution_2p;
pub mod grayscale;
pub mod hsvconv;
pub mod input;
pub mod morphology;
pub mod output;
pub mod pooling;
pub mod tracker;

use std::sync::Arc;
use vulkano::{
    buffer::CpuAccessibleBuffer,
    command_buffer::{allocator::StandardCommandBufferAllocator, PrimaryAutoCommandBuffer},
    image::StorageImage,
};

use crate::vk_init::VkContext;

pub type AutoCommandBufferBuilder = vulkano::command_buffer::AutoCommandBufferBuilder<
    PrimaryAutoCommandBuffer,
    Arc<StandardCommandBufferAllocator>,
>;

pub trait ProcessingElement {
    fn build(
        &self,
        ctx: &VkContext,
        builder: &mut AutoCommandBufferBuilder,
        input: &IoFragment,
    ) -> IoFragment;
}

// marker trait
pub trait PipeOutput {}
pub trait PipeOutputElement: PipeOutput + ProcessingElement {}

// marker trait
pub trait PipeInput {}
pub trait PipeInputElement: PipeOutput + ProcessingElement {}

#[derive(Clone, Debug)]
pub enum Io {
    Image(Arc<StorageImage>),
    Buffer(Arc<CpuAccessibleBuffer<[u8]>>),
    None,
}

#[derive(Clone)]
pub struct IoFragment {
    pub input: Io,
    pub output: Io,
    pub label: String,
}

impl IoFragment {
    pub fn input_image(&self) -> Option<Arc<StorageImage>> {
        match &self.input {
            Io::Image(img) => Some(img.clone()),
            _ => None,
        }
    }

    pub fn output_image(&self) -> Option<Arc<StorageImage>> {
        match &self.output {
            Io::Image(img) => Some(img.clone()),
            _ => None,
        }
    }

    pub fn input_buffer(&self) -> Option<Arc<CpuAccessibleBuffer<[u8]>>> {
        match &self.input {
            Io::Buffer(buf) => Some(buf.clone()),
            _ => None,
        }
    }

    pub fn output_buffer(&self) -> Option<Arc<CpuAccessibleBuffer<[u8]>>> {
        match &self.output {
            Io::Buffer(buf) => Some(buf.clone()),
            _ => None,
        }
    }

    pub fn none() -> Self {
        Self {
            input: Io::None,
            output: Io::None,
            label: "dummy".to_string(),
        }
    }
}
