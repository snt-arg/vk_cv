pub mod color_filter;
pub mod convolution;
pub mod convolution_2p;
pub mod grayscale;
pub mod hsvconv;
pub mod input;
pub mod morphology;
pub mod output;
pub mod tracker;

use std::sync::Arc;
use vulkano::{
    buffer::CpuAccessibleBuffer,
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    device::{Device, Queue},
    image::StorageImage,
};

pub trait ProcessingElement {
    fn build(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input: &IoElement,
    ) -> IoElement;
}

// marker trait
pub trait PipeOutput {}
pub trait PipeOutputElement: PipeOutput + ProcessingElement {
    // fn as_pe(&self) -> &dyn ProcessingElement {
    //     &self
    // }
}

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
pub struct IoElement {
    input: Io,
    output: Io,
}

impl IoElement {
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

    pub fn dummy() -> Self {
        Self {
            input: Io::None,
            output: Io::None,
        }
    }
}
