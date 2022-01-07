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
        input: &dyn ProcessingElement,
    );
    fn output_image(&self) -> Option<Arc<StorageImage>>;
    fn input_image(&self) -> Option<Arc<StorageImage>>;
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

pub struct DummyPE {}

impl ProcessingElement for DummyPE {
    fn build(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input: &dyn ProcessingElement,
    ) {
        unimplemented!()
    }

    fn output_image(&self) -> Option<Arc<StorageImage>> {
        None
    }

    fn input_image(&self) -> Option<Arc<StorageImage>> {
        None
    }
}
