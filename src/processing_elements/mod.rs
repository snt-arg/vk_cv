pub mod color_filter;
pub mod convolution;
pub mod convolution_2p;
pub mod grayscale;
pub mod hsvconv;
pub mod input;
pub mod morphology;
pub mod output;

use std::sync::Arc;
use vulkano::{command_buffer::PrimaryAutoCommandBuffer, image::StorageImage};

pub trait ProcessingElement {
    // fn pipeline(&self) -> Arc<ComputePipeline>;
    // fn descriptor_set(&self) -> Arc<PersistentDescriptorSet>;
    fn command_buffer(&self) -> Arc<PrimaryAutoCommandBuffer>;
    fn output_image(&self) -> Option<Arc<StorageImage>>;
    fn input_image(&self) -> Option<Arc<StorageImage>>;
}

// marker trait
pub trait PipeOutput {}

// marker trait
pub trait PipeInput {}
