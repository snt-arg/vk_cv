pub mod convolution;
pub mod filter;
pub mod grayscale;
pub mod hsvconv;
pub mod input;
pub mod output;

use std::sync::Arc;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;

pub trait ProcessingElement {
    // fn pipeline(&self) -> Arc<ComputePipeline>;
    // fn descriptor_set(&self) -> Arc<PersistentDescriptorSet>;
    fn command_buffer(&self) -> Arc<PrimaryAutoCommandBuffer>;
}
