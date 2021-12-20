pub mod convolution;
pub mod input;
pub mod output;

use std::sync::Arc;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::pipeline::ComputePipeline;

pub trait ProcessingElement {
    // fn pipeline(&self) -> Arc<ComputePipeline>;
    // fn descriptor_set(&self) -> Arc<PersistentDescriptorSet>;
    fn command_buffer(&self) -> Arc<PrimaryAutoCommandBuffer>;
}
