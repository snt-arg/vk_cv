pub mod convolution;

use std::sync::Arc;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::pipeline::ComputePipeline;

trait ProcessingElement {
    fn pipeline(&self) -> Arc<ComputePipeline>;
    fn descriptor_set(&self) -> Arc<PersistentDescriptorSet>;
}
