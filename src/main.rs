use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::PersistentDescriptorSet,
    format::Format,
    image::{view::ImageView, ImageDimensions, StorageImage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
};

use crate::processing_elements::{convolution::Convolution, output::Output};
use crate::processing_elements::{input::Input, ProcessingElement};

mod processing_elements;
mod utils;
mod vk_init;

fn main() {
    // v3d specs/properties: https://vulkan.gpuinfo.org/displayreport.php?id=13073#properties

    let (img_info, img_data) = utils::load_image("Large_Scaled_Forest_Lizard.png");

    // init device
    let (device, mut queues) = vk_init::init();

    let queue = queues.next().unwrap();

    // create a convolution pipeline
    let pe_input = Input::new(device.clone(), queue.clone(), img_data, img_info);
    let pe_conv = Convolution::new(device.clone(), queue.clone(), pe_input.output_image());
    let pe_out = Output::new(device.clone(), queue.clone(), pe_conv.output_image());

    // exec command buffer
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), pe_input.command_buffer())
        .unwrap()
        .then_execute(queue.clone(), pe_conv.command_buffer())
        .unwrap()
        .then_execute(queue.clone(), pe_out.command_buffer())
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    // check data
    future.wait(None).unwrap();

    pe_out.save_output_buffer("out.png");
}
