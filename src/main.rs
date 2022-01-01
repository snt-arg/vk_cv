mod processing_elements;
mod realsense;
mod utils;
mod vk_init;

use processing_elements::convolution_2p::Convolution2Pass;
use processing_elements::hsvconv::Hsvconv;
use realsense::Realsense;
use vulkano::sync::{self, GpuFuture};

use crate::processing_elements::{
    color_filter::ColorFilter, convolution::Convolution, grayscale::Grayscale,
    morphology::Morphology, output::Output,
};
use crate::processing_elements::{input::Input, ProcessingElement};
use anyhow::Result;

fn main() -> Result<()> {
    // let mut realsense = Realsense::open();

    // v3d specs/properties: https://vulkan.gpuinfo.org/displayreport.php?id=13073#properties

    let (img_info, img_data) = utils::load_image("j.png");

    // init device
    let (device, mut queues) = vk_init::init();
    let queue = queues.next().unwrap();

    // create a convolution pipeline
    let mut pe_input = Input::new(device.clone(), queue.clone(), &img_info);
    let pe_gsc = Grayscale::new(device.clone(), queue.clone(), &pe_input);

    let pe_hsv = Hsvconv::new(device.clone(), queue.clone(), &pe_input);
    let pe_hsv_filter = ColorFilter::new(device.clone(), queue.clone(), &pe_input);

    // let pe_conv = Convolution::new(device.clone(), queue.clone(), &pe_hsv_filter);
    // let pe_conv_2p = Convolution2Pass::new(device.clone(), queue.clone(), &pe_gsc);
    let pe_morph = Morphology::new(device.clone(), queue.clone(), &pe_gsc);
    let pe_out = Output::new(device.clone(), queue.clone(), &pe_morph);

    for i in 0..200 {
        // let color_image = realsense.fetch_image();
        //println!("{} x {}", color_image.width(), color_image.height());
        let pipeline_started = std::time::Instant::now();
        // pe_input.copy_input_data(color_image.data_slice());
        pe_input.copy_input_data(&img_data);

        // exec command buffer
        let future = cv_pipeline!(
            device,
            queue,
            input: pe_input,
            elements: [pe_gsc, pe_morph], //
            output: pe_out
        );

        // let future = sync::now(device.clone())
        //     .then_execute_same_queue(pe_input.command_buffer())?
        //     .then_execute_same_queue(pe_gsc.command_buffer())?
        //     // .then_execute_same_queue(pe_hsv.command_buffer())?
        //     // .then_execute_same_queue(pe_hsv_filter.command_buffer())?
        //     // .then_execute_same_queue(pe_conv_2p.command_buffer())?
        //     .then_execute_same_queue(pe_morph.command_buffer())?
        //     .then_execute_same_queue(pe_out.command_buffer())?
        //     .then_signal_fence_and_flush()?;

        // check data
        future.wait(None).unwrap();
        let pipeline_dt = std::time::Instant::now() - pipeline_started;
        println!("Pipeline took {} us", pipeline_dt.as_micros());

        if i == 0 {
            pe_out.save_output_buffer(&format!("out_{}.png", i));
        }
    }
    Ok(())
}
