mod processing_elements;
mod realsense;
mod utils;
mod vk_init;

use processing_elements::convolution_2p::Convolution2Pass;
use processing_elements::hsvconv::Hsvconv;
use processing_elements::morphology::Operation;
use realsense::Realsense;
use vulkano::sync::{self, GpuFuture};

use crate::processing_elements::{
    color_filter::ColorFilter, convolution::Convolution, grayscale::Grayscale,
    morphology::Morphology, output::Output, tracker::Tracker,
};
use crate::processing_elements::{input::Input, ProcessingElement};
use crate::utils::cv_pipeline;
use anyhow::Result;

fn main() -> Result<()> {
    // let mut realsense = Realsense::open();

    // v3d specs/properties: https://vulkan.gpuinfo.org/displayreport.php?id=13073#properties

    let (img_info, img_data) = utils::load_image("j.png");

    // init device
    let (device, mut queues) = vk_init::init();
    let queue = queues.next().unwrap();

    // create a convolution pipeline
    let mut pe_input = Input::new(img_info);
    let mut pe_gsc = Grayscale::new();

    let mut pe_hsv = Hsvconv::new();
    let mut pe_hsv_filter = ColorFilter::new();

    // let pe_conv = Convolution::new(device.clone(), queue.clone(), &pe_hsv_filter);
    // let pe_conv_2p = Convolution2Pass::new(device.clone(), queue.clone(), &pe_gsc);
    let mut pe_erode = Morphology::new(Operation::Erode);
    let mut pe_dilate = Morphology::new(Operation::Dilate);
    let mut pe_tracker = Tracker::new();
    let mut pe_out = Output::new();

    let pipeline_cb = cv_pipeline(
        device.clone(),
        queue.clone(),
        &mut pe_input,
        &mut [&mut pe_gsc, &mut pe_erode, &mut pe_dilate, &mut pe_tracker],
        &mut pe_out,
    );

    for i in 0..200 {
        // let color_image = realsense.fetch_image();
        //println!("{} x {}", color_image.width(), color_image.height());
        let pipeline_started = std::time::Instant::now();

        // upload image to GPU
        pe_input.copy_input_data(&img_data);

        // process on GPU
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), pipeline_cb.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        // wait till finished
        future.wait(None).unwrap();
        let pipeline_dt = std::time::Instant::now() - pipeline_started;
        println!("Pipeline took {} us", pipeline_dt.as_micros());

        if i == 0 {
            pe_out.save_output_buffer(&format!("out_{}.png", i));
        }
    }
    Ok(())
}
