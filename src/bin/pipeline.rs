use vkcv::{
    processing_elements::{
        color_filter::ColorFilter,
        convolution::Convolution,
        convolution_2p::Convolution2Pass,
        grayscale::Grayscale,
        hsvconv::Hsvconv,
        input::Input,
        morphology::{Morphology, Operation},
        output::Output,
        tracker::Tracker,
    },
    realsense::Realsense,
    utils::{cv_pipeline, load_image},
    vk_init,
};

use anyhow::Result;
use vulkano::sync::{self, GpuFuture};

fn main() -> Result<()> {
    // let mut realsense = Realsense::open();

    // v3d specs/properties:
    //
    // maxComputeWorkGroupSize: 256
    // maxImageDimension: 4096
    // maxPushConstantsSize: 128
    //
    // https://vulkan.gpuinfo.org/displayreport.php?id=13073#properties

    println!("Realsense camera tracker");

    let (img_info, img_data) = load_image("tracking_3.png");

    // init device
    let (device, mut queues) = vk_init::init();
    let queue = queues.next().unwrap();

    // create a convolution pipeline
    let mut pe_input = Input::new(img_info);
    let mut pe_gsc = Grayscale::new();

    let mut pe_hsv = Hsvconv::new();
    let mut pe_hsv_filter = ColorFilter::new([0.235, 0.419, 0.239], [0.329, 1.0, 1.0]);

    // let pe_conv = Convolution::new(device.clone(), queue.clone(), &pe_hsv_filter);
    // let pe_conv_2p = Convolution2Pass::new(device.clone(), queue.clone(), &pe_gsc);
    let mut pe_erode = Morphology::new(Operation::Erode);
    let mut pe_dilate = Morphology::new(Operation::Dilate);
    let mut pe_tracker = Tracker::new(true, false);
    let mut pe_out = Output::new();

    let pipeline_cb = cv_pipeline(
        device.clone(),
        queue.clone(),
        &mut pe_input,
        &mut [
            &mut pe_hsv,
            &mut pe_hsv_filter,
            &mut pe_erode,
            &mut pe_dilate,
            &mut pe_tracker,
        ],
        &mut pe_out,
    );

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

    pe_out.save_output_buffer("out_0.png");
    dbg!(pe_out.centeroid());

    Ok(())
}
