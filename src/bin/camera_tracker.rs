use vkcv::{
    processing_elements::{
        color_filter::ColorFilter,
        hsvconv::Hsvconv,
        input::Input,
        morphology::{Morphology, Operation},
        output::Output,
        tracker::Tracker,
    },
    realsense::Realsense,
    utils::cv_pipeline,
    vk_init,
};

use anyhow::Result;
use vulkano::sync::{self, GpuFuture};

fn main() -> Result<()> {
    // v3d specs/properties:
    //
    // maxComputeWorkGroupSize: 256
    // maxImageDimension: 4096
    // maxPushConstantsSize: 128
    //
    // https://vulkan.gpuinfo.org/displayreport.php?id=13073#properties

    std::env::set_var("DISPLAY", ":0");

    println!("Realsense camera tracker");

    let mut realsense = Realsense::open();

    // grab a couple of frames
    for _ in 0..5 {
        realsense.fetch_image();
    }

    let img_info = realsense.fetch_image().image_info();

    // init device
    let (device, mut queues) = vk_init::init();
    let queue = queues.next().unwrap();

    // create a color tracking pipeline
    let mut pe_input = Input::new(img_info);
    let mut pe_hsv = Hsvconv::new();
    let mut pe_hsv_filter = ColorFilter::new([0.235, 0.419, 0.239], [0.329, 1.0, 1.0]);
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

    loop {
        let image = realsense.fetch_image();
        // println!("{} x {}", image.width(), image.height());
        let pipeline_started = std::time::Instant::now();

        // upload image to GPU
        pe_input.copy_input_data(image.data_slice());

        // process on GPU
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), pipeline_cb.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        // wait till finished
        // std::thread::sleep(std::time::Duration::from_millis(30));

        // this appears to be a spinlock
        future.wait(None).unwrap();

        let pipeline_dt = std::time::Instant::now() - pipeline_started;
        println!("Pipeline took {} us", pipeline_dt.as_micros());

        pe_out.centeroid();
    }

    Ok(())
}
