use vkcv::{
    endpoints::{image_download::ImageDownload, image_upload::ImageUpload},
    processing_elements::{
        color_filter::ColorFilter,
        hsvconv::Hsvconv,
        input::Input,
        morphology::{Morphology, Operation},
        output::Output,
        tracker::{PoolingStrategy, Tracker},
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
    // https://docs.mesa3d.org/drivers/vc4.html
    //
    // Performance/debugging use: V3D_DEBUG=perf
    // resp. VC4_DEBUG=perf on the RPi3

    println!("Realsense camera tracker");

    // set the default display, otherwise we fallback to llvmpipe
    std::env::set_var("DISPLAY", ":0");
    std::env::set_var("V3D_DEBUG", "perf");

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
    let mut pe_hsv_filter = ColorFilter::new([0.20, 0.4, 0.239], [0.429, 1.0, 1.0]);
    let mut pe_erode = Morphology::new(Operation::Erode);
    let mut pe_dilate = Morphology::new(Operation::Dilate);
    let mut pe_tracker = Tracker::new(PoolingStrategy::PreferPooling4, false);
    let mut pe_out = Output::new();

    let (pipeline_cb, input_io, output_io) = cv_pipeline(
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

    let upload = ImageUpload::new(input_io);
    let download = ImageDownload::new(output_io);

    let mut avg_pipeline_execution_duration = std::time::Duration::ZERO;

    // train
    for i in 0..30 {
        // process on GPU
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), pipeline_cb.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        let pipeline_committed = std::time::Instant::now();

        // wait till finished
        future.wait(None).unwrap();

        if i >= 10 {
            if avg_pipeline_execution_duration.is_zero() {
                avg_pipeline_execution_duration = std::time::Instant::now() - pipeline_committed;
            }

            avg_pipeline_execution_duration = std::time::Duration::from_secs_f32(
                avg_pipeline_execution_duration.as_secs_f32() * 0.9
                    + 0.1 * (std::time::Instant::now() - pipeline_committed).as_secs_f32(),
            );
        }
    }

    println!(
        "Average duration: {}",
        avg_pipeline_execution_duration.as_millis()
    );

    loop {
        let image = realsense.fetch_image();
        // println!("{} x {}", image.width(), image.height());
        let pipeline_started = std::time::Instant::now();

        // upload image to GPU
        upload.copy_input_data(image.data_slice());

        // process on GPU
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), pipeline_cb.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        // wait till finished
        std::thread::sleep(avg_pipeline_execution_duration);
        future.wait(None).unwrap();

        let pipeline_dt = std::time::Instant::now() - pipeline_started;
        let c = download.centroid();
        println!(
            "Pipeline flushed: {} ms, coords ({},{})",
            pipeline_dt.as_millis(),
            c[0],
            c[1]
        );
    }

    Ok(())
}
