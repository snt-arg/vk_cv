use vkcv::{
    endpoints::{image_download::ImageDownload, image_upload::ImageUpload},
    processing_elements::{
        color_filter::ColorFilter,
        convolution::Convolution,
        convolution_2p::Convolution2Pass,
        grayscale::Grayscale,
        hsvconv::Hsvconv,
        input::Input,
        morphology::{Morphology, Operation},
        output::Output,
        tracker::{PoolingStrategy, Tracker},
    },
    utils::{cv_pipeline, cv_pipeline_debug, load_image},
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

    std::env::set_var("DISPLAY", ":0");
    std::env::set_var("V3D_DEBUG", "perf");

    let (img_info, img_data) = load_image("tracking_3_1px.png");

    // init device
    let (device, mut queues) = vk_init::init();
    let queue = queues.next().unwrap();

    // create a convolution pipeline
    let pe_input = Input::new(img_info);
    let pe_gsc = Grayscale::new();

    let pe_hsv = Hsvconv::new();
    let pe_hsv_filter = ColorFilter::new([0.20, 0.4, 0.239], [0.429, 1.0, 1.0]);

    // let pe_conv = Convolution::new(device.clone(), queue.clone(), &pe_hsv_filter);
    // let pe_conv_2p = Convolution2Pass::new(device.clone(), queue.clone(), &pe_gsc);
    let pe_erode = Morphology::new(Operation::Erode);
    let pe_dilate = Morphology::new(Operation::Dilate);
    let pe_tracker = Tracker::new(PoolingStrategy::SampledPooling4, false);
    let pe_out = Output::new();

    let dp = cv_pipeline_debug(
        device.clone(),
        queue.clone(),
        &pe_input,
        &[
            &pe_hsv,
            &pe_hsv_filter,
            /*&pe_erode, &pe_dilate,*/ &pe_tracker,
        ],
        &pe_out,
    );

    let upload = ImageUpload::new(dp.input.clone());
    let download = ImageDownload::new(dp.output.clone());

    // let color_image = realsense.fetch_image();
    //println!("{} x {}", color_image.width(), color_image.height());
    let pipeline_started = std::time::Instant::now();

    // upload image to GPU
    upload.copy_input_data(&img_data);

    // process on GPU & wait till finished
    dp.dispatch(device.clone(), queue.clone());
    dp.time(device.clone(), queue.clone());

    // save images
    dp.save_all("pipeline");

    let pipeline_dt = std::time::Instant::now() - pipeline_started;
    println!("Pipeline took {} us", pipeline_dt.as_micros());

    download.save_output_buffer("out_0.png");
    dbg!(download.centroid());

    Ok(())
}
