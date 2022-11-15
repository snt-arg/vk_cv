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
        pooling::{self, Pooling},
        tracker::{self, Canvas, Tracker},
    },
    utils::{cv_pipeline_sequential, cv_pipeline_sequential_debug, load_image},
    vk_init,
};

use anyhow::Result;

fn main() -> Result<()> {
    // let mut realsense = Realsense::open();

    // v3d specs/properties:
    //
    // maxComputeWorkGroupSize: 256
    // maxImageDimension: 4096
    // maxPushConstantsSize: 128
    //
    // https://vulkan.gpuinfo.org/displayreport.php?id=13073#properties

    std::env::set_var("DISPLAY", ":0");
    std::env::set_var("V3D_DEBUG", "perf");

    let (img_info, img_data) = load_image("lab_image_2_rgba.png");

    // init device
    let ctx = vk_init::init().unwrap();

    // create a convolution pipeline
    let pe_input = Input::new(img_info);
    let pe_gsc = Grayscale::new();

    let pe_hsv = Hsvconv::new();
    let pe_hsv_filter = ColorFilter::new([0.3, 0.6, 0.239], [0.5, 1.0, 1.0]);
    let pe_pooling = Pooling::new(pooling::Operation::Max);
    // let pe_conv = Convolution::new(device.clone(), queue.clone(), &pe_hsv_filter);
    // let pe_conv_2p = Convolution2Pass::new(device.clone(), queue.clone(), &pe_gsc);
    let pe_erode = Morphology::new(Operation::Erode);
    let pe_dilate = Morphology::new(Operation::Dilate);
    let pe_tracker = Tracker::new(tracker::PoolingStrategy::Pooling4, Canvas::Pad);
    let pe_out = Output::new();

    let dp = cv_pipeline_sequential_debug(
        &ctx,
        &pe_input,
        &[
            &pe_hsv,
            &pe_hsv_filter,
            &pe_pooling,
            &pe_erode,
            &pe_dilate,
            &pe_tracker,
        ],
        &pe_out,
    );

    let upload = ImageUpload::from_io(dp.input.clone()).unwrap();
    let mut download = ImageDownload::from_io(dp.output.clone()).unwrap();

    // let color_image = realsense.fetch_image();
    //println!("{} x {}", color_image.width(), color_image.height());
    let pipeline_started = std::time::Instant::now();

    // upload image to GPU
    upload.copy_input_data(&img_data);

    // process on GPU & wait till finished
    dp.time(&ctx);

    // save images
    dp.save_all(&ctx, "out", "pipeline-");

    let pipeline_dt = std::time::Instant::now() - pipeline_started;
    println!("Pipeline took {} us", pipeline_dt.as_micros());

    let tf_img = download.transfer();
    tf_img.save_output_buffer("out_0.png");
    dbg!(tracker::centroid(&tf_img));

    Ok(())
}
