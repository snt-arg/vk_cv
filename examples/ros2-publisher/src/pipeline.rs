use vkcv::{
    endpoints::{image_download::ImageDownload, image_upload::ImageUpload},
    processing_elements::{
        color_filter::ColorFilter,
        hsvconv::Hsvconv,
        input::Input,
        morphology::{Morphology, Operation},
        output::Output,
        tracker::{Canvas, PoolingStrategy, Tracker},
    },
    realsense::Realsense,
    utils::cv_pipeline_sequential,
    vk_init,
    vulkano::sync::{self, GpuFuture},
};

pub type Point3 = r2r::geometry_msgs::msg::Point;

use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    hsv_min: [f32; 3],
    hsv_max: [f32; 3],
    min_area: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hsv_min: [0.20, 0.4, 0.239],
            hsv_max: [0.429, 1.0, 1.0],
            min_area: 225,
        }
    }
}

pub fn process_blocking(config: Config, sender: UnboundedSender<Point3>) {
    println!("CV: Realsense camera tracker");

    // set the default display, otherwise we fallback to llvmpipe
    // std::env::set_var("DISPLAY", ":0");
    // std::env::set_var("V3D_DEBUG", "perf");

    println!("CV: Opening camera...");
    let mut camera = Realsense::open(&[640, 480], 30, &[640, 480], 30).unwrap();

    // grab a couple of frames
    for _ in 0..5 {
        camera.fetch_image();
    }

    let img_info = camera.fetch_image().0.image_info();

    // init device
    let (device, mut queues) = vk_init::init();
    let queue = queues.next().unwrap();

    // create a color tracking pipeline
    let pe_input = Input::new(img_info);
    let pe_hsv = Hsvconv::new();
    let pe_hsv_filter = ColorFilter::new(config.hsv_min, config.hsv_max);
    let pe_erode = Morphology::new(Operation::Erode);
    let pe_dilate = Morphology::new(Operation::Dilate);
    let pe_tracker = Tracker::new(PoolingStrategy::Pooling4, Canvas::Pad);
    let pe_out = Output::new();

    let (pipeline_cb, input_io, output_io) = cv_pipeline_sequential(
        device.clone(),
        queue.clone(),
        &pe_input,
        &[&pe_hsv, &pe_hsv_filter, &pe_erode, &pe_dilate, &pe_tracker],
        &pe_out,
    );

    let upload = ImageUpload::from_io(input_io).unwrap();
    let download = ImageDownload::from_io(output_io).unwrap();

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
        "CV: Average duration: {} ms",
        avg_pipeline_execution_duration.as_millis()
    );

    println!("CV: Entering main loop");
    loop {
        // grab depth and color image from the realsense
        let (color_image, depth_image) = camera.fetch_image();

        // upload image to GPU
        upload.copy_input_data(color_image.data_slice());

        // process on GPU
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), pipeline_cb.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        // wait till finished
        std::thread::sleep(avg_pipeline_execution_duration); // the results are likely ready after we wake up
        future.wait(None).unwrap(); // spin-lock?

        // print results
        let (c, area) = download.centroid();
        let area_px = (area * color_image.area() as f32) as u32;

        // get the depth only if our object is bigger than 225px² (15x15)
        if area_px > config.min_area {
            let pixel_coords = [
                c[0] * color_image.width() as f32,
                c[1] * color_image.height() as f32,
            ];
            let depth = camera.depth_at_pixel(&pixel_coords, &color_image, &depth_image);

            // de-project to obtain a 3D point in camera coordinates
            if let Some(depth) = depth {
                let point = camera.deproject_pixel(&pixel_coords, depth, &color_image);

                sender
                    .send(Point3 {
                        x: point[0] as f64,
                        y: point[1] as f64,
                        z: point[2] as f64,
                    })
                    .unwrap();
            }
        }
    }
}