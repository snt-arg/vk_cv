use std::{error::Error, sync::mpsc::Receiver};

use crate::msg;
use vkcv::{
    draw::{draw_centroid, OwnedImage},
    endpoints::{image_download::ImageDownload, image_upload::ImageUpload},
    processing_elements::{
        color_filter::ColorFilter,
        hsvconv::Hsvconv,
        input::Input,
        morphology::{Morphology, Operation},
        output::Output,
        pooling::{self, Pooling},
        tracker::{self, Canvas, PoolingStrategy, Tracker},
    },
    realsense::Realsense,
    utils::{cv_pipeline_sequential, ImageInfo},
    vk_init,
    vulkano::{
        self,
        sync::{self, GpuFuture},
    },
};

pub type Point3 = msg::geometry_msgs::Point;
pub type Bool = msg::std_msgs::Bool;
pub type Image = OwnedImage;
pub type RosImageCompressed = msg::sensor_msgs::CompressedImage;

use tokio::sync::mpsc::UnboundedSender;

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub hsv_min: [f32; 3],
    pub hsv_max: [f32; 3],
    pub min_area: u32,
    pub transmit_image: bool,
    pub transmit_depth_image: bool,
    pub process_image: bool,
    pub verbose: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hsv_min: [0.3, 0.6, 0.239],
            hsv_max: [0.5, 1.0, 1.0],
            min_area: 4 * 4,
            transmit_image: false,
            transmit_depth_image: false,
            process_image: true,
            verbose: false,
        }
    }
}

pub async fn process_blocking(
    config: Config,
    sender_point3: UnboundedSender<Point3>,
    sender_image: UnboundedSender<OwnedImage>,
    sender_depth_image: UnboundedSender<OwnedImage>,
    exit_signal: Receiver<bool>,
) -> Result<(), Box<dyn Error + Sync + Send>> {
    println!("CV: Realsense camera tracker");

    // set the default display, otherwise we fallback to llvmpipe
    // std::env::set_var("DISPLAY", ":0");
    // std::env::set_var("V3D_DEBUG", "perf");

    let resolution = [640, 480];
    let target_fps = 30;
    println!(
        "CV: Opening camera ({}x{}@{}fps)",
        resolution[0], resolution[1], target_fps
    );
    let mut camera = Realsense::open(&resolution, target_fps, &resolution, target_fps)?;

    // grab a couple of frames
    for _ in 0..5 {
        camera.fetch_image(false);
    }

    let img_info = camera.fetch_image(false).0.image_info();

    // init device
    let ctx = vk_init::init().unwrap();

    // create a color tracking pipeline
    let pe_input = Input::new(img_info);
    let pe_hsv = Hsvconv::new();
    let pe_hsv_filter = ColorFilter::new(config.hsv_min, config.hsv_max);
    let pe_erode = Morphology::new(Operation::Erode);
    let pe_dilate = Morphology::new(Operation::Dilate);
    let pe_tracker = Tracker::new(PoolingStrategy::Pooling4, Canvas::Pad);
    let pe_pooling = Pooling::new(pooling::Operation::Max); // 2x2
    let pe_out = Output::new();

    let (pipeline_cb, input_io, output_io) = cv_pipeline_sequential(
        &ctx,
        &pe_input,
        &[
            &pe_hsv,
            &pe_hsv_filter,
            &pe_erode,
            &pe_dilate,
            &pe_pooling,
            &pe_tracker,
        ],
        &pe_out,
    );

    let upload = ImageUpload::from_io(input_io).unwrap();
    let mut download = ImageDownload::from_io(output_io).unwrap();

    let mut avg_pipeline_execution_duration = std::time::Duration::ZERO;

    // train
    for i in 0..30 {
        // process on GPU
        let future = sync::now(ctx.device.clone())
            .then_execute(ctx.queue.clone(), pipeline_cb.clone())
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
        let (color_image, depth_image) = camera.fetch_image(true);

        // upload image to GPU
        upload.copy_input_data(color_image.data_slice());

        // process on GPU
        let future = sync::now(ctx.device.clone())
            .then_execute(ctx.queue.clone(), pipeline_cb.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        // wait till finished
        std::thread::sleep(avg_pipeline_execution_duration); // the results are likely ready after we wake up
        future.await?; // spin-lock?

        // print results
        let (c, area) = tracker::centroid(&download.transfer());
        let area_px = (area * color_image.area() as f32) as u32;

        // owned image
        let mut owned_image = OwnedImage {
            buffer: color_image.data_slice().to_vec(),
            info: ImageInfo {
                width: color_image.width(),
                height: color_image.height(),
                format: vulkano::format::Format::R8G8B8A8_UINT,
            },
        };

        // get actual depth image with holes filled
        let depth_image = depth_image.get();

        // get the depth only if our object is bigger than 225px² (15x15)
        if area_px > config.min_area {
            let pixel_coords = [
                c[0] * color_image.width() as f32,
                c[1] * color_image.height() as f32,
            ];
            let depth = camera.depth_at_pixel(&pixel_coords, &color_image, &depth_image);

            if config.verbose {
                println!(
                    "px coords {}, {}\tdepth {:?}m",
                    pixel_coords[0], pixel_coords[1], depth
                );
            }

            // draw centroid
            if config.process_image {
                draw_centroid(&mut owned_image, &pixel_coords);
            }

            // de-project to obtain a 3D point in camera coordinates
            if let Some(depth) = depth {
                let point = camera.deproject_pixel(&pixel_coords, depth, &color_image);

                // ignore this measurement if we hit a hole in the depth image
                if point[2] > 0.0 && rosrust::is_ok() {
                    sender_point3
                        .send(Point3 {
                            x: point[0] as f64,
                            y: point[1] as f64,
                            z: point[2] as f64,
                        })
                        .unwrap();
                }
            }
        }

        // send image
        if config.transmit_image {
            if rosrust::is_ok() {
                sender_image
                    .send(owned_image)
                    .expect("Failed to transmit image");
            }
        }

        // send depth image
        if config.transmit_depth_image {
            // convert from monochrome to rgb format
            let owned_image = OwnedImage {
                buffer: depth_image.to_owned_rgb(),
                info: ImageInfo {
                    width: depth_image.width(),
                    height: depth_image.height(),
                    format: vulkano::format::Format::R8G8B8_UINT,
                },
            };

            if rosrust::is_ok() {
                sender_depth_image
                    .send(owned_image)
                    .expect("Failed to transmit image");
            }
        }

        if let Ok(_) = exit_signal.try_recv() {
            println!("exit camera thread");
            return Ok(());
        }
    }
}
