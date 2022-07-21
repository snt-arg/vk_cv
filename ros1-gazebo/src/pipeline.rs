use rosrust::Duration;
use tokio::sync::mpsc::{Receiver, Sender};

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
    realsense::{ColorFrame, Realsense},
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

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

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
            hsv_min: [0.138, 0.6, 0.239],
            hsv_max: [0.198, 1.0, 1.0],
            min_area: 4 * 4,
            transmit_image: false,
            transmit_depth_image: false,
            process_image: true,
            verbose: false,
        }
    }
}

pub fn process_blocking(
    config: Config,
    sender_point3: Sender<Point3>,
    sender_image: Sender<OwnedImage>,
    sender_depth_image: Sender<OwnedImage>,
    mut ros_color_image: Receiver<msg::sensor_msgs::Image>,
    mut ros_depth_image: Receiver<msg::sensor_msgs::Image>,
    camera_info: msg::sensor_msgs::CameraInfo,
) -> anyhow::Result<()> {
    println!("CV: Realsense camera tracker");

    // init device
    let (device, queue) = vk_init::init();

    // get image info
    let img_info = ImageInfo {
        width: camera_info.width,
        height: camera_info.height,
        format: vkcv::utils::Format::R8G8B8_UINT,
    };

    // get image info
    let pipeline_info = ImageInfo {
        width: camera_info.width,
        height: camera_info.height,
        format: vkcv::utils::Format::R8G8B8A8_UNORM,
    };

    // and projection matrix K
    let camera_proj = nalgebra::Matrix3x4::from_row_slice(&camera_info.P);

    // create a color tracking pipeline
    let pe_input = Input::new(pipeline_info);
    let pe_hsv = Hsvconv::new();
    let pe_hsv_filter = ColorFilter::new(config.hsv_min, config.hsv_max);
    let pe_erode = Morphology::new(Operation::Erode);
    let pe_dilate = Morphology::new(Operation::Dilate);
    let pe_tracker = Tracker::new(PoolingStrategy::Pooling4, Canvas::Pad);
    let pe_pooling = Pooling::new(pooling::Operation::Max); // 2x2
    let pe_out = Output::new();

    let (pipeline_cb, input_io, output_io) = cv_pipeline_sequential(
        device.clone(),
        queue.clone(),
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

    println!("CV: Entering main loop");
    while rosrust::is_ok() {
        // grab depth and color image from the gazebo realsense
        let color_frame =
            vkcv::utils::rgb8_to_rgba8(&img_info, &ros_color_image.blocking_recv().unwrap().data);
        let depth_frame = ros_depth_image.blocking_recv().unwrap();

        // upload image to GPU
        upload.copy_input_data(&color_frame.1);

        // process on GPU
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), pipeline_cb.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        // wait till finished
        future.wait(None).unwrap(); // spin-lock

        // print results
        let (c, area) = tracker::centroid(&download.transfer());
        let area_px = (area * color_frame.0.area() as f32) as u32;

        //println!("got frame {}, {}", area_px, config.min_area);

        // owned image
        let mut owned_image = OwnedImage {
            buffer: color_frame.1.clone(),
            info: ImageInfo {
                width: color_frame.0.width,
                height: color_frame.0.height,
                format: vulkano::format::Format::R8G8B8A8_UINT,
            },
        };

        // get the depth only if our object is bigger than 225px² (15x15)
        if area_px > config.min_area {
            let pixel_coords = [
                c[0] * color_frame.0.width as f32,
                c[1] * color_frame.0.height as f32,
            ];

            let stride = depth_frame.width * 2; // u16
            let offset = pixel_coords[1] as u32 * stride + pixel_coords[0] as u32 * 2;

            let depth = if offset < depth_frame.data.len() as u32 {
                let depth_u16 = u16::from_ne_bytes([
                    depth_frame.data[(offset) as usize],
                    depth_frame.data[(offset + 1) as usize],
                ]);

                Some(depth_u16 as f32 / 1000.0 as f32 + 0.1)
            } else {
                None
            };

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
                // http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/CameraInfo.html
                let fx = *camera_proj.get((0, 0)).unwrap() as f32;
                let fy = *camera_proj.get((1, 1)).unwrap() as f32;
                let cx = *camera_proj.get((0, 2)).unwrap() as f32;
                let cy = *camera_proj.get((1, 2)).unwrap() as f32;

                // println!("{}, cx {} cy {}, fx {}, fy {}", camera_proj, cx, cy, fx, fy);

                // reproject
                let z = depth;
                let x = (pixel_coords[0] - cx) / fx * z;
                let y = (pixel_coords[1] - cy) / fy * z;

                let point = [-x, -y, z];

                // ignore this measurement if we hit a hole in the depth image
                if point[2] > 0.0 && rosrust::is_ok() {
                    sender_point3.blocking_send(Point3 {
                        x: point[0] as f64,
                        y: point[1] as f64,
                        z: point[2] as f64,
                    })?;
                }
            }
        }

        // send image
        if config.transmit_image {
            sender_image.blocking_send(owned_image)?;
        }

        // send depth image
        if config.transmit_depth_image {
            // convert from monochrome to rgb format
            let owned_image = OwnedImage {
                buffer: depth_frame.data.clone(),
                info: ImageInfo {
                    width: img_info.width,
                    height: img_info.height,
                    format: vulkano::format::Format::R8G8B8_UINT,
                },
            };

            sender_depth_image.blocking_send(owned_image)?;
        }
    }

    Ok(())
}
