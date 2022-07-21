mod msg;
mod pipeline;

use structopt::StructOpt;
use tokio::{signal, sync::mpsc};
use turbojpeg::{Compressor, Image, PixelFormat};

#[derive(StructOpt, Debug)]
#[structopt(name = "ros1-publisher")]
struct Opt {
    /// Be verbose
    #[structopt(short, long)]
    verbose: bool,

    /// Transmits the camera image with a crosshair.
    /// Images are compressed via libjpegturbo.
    /// WARNING: This may generate a lot of data!
    #[structopt(short, long)]
    transmit_image: bool,

    /// Transmit depth image.
    #[structopt(long)]
    transmit_depth_image: bool,

    /// Transmit unprocessed (yet compressed) color image.
    #[structopt(long)]
    raw_color_image: bool,

    /// Compression quality.
    #[structopt(short, long, default_value = "60")]
    compressor_quality: i32,

    /// Lock timeout in ms.
    #[structopt(short, long, default_value = "1000")]
    lock_timeout: u64,

    /// The smallest area in pixels required by the detector. Smaller areas will be ignored.
    #[structopt(short, long, default_value = "16")]
    min_area: u32,

    /// Roslaunch adds some special args
    /// e.g. __name:=... __log:=...
    #[structopt(name = "__ros_args", default_value = "")]
    _rargs: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    // mpsc
    let (cv_point3_tx, mut cv_point3_rx) = mpsc::channel::<pipeline::Point3>(1);
    let (cv_image_tx, mut cv_image_rx) = mpsc::channel::<pipeline::Image>(1);
    let (cv_depth_image_tx, mut cv_depth_image_rx) = mpsc::channel::<pipeline::Image>(1);

    let (ros_color_image_tx, ros_color_image_rx) = mpsc::channel::<msg::sensor_msgs::Image>(1);
    let (ros_depth_image_tx, ros_depth_image_rx) = mpsc::channel::<msg::sensor_msgs::Image>(1);
    let (ros_camera_info_tx, mut ros_camera_info_rx) =
        mpsc::channel::<msg::sensor_msgs::CameraInfo>(1);
    let (mavros_pose_tx, mut mavros_pose_rx) = mpsc::channel::<msg::geometry_msgs::PoseStamped>(1);

    // ros setup
    rosrust::init("vkcv");

    // pose
    let uav_pose = std::sync::Mutex::new(msg::geometry_msgs::PoseStamped::default());

    // publishers
    let point_pub = rosrust::publish::<pipeline::Point3>("/vkcv/local_point", 1)?;
    let world_point_pub = rosrust::publish::<pipeline::Point3>("/vkcv/point", 1)?;

    let image_pub =
        rosrust::publish::<pipeline::RosImageCompressed>("/vkcv/camera_image/compressed", 1)?;

    let depth_image_pub =
        rosrust::publish::<pipeline::RosImageCompressed>("/vkcv/camera_depth_image/compressed", 1)?;

    let lock_pub = rosrust::publish::<pipeline::Bool>("/vkcv/lock", 1)?;

    // setup vkcv
    let cv_config = pipeline::Config {
        transmit_image: opt.transmit_image,
        transmit_depth_image: opt.transmit_depth_image,
        process_image: !opt.raw_color_image,
        verbose: opt.verbose,
        min_area: opt.min_area,
        ..Default::default()
    };

    // vkcv processing thread
    let vkcv_handle = tokio::task::spawn_blocking(move || {
        let ros_camera_info = ros_camera_info_rx.blocking_recv().unwrap();

        pipeline::process_blocking(
            cv_config,
            cv_point3_tx,
            cv_image_tx,
            cv_depth_image_tx,
            ros_color_image_rx,
            ros_depth_image_rx,
            ros_camera_info,
        )
    });

    // setup jpeg compressor
    let mut compressor = Compressor::new()?;
    compressor.set_quality(opt.compressor_quality);

    // heartbeat
    let mut lock_ticker = tokio::time::interval(std::time::Duration::from_millis(250));

    // publishing thread
    let main_handle = tokio::task::spawn(async move {
        let mut last_seen = None;

        // subscribers
        // declare them here such that they go out of scope when exiting this task
        // (the channels are subsequently closed)
        let _color_img_sub = rosrust::subscribe(
            "/camera/color/image_raw",
            1,
            move |msg: msg::sensor_msgs::Image| {
                ros_color_image_tx.try_send(msg).ok();
            },
        )
        .unwrap();
        let _depth_img_sub = rosrust::subscribe(
            "/camera/depth/image_raw",
            1,
            move |msg: msg::sensor_msgs::Image| {
                ros_depth_image_tx.try_send(msg).ok();
            },
        )
        .unwrap();
        let _cam_info_sub = rosrust::subscribe(
            "/camera/color/camera_info",
            1,
            move |msg: msg::sensor_msgs::CameraInfo| {
                ros_camera_info_tx.try_send(msg).ok();
            },
        )
        .unwrap();
        let _uav_pose_sub = rosrust::subscribe(
            "/mavros/local_position/pose",
            1,
            move |msg: msg::geometry_msgs::PoseStamped| {
                mavros_pose_tx.try_send(msg).ok();
            },
        )
        .unwrap();

        loop {
            tokio::select! {
                _ = lock_ticker.tick() => {
                    let has_lock = if let Some(last_seen) = last_seen {
                        std::time::Instant::now() - last_seen < std::time::Duration::from_millis(opt.lock_timeout)
                    } else {
                        false
                    };

                    lock_pub.send(pipeline::Bool {
                        data: has_lock
                    }).expect("Failed to send '~/lock'");

                    if opt.verbose {
                        println!("Has lock: {}", has_lock);
                    }

                    if !rosrust::is_ok() {
                        return;
                    }
                },
                Some(pose) = mavros_pose_rx.recv() => {
                    if let Ok(mut lock) = uav_pose.lock() {
                        *lock = pose;
                    }
                }
                Some(msg) = cv_point3_rx.recv() => {
                    last_seen = Some(std::time::Instant::now());

                    // TODO: publish point in vehicle frame
                    if let Ok(pose) = uav_pose.lock() {
                        let pos = &pose.pose.position;
                        let ori = &pose.pose.orientation;

                        let transl = nalgebra::Vector3::new(pos.x, pos.y, pos.z + 0.35);
                        let quat = nalgebra::Quaternion::new(ori.w, ori.x, ori.y, ori.z);
                        let quat = nalgebra::UnitQuaternion::new_unchecked(quat);
                        let wtb = nalgebra::Isometry3::new(transl, quat.scaled_axis());

                        // let quat: nalgebra::UnitQuaternion<f64> = nalgebra::UnitQuaternion::from_euler_angles(0.0, 0.5235987756, -1.570796327);
                        let quat: nalgebra::UnitQuaternion<f64> = nalgebra::UnitQuaternion::from_euler_angles(2.094395102, 0.0, 0.0);
                        let transl = nalgebra::Vector3::new(0.0, 0.22, -0.025);
                        let btc = nalgebra::Isometry3::new(transl, quat.scaled_axis());

                        let point_pos = nalgebra::Point3::new(msg.x, msg.y, msg.z);
                        let point_in_world = (wtb * btc) * point_pos;
                        world_point_pub.send(msg::geometry_msgs::Point {
                            x: point_in_world[0],
                            y: point_in_world[1],
                            z: point_in_world[2]
                        }).expect("Failed to send '~/point'");
                    };

                    point_pub.send(msg).expect("Failed to send '~/local_point'");
                },
                Some(image) = cv_image_rx.recv() => {
                    let image = Image {
                        pixels: image.buffer.as_slice(),
                        width: image.info.width as usize,
                        pitch: image.info.stride() as usize,
                        height: image.info.height as usize,
                        format: PixelFormat::RGBA,
                    };
                    if let Ok(jpeg_data) = compressor.compress_to_vec(image) {
                        let ros_img_msg = pipeline::RosImageCompressed {
                            format: "jpeg".to_string(),
                            data: jpeg_data,
                            ..Default::default()
                        };

                        image_pub.send(ros_img_msg).expect("Failed to send '~/camera_image/compressed'");
                    }
                }
                Some(image) = cv_depth_image_rx.recv() => {
                    let image = Image {
                        pixels: image.buffer.as_slice(),
                        width: image.info.width as usize,
                        pitch: image.info.stride() as usize,
                        height: image.info.height as usize,
                        format: PixelFormat::RGB,
                    };

                    if let Ok(jpeg_data) = compressor.compress_to_vec(image) {
                        let ros_img_msg = pipeline::RosImageCompressed {
                            format: "jpeg".to_string(),
                            data: jpeg_data,
                            ..Default::default()
                        };
                        depth_image_pub.send(ros_img_msg).expect("Failed to send '~/camera_depth_image/compressed'");
                    }
                }
                Ok(_) = signal::ctrl_c() => {
                    println!("interrupt");
                    rosrust::shutdown();
                    break; // exit thread
                }
            }
        }
        println!("exit main loop");
    });

    tokio::join!(main_handle, vkcv_handle);
    println!("shutdown");

    Ok(())
}
