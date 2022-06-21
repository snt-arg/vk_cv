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
    #[structopt(short, long, default_value = "110")]
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
    let (cv_point3_tx, mut cv_point3_rx) = mpsc::unbounded_channel::<pipeline::Point3>();
    let (cv_image_tx, mut cv_image_rx) = mpsc::unbounded_channel::<pipeline::Image>();
    let (cv_depth_image_tx, mut cv_depth_image_rx) = mpsc::unbounded_channel::<pipeline::Image>();

    // ros setup
    rosrust::init("vkcv");

    // publishers
    let point_pub = rosrust::publish::<pipeline::Point3>("/vkcv/local_point", 1)?;

    let image_pub =
        rosrust::publish::<pipeline::RosImageCompressed>("/vkcv/camera_image/compressed", 1)?;

    let depth_image_pub =
        rosrust::publish::<pipeline::RosImageCompressed>("/vkcv/camera_depth_image/compressed", 1)?;

    let lock_pub = rosrust::publish::<pipeline::Bool>("/vkcv/lock", 1)?;

    let (exit_tx, exit_rx) = std::sync::mpsc::channel();

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
        match pipeline::process_blocking(
            cv_config,
            cv_point3_tx,
            cv_image_tx,
            cv_depth_image_tx,
            exit_rx,
        ) {
            Err(_) => println!("Cannot open camera"),
            _ => (),
        }
    });

    // setup jpeg compressor
    let mut compressor = Compressor::new()?;
    compressor.set_quality(opt.compressor_quality);

    // heartbeat
    let mut lock_ticker = tokio::time::interval(std::time::Duration::from_millis(250));

    // publishing thread
    let exit_tx_main = exit_tx.clone();
    let main_handle = tokio::task::spawn(async move {
        let mut last_seen = None;

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
                Some(msg) = cv_point3_rx.recv() => {
                    last_seen = Some(std::time::Instant::now());
                    point_pub.send(msg).expect("Failed to send '~/local_point'");
                    // TODO: publish point in vehicle frame
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
                    exit_tx_main.send(true).unwrap();
                    return; // exit thread
                }
            }
        }
    });

    // keep running (blocking)
    rosrust::spin();
    println!("exit ros node, wait for threads to finish...");
    exit_tx.send(true).unwrap();
    tokio::join!(main_handle, vkcv_handle);
    println!("exit threads");

    Ok(())
}
