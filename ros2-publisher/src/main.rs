mod pipeline;

use r2r::QosProfile;
use std::time::Duration;
use structopt::StructOpt;
use tokio::sync::mpsc;
use turbojpeg::{Compressor, Image, PixelFormat};

#[derive(StructOpt, Debug)]
#[structopt(name = "basic")]
struct Opt {
    /// Be verbose
    #[structopt(short, long)]
    verbose: bool,

    /// Transmits the camera image with a crosshair.
    /// Images are compressed via libjpegturbo.
    /// WARNING: This may generate a lot of data!
    #[structopt(short, long)]
    transmit_image: bool,

    /// Compression quality.
    /// Default: 70.
    #[structopt(short, long, default_value = "70")]
    compressor_quality: i32,

    /// Lock timeout in ms.
    /// Default: 1000.
    #[structopt(short, long, default_value = "1000")]
    lock_timeout: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    // mpsc
    let (cv_point3_tx, mut cv_point3_rx) = mpsc::unbounded_channel::<pipeline::Point3>();
    let (cv_image_tx, mut cv_image_rx) = mpsc::unbounded_channel::<pipeline::Image>();

    // ros setup
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "vkcv", "")?;

    // publishers
    let point_pub =
        node.create_publisher::<pipeline::Point3>("~/local_point", QosProfile::default())?;

    let image_pub = node.create_publisher::<pipeline::RosImageCompressed>(
        "~/camera_image/compressed",
        QosProfile::default(),
    )?;

    let lock_pub = node.create_publisher::<pipeline::Bool>("~/lock", QosProfile::default())?;

    // spinner
    let ros_handle = tokio::task::spawn_blocking(move || loop {
        node.spin_once(Duration::from_millis(100));
        std::thread::sleep(Duration::from_millis(10));
    });

    // setup vkcv
    let cv_config = pipeline::Config {
        transmit_image: opt.transmit_image,
        verbose: opt.verbose,
        ..Default::default()
    };

    let _vkcv_handle = tokio::task::spawn_blocking(move || {
        // vkcv loop
        pipeline::process_blocking(cv_config, cv_point3_tx, cv_image_tx);
    });

    // setup jpeg compressor
    let mut compressor = Compressor::new()?;
    compressor.set_quality(opt.compressor_quality);

    // heartbeat
    let mut lock_ticker = tokio::time::interval(std::time::Duration::from_millis(250));

    // main loop
    let _main_handle = tokio::task::spawn(async move {
        let mut last_seen = None;

        loop {
            tokio::select! {
                _ = lock_ticker.tick() => {
                    let has_lock = if let Some(last_seen) = last_seen {
                        std::time::Instant::now() - last_seen < std::time::Duration::from_millis(opt.lock_timeout)
                    } else {
                        false
                    };

                    lock_pub.publish(&pipeline::Bool {
                        data: has_lock
                    }).unwrap();

                    if opt.verbose {
                        println!("Has lock: {}", has_lock);
                    }
                },
                Some(msg) = cv_point3_rx.recv() => {
                    last_seen = Some(std::time::Instant::now());
                    point_pub.publish(&msg).unwrap();
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

                        image_pub.publish(&ros_img_msg).unwrap();
                    }
                }
            }
        }
    });

    ros_handle.await?;

    Ok(())
}
