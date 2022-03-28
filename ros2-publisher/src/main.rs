mod pipeline;

use r2r::QosProfile;
use std::time::Duration;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // mpsc
    let (cv_point3_tx, mut cv_point3_rx) = mpsc::unbounded_channel::<pipeline::Point3>();
    let (cv_image_tx, mut cv_image_rx) = mpsc::unbounded_channel::<pipeline::RosImage>();

    // ros setup
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "vkcv", "")?;

    // publishers
    let point_pub =
        node.create_publisher::<pipeline::Point3>("/camera_point", QosProfile::default())?;
    let local_point_pub =
        node.create_publisher::<pipeline::Point3>("/camera_local_point", QosProfile::default())?;

    let image_pub =
        node.create_publisher::<pipeline::RosImage>("/camera_image", QosProfile::default())?;

    // spinner
    let ros_handle = tokio::task::spawn_blocking(move || loop {
        node.spin_once(Duration::from_millis(100));
        std::thread::sleep(Duration::from_millis(10));
    });

    // setup vkcv
    let cv_config = pipeline::Config::default();

    let _vkcv_handle = tokio::task::spawn_blocking(move || {
        // vkcv loop
        pipeline::process_blocking(cv_config, cv_point3_tx, cv_image_tx);
    });

    // main loop
    let _main_handle = tokio::task::spawn(async move {
        loop {
            while let Ok(msg) = cv_point3_rx.try_recv() {
                point_pub.publish(&msg).unwrap();
                // TODO: publish point in vehicle frame
            }
            while let Ok(msg) = cv_image_rx.try_recv() {
                image_pub.publish(&msg).unwrap();
                // TODO: publish point in vehicle frame
            }
        }
    });

    ros_handle.await?;

    Ok(())
}
