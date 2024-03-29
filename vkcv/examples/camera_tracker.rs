use std::io::Write;

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
    utils::{self, ImageInfo},
    utils::{cv_pipeline_sequential, cv_pipeline_sequential_debug},
    vk_init,
};

use anyhow::Result;
use vulkano::sync::{self, GpuFuture};

use sysinfo::{Pid, PidExt, ProcessExt, System, SystemExt};

const DBG_PROFILE: bool = false;

fn main() -> Result<()> {
    // setup sysinfo
    let mut sys = System::new();
    sys.refresh_processes();

    let mut last_stats = std::time::Instant::now();

    // open histogram file
    let hist_file =
        std::fs::File::create(format!("{}/hist.csv", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let mut hist_buf = std::io::BufWriter::new(hist_file);
    hist_buf
        .write_all(&"frame,pipeline_time,fps,cpu\n".as_bytes())
        .unwrap();

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
    //std::env::set_var("DISPLAY", ":0");
    //std::env::set_var("V3D_DEBUG", "perf");

    // depth resolutions
    //let res = [848, 480];
    let res = [640, 480];
    //let res = [480, 270];
    let fps = 30;
    let mut camera = Realsense::open(&res, fps, &res, fps).unwrap();

    // camera.dump_intrinsic(Some((640, 480)));

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
    let pe_hsv_filter = ColorFilter::new([0.20, 0.4, 0.239], [0.429, 1.0, 1.0]);
    let pe_erode = Morphology::new(Operation::Erode);
    let pe_dilate = Morphology::new(Operation::Dilate);
    let pe_pooling = Pooling::new(pooling::Operation::Max); // 2x2
    let pe_tracker = Tracker::new(PoolingStrategy::Pooling4, Canvas::Pad);
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

    let pipeline_dbg = cv_pipeline_sequential_debug(
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
        "Average duration: {}",
        avg_pipeline_execution_duration.as_millis()
    );

    let start_of_program = std::time::Instant::now();
    let mut frame = 0u32;
    let mut last_frame = 0;

    loop {
        // grab depth and color image from the realsense
        let (color_image, depth_image) = camera.fetch_image(true);

        // time
        let pipeline_started = std::time::Instant::now();

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
        future.wait(None).unwrap(); // spin-lock?

        // get processed depth image
        let depth_image = depth_image.get();

        // print results
        let pipeline_dt = std::time::Instant::now() - pipeline_started;
        let tf_image = download.transfer();
        let (c, area) = tracker::centroid(&tf_image);
        let area_px = (area * color_image.area() as f32) as u32;

        if DBG_PROFILE {
            println!(
                "[{}] Pipeline flushed: {} ms, coords [{:.2},{:.2}] ({} px²)",
                frame,
                pipeline_dt.as_millis(),
                c[0],
                c[1],
                area_px
            );
        }

        // get the depth only if our object is bigger than 225px² (15x15)
        if area_px > 16 {
            let pixel_coords = [
                c[0] * color_image.width() as f32,
                c[1] * color_image.height() as f32,
            ];
            let depth = camera.depth_at_pixel(&pixel_coords, &color_image, &depth_image);

            // de-project to obtain a 3D point in camera coordinates
            if let Some(depth) = depth {
                let point = camera.deproject_pixel(&pixel_coords, depth, &color_image);
                println!(
                    "[{}] Target located at: [{:.2},{:.2},{:.2}] ({} px²)",
                    frame, point[0], point[1], point[2], area_px,
                );
            }
            //dbg!(c);

            // paint the centroid
            if DBG_PROFILE && frame % 15 == 0 {
                let mut owned_image = OwnedImage {
                    buffer: color_image.data_slice().to_vec(),
                    info: ImageInfo {
                        width: color_image.width(),
                        height: color_image.height(),
                        format: vulkano::format::Format::R8G8B8A8_UINT,
                    },
                };
                draw_centroid(&mut owned_image, &pixel_coords, 2.0);
                utils::write_image(
                    &format!("out/centroid-{}", frame),
                    &owned_image.buffer,
                    &img_info,
                );
            }
        }

        // debug
        // break down the cost of the individual stages
        if DBG_PROFILE && frame % 30 == 0 {
            // time the execution of the individual stages
            pipeline_dbg.time(&ctx);

            // save a snapshot of all stages in the pipeline
            let upload = ImageUpload::from_io(pipeline_dbg.input.clone()).unwrap();
            upload.copy_input_data(color_image.data_slice());
            let prefix = std::time::Instant::now().duration_since(start_of_program);
            pipeline_dbg.save_all(&ctx, "out", &format!("{}-", prefix.as_millis()));
        }

        // print stats
        let pid = Pid::from_u32(std::process::id());
        let mut cpu_usage = f32::NAN;
        let mut fps = f32::NAN;
        if std::time::Instant::now() - last_stats > std::time::Duration::from_millis(500) {
            last_stats = std::time::Instant::now();

            if sys.refresh_process(pid) {
                let proc = sys.processes().get(&pid).unwrap();
                cpu_usage = proc.cpu_usage();
            }

            fps = (frame - last_frame) as f32 / std::time::Duration::from_millis(500).as_secs_f32();
            last_frame = frame;

            println!("fps: {:.0}, cpu: {:.0}%", fps, cpu_usage);
        }

        hist_buf
            .write_all(
                &format!(
                    "{},{},{},{}\n",
                    frame,
                    pipeline_dt.as_secs_f32(),
                    fps,
                    cpu_usage
                )
                .as_bytes(),
            )
            .unwrap();

        frame += 1;
    }
}
