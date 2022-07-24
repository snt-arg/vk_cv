use std::ffi::c_void;
use std::io::Write;
use std::time::Instant;

use opencv::core::CV_8UC4;
use opencv::core::{in_range, Mat_AUTO_STEP, Vec4f, Vec4w, BORDER_DEFAULT, CV_32FC4};
use opencv::highgui;
use opencv::imgproc;
use opencv::prelude::*;
use sysinfo::{Pid, PidExt, ProcessExt, System, SystemExt};
use vkcv::realsense::Realsense;

const SHOW_GUI: bool = false;

fn main() {
    // setup realsense
    let resolution = [640, 480];
    let target_fps = 30;
    println!(
        "CV: Opening camera ({}x{}@{}fps)",
        resolution[0], resolution[1], target_fps
    );
    let mut camera = Realsense::open(&resolution, target_fps, &resolution, target_fps).unwrap();

    // grab a couple of frames
    for _ in 0..5 {
        camera.fetch_image(false);
    }

    // setup sysinfo
    let mut sys = System::new();
    sys.refresh_processes();

    // open histogram file
    let hist_file =
        std::fs::File::create(format!("{}/hist.csv", env!("CARGO_MANIFEST_DIR"))).unwrap();
    let mut hist_buf = std::io::BufWriter::new(hist_file);
    hist_buf
        .write_all(&"frame,pipeline_time,fps,cpu\n".as_bytes())
        .unwrap();

    // ui
    let window = "video capture";
    if SHOW_GUI {
        highgui::named_window(window, highgui::WINDOW_AUTOSIZE).unwrap();
    }

    let mut bgr_image = Mat::default();
    let mut hsv_image = Mat::default();
    let mut rgba_image = Mat::default();
    let mut mask = Mat::default();
    let mut mask_open = Mat::default();
    let mut mask_close = Mat::default();
    let mut reduced_image = Mat::default();
    let mut coordinate_mask = Mat::default();

    let hsv_low = Mat::from_slice(&[66_u8, 75, 61]).unwrap();
    let hsv_up = Mat::from_slice(&[100_u8, 255, 255]).unwrap();

    let mut last_stats = Instant::now();
    let mut frame = 0u32;
    let mut last_frame = 0;

    loop {
        // get camera image
        let (color_frame, depth_frame) = camera.fetch_image(true);

        // wrap it into a OpenCV mat
        let mut color = unsafe {
            Mat::new_rows_cols_with_data(
                color_frame.height() as i32,
                color_frame.width() as i32,
                CV_8UC4,
                color_frame.data_slice().as_ptr() as *mut c_void,
                Mat_AUTO_STEP,
            )
            .unwrap()
            .to_owned()
        };

        // rgba to bgr
        opencv::imgproc::cvt_color(&color, &mut bgr_image, imgproc::COLOR_RGBA2BGR, 0).unwrap();

        // measure time from here
        let pipeline_start = Instant::now();

        // 1. color filter
        opencv::imgproc::cvt_color(&bgr_image, &mut hsv_image, imgproc::COLOR_BGR2HSV, 0).unwrap();
        in_range(&hsv_image, &hsv_low, &hsv_up, &mut mask).unwrap();

        // 2. morphological filtering
        let kernel = opencv::imgproc::get_structuring_element(
            0,
            opencv::core::Size::new(3, 3),
            opencv::core::Point::new(1, 1),
        )
        .unwrap();

        // - close
        imgproc::morphology_ex(
            &mask,
            &mut mask_close,
            imgproc::MORPH_CLOSE,
            &kernel,
            opencv::core::Point::new(1, 1),
            1,
            BORDER_DEFAULT,
            opencv::core::Scalar::new(0.0, 0.0, 0.0, 0.0),
        )
        .unwrap();

        // - open
        imgproc::morphology_ex(
            &mask_close,
            &mut mask_open,
            imgproc::MORPH_OPEN,
            &kernel,
            opencv::core::Point::new(1, 1),
            1,
            BORDER_DEFAULT,
            opencv::core::Scalar::new(0.0, 0.0, 0.0, 0.0),
        )
        .unwrap();

        // 3. coordinate mask
        create_coordinate_mask(&mask_open, &mut coordinate_mask);

        // 4. centroid
        let c = calculate_centroid(&coordinate_mask);

        // depth frame
        let depth_frame = depth_frame.get();

        // print
        let dt = Instant::now() - pipeline_start;
        println!(
            "[{}] Pipeline took {}ms, c [{:.1},{:.1}]",
            frame,
            dt.as_millis(),
            c[0],
            c[1]
        );

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
                &format!("{},{},{},{}\n", frame, dt.as_secs_f32(), fps, cpu_usage).as_bytes(),
            )
            .unwrap();
        hist_buf.flush().unwrap();

        frame += 1;

        // gui
        if SHOW_GUI {
            highgui::imshow(window, &mut coordinate_mask).unwrap();

            let key = highgui::wait_key(10).unwrap();
            if key > 0 && key != 255 {
                break;
            }
        }
    }
}

fn create_coordinate_mask(
    src: &dyn opencv::core::ToInputArray,
    dst: &mut dyn opencv::core::ToOutputArray,
) {
    let input = src.input_array().unwrap().get_mat(-1).unwrap();
    let rows = input.rows();
    let cols = input.cols();

    // create/resize underlying buffer and
    // avoid allocating a new matrix on every call
    if input.size().unwrap()
        != dst
            .output_array()
            .unwrap()
            .get_mat(-1)
            .unwrap()
            .size()
            .unwrap()
    {
        let out_mat = Mat::new_rows_cols_with_default(
            rows,
            cols,
            Vec4f::typ(),
            opencv::core::Scalar::new(0.0, 0.0, 0.0, 0.0),
        )
        .unwrap();

        dst.output_array().unwrap().assign_1(&out_mat).unwrap();
    };

    let mut out_mat = dst.output_array().unwrap().get_mat(-1).unwrap();

    for r in 0..rows {
        for c in 0..cols {
            let binary_in = unsafe { input.at_2d_unchecked::<u8>(r, c).unwrap() };
            let d = unsafe { out_mat.at_2d_unchecked_mut::<Vec4f>(r, c).unwrap() };

            if *binary_in > 0 {
                *d = Vec4f::new(r as f32 / rows as f32, c as f32 / cols as f32, 1.0, 0.0);
            } else {
                *d = Vec4f::new(0.0, 0.0, 0.0, 0.0);
            }
        }
    }

    dst.output_array().unwrap().assign_1(&out_mat).unwrap();
}

fn calculate_centroid(src: &dyn opencv::core::ToInputArray) -> [f32; 2] {
    let input = src.input_array().unwrap().get_mat(-1).unwrap();
    let rows = input.rows();
    let cols = input.cols();

    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut s = 0.0;

    for r in 0..rows {
        for c in 0..cols {
            let d = unsafe { input.at_2d_unchecked::<Vec4f>(r, c).unwrap() };
            sx += d[0];
            sy += d[1];
            s += d[2];
        }
    }

    if s > 0.0 {
        [sx / s, sy / s]
    } else {
        [0.0, 0.0]
    }
}
