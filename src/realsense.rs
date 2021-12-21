use std::ffi::c_void;

use realsense_sys::*;

#[derive(Debug)]
pub struct Realsense {
    ctx: *mut rs2_context,
    config: *mut rs2_config,
    pipe: *mut rs2_pipeline,
    profile: *mut rs2_pipeline_profile,
    dev: *mut rs2_device,
}

pub struct Frame {
    frame: *mut rs2_frame,
}

impl Frame {
    pub fn bytes_count(&self) -> u32 {
        unsafe { rs2_get_frame_data_size(self.frame, std::ptr::null_mut()) as u32 }
    }

    pub fn width(&self) -> u32 {
        unsafe { rs2_get_frame_width(self.frame, std::ptr::null_mut()) as u32 }
    }

    pub fn height(&self) -> u32 {
        unsafe { rs2_get_frame_height(self.frame, std::ptr::null_mut()) as u32 }
    }

    pub fn data_slice(&self) -> &[u8] {
        unsafe {
            let ptr = rs2_get_frame_data(self.frame, std::ptr::null_mut()) as *const u8;
            std::slice::from_raw_parts(ptr, self.bytes_count() as usize)
        }
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        unsafe {
            rs2_release_frame(self.frame);
        }
    }
}

impl Realsense {
    pub fn new() -> Self {
        unsafe {
            let err = std::ptr::null_mut();
            let ctx = rs2_create_context(RS2_API_VERSION as i32, err);

            if err != std::ptr::null_mut() {
                println!("got an error!");
            }
            //let msg = rs2_get_error_message(*err);

            let device_list = rs2_query_devices(ctx, err);
            let dev = rs2_create_device(device_list, 0, err);

            let pipe = rs2_create_pipeline(ctx, std::ptr::null_mut());
            let config = rs2_create_config(std::ptr::null_mut());
            rs2_config_enable_stream(
                config,
                rs2_stream_RS2_STREAM_COLOR,
                0,
                640,
                480,
                rs2_format_RS2_FORMAT_RGBA8,
                30,
                std::ptr::null_mut(),
            );
            let profile = rs2_pipeline_start_with_config(pipe, config, std::ptr::null_mut());

            Realsense {
                ctx,
                config,
                pipe,
                profile,
                dev,
            }
        }
    }

    pub fn fetch_image(&mut self) -> Frame {
        let mut color_frame = Frame {
            frame: std::ptr::null_mut(),
        };

        unsafe {
            let err = std::ptr::null_mut();
            let composite = rs2_pipeline_wait_for_frames(self.pipe, RS2_DEFAULT_TIMEOUT, err);

            let frames_count = rs2_embedded_frames_count(composite, err);

            for i in 0..frames_count {
                let frame = rs2_extract_frame(composite, i, err);

                if err != std::ptr::null_mut() {
                    println!("got an error!");
                }

                let h = rs2_get_frame_height(frame, std::ptr::null_mut());
                let w = rs2_get_frame_width(frame, std::ptr::null_mut());

                color_frame.frame = frame;
            }

            rs2_release_frame(composite);
        }

        color_frame
    }
}
