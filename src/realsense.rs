use std::{ffi::c_void, ptr};

use realsense_sys::*;
use vulkano::format::Format::R8G8B8A8_UNORM;

use crate::utils::ImageInfo;

#[derive(Debug)]
pub struct Realsense {
    ctx: *mut rs2_context,
    config: *mut rs2_config,
    pipe: *mut rs2_pipeline,
    profile: *mut rs2_pipeline_profile,
    dev: *mut rs2_device,
}

impl Realsense {
    pub fn open() -> Self {
        unsafe {
            let err = ptr::null_mut();
            let ctx = rs2_create_context(RS2_API_VERSION as i32, err);

            if err != ptr::null_mut() {
                println!("got an error!");
            }
            //let msg = rs2_get_error_message(*err);

            let device_list = rs2_query_devices(ctx, err);
            let dev = rs2_create_device(device_list, 0, err);

            let pipe = rs2_create_pipeline(ctx, ptr::null_mut());
            let config = rs2_create_config(ptr::null_mut());
            rs2_config_enable_stream(
                config,
                rs2_stream_RS2_STREAM_COLOR,
                0,
                640,
                480,
                rs2_format_RS2_FORMAT_RGBA8,
                30,
                ptr::null_mut(),
            );
            let profile = rs2_pipeline_start_with_config(pipe, config, ptr::null_mut());

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
            frame: ptr::null_mut(),
        };

        unsafe {
            let err = ptr::null_mut();
            let composite = rs2_pipeline_wait_for_frames(self.pipe, RS2_DEFAULT_TIMEOUT, err);

            let frames_count = rs2_embedded_frames_count(composite, err);

            for i in 0..frames_count {
                let frame = rs2_extract_frame(composite, i, err);

                if err != ptr::null_mut() {
                    println!("got an error!");
                }

                let h = rs2_get_frame_height(frame, ptr::null_mut());
                let w = rs2_get_frame_width(frame, ptr::null_mut());

                color_frame.frame = frame;
            }

            rs2_release_frame(composite);
        }

        color_frame
    }
}

impl Drop for Realsense {
    fn drop(&mut self) {
        unsafe {
            rs2_delete_pipeline(self.pipe);
            rs2_delete_pipeline_profile(self.profile);
            rs2_delete_config(self.config);
            rs2_delete_device(self.dev);
            rs2_delete_context(self.ctx);
        }
    }
}

pub struct Frame {
    frame: *mut rs2_frame,
}

impl Frame {
    pub fn bytes_count(&self) -> u32 {
        unsafe { rs2_get_frame_data_size(self.frame, ptr::null_mut()) as u32 }
    }

    pub fn width(&self) -> u32 {
        unsafe { rs2_get_frame_width(self.frame, ptr::null_mut()) as u32 }
    }

    pub fn height(&self) -> u32 {
        unsafe { rs2_get_frame_height(self.frame, ptr::null_mut()) as u32 }
    }

    pub fn data_slice(&self) -> &[u8] {
        unsafe {
            let ptr = rs2_get_frame_data(self.frame, ptr::null_mut()) as *const u8;
            std::slice::from_raw_parts(ptr, self.bytes_count() as usize)
        }
    }

    pub fn image_info(&self) -> ImageInfo {
        ImageInfo {
            width: self.width(),
            height: self.height(),
            format: R8G8B8A8_UNORM,
        }
    }

    pub fn stride(&self) -> u32 {
        unsafe { rs2_get_frame_stride_in_bytes(self.frame, ptr::null_mut()) as u32 }
    }

    pub fn crop(&self, new_width: u32, new_height: u32) -> (ImageInfo, Vec<u8>) {
        let mut data = vec![0; (new_width * new_height * 4) as usize];
        let old_data = self.data_slice();
        let stride = new_width * 4;
        let old_stride = self.stride();

        for y in 0..new_height.min(self.height()) {
            let offset = (y * stride) as usize;
            let (_, right) = data.split_at_mut(offset);

            let old_offset_from = (y * old_stride) as usize;
            let count = self.width().min(new_width);
            let old_offset_to = old_offset_from + count as usize;
            right.copy_from_slice(&old_data[old_offset_from..old_offset_to]);
        }

        (
            ImageInfo {
                width: new_width,
                height: new_height,
                format: self.image_info().format,
            },
            data.to_vec(),
        )
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        unsafe {
            rs2_release_frame(self.frame);
        }
    }
}
