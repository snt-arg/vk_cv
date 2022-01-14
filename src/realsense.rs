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

    depth_scale: f32,
}

impl Realsense {
    pub fn open(
        color_dimensions: [u32; 2],
        color_framerate: u32,
        depth_dimensions: [u32; 2],
        depth_framerate: u32,
    ) -> Self {
        unsafe {
            let err = ptr::null_mut();
            let ctx = rs2_create_context(RS2_API_VERSION as i32, err);

            if err != ptr::null_mut() {
                println!("got an error!");
            }

            //let msg = rs2_get_error_message(*err);

            let device_list = rs2_query_devices(ctx, err);
            let dev = rs2_create_device(device_list, 0, err);
            rs2_delete_device_list(device_list);

            // query depth scale
            let mut depth_scale = 1.0;
            let sensor_list = rs2_query_sensors(dev, ptr::null_mut());
            let sensor_count = rs2_get_sensors_count(sensor_list, ptr::null_mut());
            for i in 0..sensor_count {
                let sensor = rs2_create_sensor(sensor_list, i, ptr::null_mut());

                // check for depth sensor
                if rs2_is_sensor_extendable_to(
                    sensor,
                    rs2_extension_RS2_EXTENSION_DEPTH_SENSOR,
                    ptr::null_mut(),
                ) > 0
                {
                    depth_scale = rs2_get_depth_scale(sensor, ptr::null_mut());
                    rs2_delete_sensor(sensor);
                    break;
                }

                rs2_delete_sensor(sensor);
            }
            rs2_delete_sensor_list(sensor_list);

            // setup pipeline
            let pipe = rs2_create_pipeline(ctx, ptr::null_mut());
            let config = rs2_create_config(ptr::null_mut());
            rs2_config_enable_stream(
                config,
                rs2_stream_RS2_STREAM_COLOR,
                0,
                color_dimensions[0] as i32,
                color_dimensions[1] as i32,
                rs2_format_RS2_FORMAT_RGBA8,
                color_framerate as i32,
                ptr::null_mut(),
            );
            rs2_config_enable_stream(
                config,
                rs2_stream_RS2_STREAM_DEPTH,
                0,
                depth_dimensions[0] as i32,
                depth_dimensions[1] as i32,
                rs2_format_RS2_FORMAT_Z16,
                depth_framerate as i32,
                ptr::null_mut(),
            );
            let profile = rs2_pipeline_start_with_config(pipe, config, ptr::null_mut());

            //let one_meter = rs2_get_depth_scale(sensor, error)

            Realsense {
                ctx,
                config,
                pipe,
                profile,
                dev,
                depth_scale,
            }
        }
    }

    pub fn fetch_image(&mut self) -> (Frame, Frame) {
        let mut color_frame = Frame {
            frame: ptr::null_mut(),
        };
        let mut depth_frame = Frame {
            frame: ptr::null_mut(),
        };

        unsafe {
            let err = ptr::null_mut();
            let composite = rs2_pipeline_wait_for_frames(self.pipe, RS2_DEFAULT_TIMEOUT, err);

            let frames_count = rs2_embedded_frames_count(composite, err);

            for i in 0..frames_count {
                let frame_ptr = rs2_extract_frame(composite, i, err);

                // depth frame
                if rs2_is_frame_extendable_to(
                    frame_ptr,
                    rs2_extension_RS2_EXTENSION_DEPTH_FRAME,
                    err,
                ) > 0
                {
                    depth_frame.frame = frame_ptr;
                }

                // color resp. video frame
                if rs2_is_frame_extendable_to(
                    frame_ptr,
                    rs2_extension_RS2_EXTENSION_VIDEO_FRAME,
                    err,
                ) > 0
                {
                    color_frame.frame = frame_ptr;
                }
            }

            rs2_release_frame(composite);
        }

        (color_frame, depth_frame)
    }
}

impl Drop for Realsense {
    fn drop(&mut self) {
        unsafe {
            rs2_pipeline_stop(self.pipe, ptr::null_mut());
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
