use std::{
    ffi::{c_void, CStr},
    ops::Deref,
    ptr,
};

use realsense_sys::*;
use vulkano::format::Format::R8G8B8A8_UNORM;

use crate::utils::ImageInfo;

fn check_err(err: *const rs2_error) -> Result<(), String> {
    unsafe {
        if err != ptr::null() {
            let msg = rs2_get_error_message(err);
            let msg_str = std::ffi::CStr::from_ptr(msg);
            return Err(msg_str.to_str().unwrap().to_owned());
        }
    }
    Ok(())
}

fn panic_err(err: *const rs2_error) {
    check_err(err).unwrap();
}

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
        color_dimensions: &[u32; 2],
        color_framerate: u32,
        depth_dimensions: &[u32; 2],
        depth_framerate: u32,
    ) -> Result<Self, String> {
        unsafe {
            let mut err = ptr::null_mut();
            let ctx = rs2_create_context(RS2_API_VERSION as i32, &mut err);
            check_err(err)?;

            let device_list = rs2_query_devices(ctx, &mut err);
            check_err(err)?;

            let device_count = rs2_get_device_count(device_list, &mut err);
            check_err(err)?;
            if device_count == 0 {
                return Err("No realsense camera found".to_string());
            }

            let dev = rs2_create_device(device_list, 0, &mut err);
            check_err(err)?;
            rs2_delete_device_list(device_list);

            // query depth scale
            let mut depth_scale = 1.0;
            let sensor_list = rs2_query_sensors(dev, &mut err);
            check_err(err)?;
            let sensor_count = rs2_get_sensors_count(sensor_list, &mut err);
            check_err(err)?;
            for i in 0..sensor_count {
                let sensor = rs2_create_sensor(sensor_list, i, &mut err);
                panic_err(err);

                // check for depth sensor
                if rs2_is_sensor_extendable_to(
                    sensor,
                    rs2_extension_RS2_EXTENSION_DEPTH_SENSOR,
                    ptr::null_mut(),
                ) > 0
                {
                    depth_scale = rs2_get_depth_scale(sensor, &mut err);
                    panic_err(err);
                    rs2_delete_sensor(sensor);
                    break;
                }

                rs2_delete_sensor(sensor);
            }
            rs2_delete_sensor_list(sensor_list);

            // setup pipeline
            let pipe = rs2_create_pipeline(ctx, &mut err);
            check_err(err)?;
            let config = rs2_create_config(&mut err);
            check_err(err)?;
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
            let profile = rs2_pipeline_start_with_config(pipe, config, &mut err);
            check_err(err)?;

            Ok(Realsense {
                ctx,
                config,
                pipe,
                profile,
                dev,
                depth_scale,
            })
        }
    }

    pub fn fetch_image(&mut self) -> (ColorFrame, DepthFrame) {
        let mut color_frame = Frame {
            frame: ptr::null_mut(),
        };
        let mut depth_frame = Frame {
            frame: ptr::null_mut(),
        };

        unsafe {
            let mut err = ptr::null_mut();
            let composite = rs2_pipeline_wait_for_frames(self.pipe, RS2_DEFAULT_TIMEOUT, &mut err);
            panic_err(err);

            let frames_count = rs2_embedded_frames_count(composite, &mut err);
            panic_err(err);

            for i in 0..frames_count {
                let frame_ptr = rs2_extract_frame(composite, i, &mut err);
                panic_err(err);

                // depth frame
                if rs2_is_frame_extendable_to(
                    frame_ptr,
                    rs2_extension_RS2_EXTENSION_DEPTH_FRAME,
                    &mut err,
                ) > 0
                {
                    depth_frame.frame = frame_ptr;
                }

                // color resp. video frame
                if rs2_is_frame_extendable_to(
                    frame_ptr,
                    rs2_extension_RS2_EXTENSION_VIDEO_FRAME,
                    &mut err,
                ) > 0
                {
                    color_frame.frame = frame_ptr;
                }
            }

            rs2_release_frame(composite);
        }

        (ColorFrame(color_frame), DepthFrame(depth_frame))
    }

    pub fn depth_at_pixel(
        &self,
        color_px: &[f32; 2],
        color_frame: &ColorFrame,
        depth_frame: &DepthFrame,
    ) -> Option<f32> {
        unsafe {
            if !(f32::is_finite(color_px[0]) && f32::is_finite(color_px[1])) {
                return None;
            }

            let mut err = ptr::null_mut();

            let color_stream_profile = rs2_get_frame_stream_profile(color_frame.frame, &mut err);
            panic_err(err);

            let depth_stream_profile = rs2_get_frame_stream_profile(depth_frame.frame, &mut err);
            panic_err(err);

            let mut video_intrinsics = std::mem::zeroed::<rs2_intrinsics>();
            rs2_get_video_stream_intrinsics(color_stream_profile, &mut video_intrinsics, &mut err);
            panic_err(err);

            let mut depth_intrinsics = std::mem::zeroed::<rs2_intrinsics>();
            rs2_get_video_stream_intrinsics(depth_stream_profile, &mut depth_intrinsics, &mut err);
            panic_err(err);

            let mut depth2video_extrinsics = std::mem::zeroed::<rs2_extrinsics>();
            rs2_get_extrinsics(
                depth_stream_profile,
                color_stream_profile,
                &mut depth2video_extrinsics,
                &mut err,
            );
            panic_err(err);

            let mut video2depth_extrinsics = std::mem::zeroed::<rs2_extrinsics>();
            rs2_get_extrinsics(
                color_stream_profile,
                depth_stream_profile,
                &mut video2depth_extrinsics,
                &mut err,
            );
            panic_err(err);

            let mut depth_pixel = [0.0, 0.0];

            rs2_project_color_pixel_to_depth_pixel(
                depth_pixel.as_mut_ptr(),
                depth_frame.data_ptr() as *const u16,
                self.depth_scale,
                0.15,
                12.6,
                &depth_intrinsics,
                &video_intrinsics,
                &video2depth_extrinsics,
                &depth2video_extrinsics,
                color_px.as_ptr(),
            );

            depth_frame
                .pixel_as_u16([depth_pixel[0] as u32, depth_pixel[1] as u32])
                .map(|depth| depth as f32 * self.depth_scale)
        }
    }

    pub fn deproject_pixel(
        &self,
        color_px: &[f32; 2],
        depth: f32,
        color_frame: &ColorFrame,
    ) -> [f32; 3] {
        unsafe {
            let mut err = ptr::null_mut();

            let color_stream_profile = rs2_get_frame_stream_profile(color_frame.frame, &mut err);
            panic_err(err);

            let mut video_intrinsics = std::mem::zeroed::<rs2_intrinsics>();
            rs2_get_video_stream_intrinsics(color_stream_profile, &mut video_intrinsics, &mut err);
            panic_err(err);

            let mut point = [0.0, 0.0, 0.0];
            rs2_deproject_pixel_to_point(
                point.as_mut_ptr(),
                &video_intrinsics,
                color_px.as_ptr(),
                depth,
            );

            point
        }
    }

    pub fn dump_intrinsic(&self, res: Option<(i32, i32)>) {
        println!("Dump intrinsics");
        unsafe {
            let mut err = ptr::null_mut();

            let sensor_list = rs2_query_sensors(self.dev, &mut err);
            panic_err(err);

            let sensor_count = rs2_get_sensors_count(sensor_list, &mut err);
            panic_err(err);

            for i in 0..sensor_count {
                let sensor = rs2_create_sensor(sensor_list, i, &mut err);
                panic_err(err);

                let profile_list = rs2_get_stream_profiles(sensor, &mut err);
                panic_err(err);

                let profile_count = rs2_get_stream_profiles_count(profile_list, &mut err);
                panic_err(err);

                let name =
                    rs2_get_sensor_info(sensor, rs2_camera_info_RS2_CAMERA_INFO_NAME, &mut err);
                panic_err(err);

                println!("Sensor name: {:?}", CStr::from_ptr(name));

                for p in 0..profile_count {
                    let stream_profile = rs2_get_stream_profile(profile_list, p, &mut err);
                    panic_err(err);

                    let mut video_intrinsics = std::mem::zeroed::<rs2_intrinsics>();
                    rs2_get_video_stream_intrinsics(
                        stream_profile,
                        &mut video_intrinsics,
                        &mut err,
                    );
                    panic_err(err);

                    if let Some((width, height)) = res {
                        if video_intrinsics.width == width && video_intrinsics.height == height {
                            dbg!(video_intrinsics);
                        }
                    } else {
                        dbg!(video_intrinsics);
                    }
                }

                rs2_delete_stream_profiles_list(profile_list);
                rs2_delete_sensor(sensor);
            }

            rs2_delete_sensor_list(sensor_list);
        }
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

pub struct DepthFrame(Frame);

impl Deref for DepthFrame {
    type Target = Frame;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ColorFrame(Frame);

impl Deref for ColorFrame {
    type Target = Frame;

    fn deref(&self) -> &Self::Target {
        &self.0
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

    fn data_ptr(&self) -> *const c_void {
        unsafe { rs2_get_frame_data(self.frame, ptr::null_mut()) }
    }

    pub fn pixel_as_u16(&self, p: [u32; 2]) -> Option<u16> {
        // this assumes that every pixel is 2bytes (16bits)
        // depth image (typically)

        let stride = self.stride();
        let x = p[0];
        let y = p[1];

        if x < self.width() && y < self.height() {
            let b1 = self
                .data_slice()
                .get((x * 2 + y * stride) as usize)
                .unwrap();
            let b2 = self
                .data_slice()
                .get(((x * 2 + y * stride) + 1) as usize)
                .unwrap();
            let v = u16::from_ne_bytes([*b1, *b2]);
            return Some(v);
        }

        None
    }

    pub fn area(&self) -> u32 {
        self.width() * self.height()
    }

    pub fn image_info(&self) -> ImageInfo {
        ImageInfo {
            width: self.width(),
            height: self.height(),
            format: R8G8B8A8_UNORM,
        }
    }

    pub fn bytes_per_pixel(&self) -> u32 {
        self.stride() / self.width()
    }

    pub fn stride(&self) -> u32 {
        unsafe { rs2_get_frame_stride_in_bytes(self.frame, ptr::null_mut()) as u32 }
    }

    pub fn save(&self, image_path: &str) {
        crate::utils::write_image(
            image_path,
            self.data_slice(),
            &ImageInfo {
                width: self.width(),
                height: self.height(),
                format: R8G8B8A8_UNORM,
            },
        );
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
