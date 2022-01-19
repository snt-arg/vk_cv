use vulkano::image::ImageAccess;

use crate::{
    processing_elements::{Io, IoFragment},
    utils::{self, ImageInfo},
};

pub struct ImageDownload {
    io: IoFragment,
}

impl ImageDownload {
    pub fn from_io(io: IoFragment) -> Result<Self, &'static str> {
        match &io.output {
            Io::Buffer(_) => Ok(Self { io }),
            _ => Err("Output needs to be a buffer"),
        }
    }

    pub fn save_output_buffer(&self, filename: &str) {
        let buffer = self.io.output_buffer().unwrap();
        let buffer_content = buffer.read().unwrap();
        let input_img = self.io.input_image().unwrap();

        let info = ImageInfo {
            width: input_img.dimensions().width(),
            height: input_img.dimensions().height(),
            format: input_img.format(),
        };

        utils::write_image(filename, &buffer_content, info);
    }

    pub fn centroid(&self) -> ([f32; 2], f32) {
        let buffer = self.io.output_buffer().unwrap();
        let data = buffer.read().unwrap();

        let x = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        // dbg!(x);
        let y = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        // dbg!(y);
        let z = f32::from_le_bytes([data[8], data[9], data[10], data[11]]);
        // dbg!(z);

        // dbg!(x / z, y / z);

        ([x / z, y / z], z)
    }
}
