use vulkano::image::ImageAccess;

use crate::{
    processing_elements::{Io, IoFragment},
    utils::{self, ImageInfo},
};

pub struct ImageDownload {
    io: IoFragment,
    info: ImageInfo,
    buffer: Vec<u8>,
}

impl ImageDownload {
    pub fn from_io(io: IoFragment) -> Result<Self, &'static str> {
        match &io.output {
            Io::Buffer(_) => {
                let input_img = io.input_image().unwrap();

                let info = ImageInfo {
                    width: input_img.dimensions().width(),
                    height: input_img.dimensions().height(),
                    format: input_img.format(),
                };

                Ok(Self {
                    io,
                    info,
                    buffer: Vec::new(),
                })
            }
            _ => Err("Output needs to be a buffer"),
        }
    }

    pub fn image_info(&self) -> &ImageInfo {
        &self.info
    }

    pub fn transfer<'a>(&'a mut self) -> TransferredImage<'a> {
        let buffer = self.io.output_buffer().unwrap();
        let buffer_content = buffer.read().unwrap();
        if self.buffer.len() != buffer_content.len() {
            self.buffer.resize(buffer_content.len(), 0);
        }

        self.buffer.copy_from_slice(&buffer_content);

        TransferredImage {
            buffer: &self.buffer,
            info: &self.info,
        }
    }

    pub fn transferred_image<'a>(&'a self) -> TransferredImage<'a> {
        TransferredImage {
            buffer: &self.buffer,
            info: &self.info,
        }
    }
}

pub struct TransferredImage<'a> {
    buffer: &'a [u8],
    info: &'a ImageInfo,
}

impl<'a> TransferredImage<'a> {
    pub fn buffer_content(&self) -> &[u8] {
        &self.buffer
    }

    pub fn save_output_buffer(&self, filename: &str) {
        utils::write_image(filename, &self.buffer_content(), &self.info);
    }

    pub fn info(&self) -> &'a ImageInfo {
        self.info
    }
}
