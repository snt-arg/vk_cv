use std::sync::Arc;
use std::{fs::File, io::BufWriter, path::Path};

use vulkano::command_buffer::CommandBufferExecFuture;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::{ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage};
use vulkano::sync::NowFuture;
use vulkano::sync::{self, FenceSignalFuture, GpuFuture};

use anyhow::Result;

use crate::processing_elements::{PipeInput, PipeOutput, ProcessingElement};

pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub format: Format,
}

impl ImageInfo {
    pub fn bytes_count(&self) -> u32 {
        self.width * self.height * self.format.size().unwrap() as u32
    }

    pub fn from_image(image: &Arc<StorageImage>, format: Format) -> Self {
        Self {
            width: image.dimensions().width(),
            height: image.dimensions().height(),
            format,
        }
    }
}

impl From<&Arc<StorageImage>> for ImageInfo {
    fn from(image: &Arc<StorageImage>) -> Self {
        Self {
            width: image.dimensions().width(),
            height: image.dimensions().height(),
            format: image.format(),
        }
    }
}

pub fn load_image(image_path: &str) -> (ImageInfo, Vec<u8>) {
    let p = format!("{}/media/{}", env!("CARGO_MANIFEST_DIR"), image_path);
    let path = Path::new(&p);
    let decoder = png::Decoder::new(File::open(path).unwrap());
    let mut reader = decoder.read_info().unwrap();
    let mut img_data = vec![0; reader.output_buffer_size()];
    let oi = reader.next_frame(&mut img_data).unwrap();

    println!(
        "Loaded image '{}' ({}x{}) format {:?} {:?} bits",
        image_path, oi.width, oi.height, oi.color_type, oi.bit_depth
    );

    (
        ImageInfo {
            width: oi.width,
            height: oi.height,
            format: Format::R8G8B8A8_UNORM,
        },
        img_data,
    )
}

pub fn write_image(image_path: &str, data: &[u8], img_info: ImageInfo) {
    let p = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), image_path);
    let path = Path::new(&p);
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, img_info.width, img_info.height);

    match img_info.format {
        Format::R8G8B8A8_UNORM => encoder.set_color(png::ColorType::Rgba),
        Format::R8_UNORM => encoder.set_color(png::ColorType::Grayscale),
        _ => unimplemented!(),
    }

    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(data).unwrap();
}

pub fn create_storage_image(
    device: Arc<Device>,
    queue: Arc<Queue>,
    img_info: &ImageInfo,
) -> Arc<StorageImage> {
    let usage = ImageUsage {
        transfer_source: true,
        transfer_destination: true,
        storage: true,
        ..ImageUsage::none()
    };
    let flags = ImageCreateFlags::none();

    StorageImage::with_usage(
        device.clone(),
        ImageDimensions::Dim2d {
            width: img_info.width,
            height: img_info.height,
            array_layers: 1,
        },
        img_info.format,
        usage,
        flags,
        Some(queue.family()),
    )
    .unwrap()
}

#[macro_export]
macro_rules! cv_pipeline {
    ($device:expr,$queue:expr,input: $input:expr,elements: [$($pe:expr),*],output: $output:expr) => {
        {
            // type check
            let _:&dyn processing_elements::PipeInput = &$input;
            let _:&dyn processing_elements::PipeOutput = &$output;

            // build future
            sync::now($device.clone()).then_execute($queue.clone(), $input.command_buffer()).unwrap()
            $(
                .then_execute_same_queue($pe.command_buffer()).unwrap()
            )*
            .then_execute_same_queue($output.command_buffer()).unwrap()
            .then_signal_fence_and_flush().unwrap()
        }
    };
}
