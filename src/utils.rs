use std::sync::Arc;
use std::{fs::File, io::BufWriter, path::Path};

use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::{ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage};

use crate::processing_elements::{IoElement, PipeInput, PipeOutput, ProcessingElement};

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
        Format::R32G32B32A32_SFLOAT => {
            dbg!(data);
            let x = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            dbg!(x);
            let y = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
            dbg!(y);
            let z = f32::from_le_bytes([data[8], data[9], data[10], data[11]]);
            dbg!(z);

            dbg!(x / z, y / z);

            return;
        }
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
        sampled: true,
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

pub fn workgroups(dimensions: &[u32; 2], local_size: &[u32; 2]) -> [u32; 3] {
    [
        (dimensions[0] as f32 / local_size[0] as f32).ceil() as u32,
        (dimensions[1] as f32 / local_size[1] as f32).ceil() as u32,
        1,
    ]
}

pub fn cv_pipeline<I, O>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    input: &mut I,
    elements: &mut [&mut dyn ProcessingElement],
    output: &mut O,
) -> (Arc<PrimaryAutoCommandBuffer>, IoElement, IoElement)
where
    I: PipeInput + ProcessingElement,
    O: PipeOutput + ProcessingElement,
{
    let mut builder = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        vulkano::command_buffer::CommandBufferUsage::MultipleSubmit,
    )
    .unwrap();

    let dummy = IoElement::dummy();
    let input_io = input.build(device.clone(), queue.clone(), &mut builder, &dummy);

    let mut io_elements = vec![input_io.clone()];

    for pe in elements {
        io_elements.push(pe.build(
            device.clone(),
            queue.clone(),
            &mut builder,
            io_elements.last().as_ref().unwrap(),
        ));
    }

    let output_io = output.build(
        device.clone(),
        queue.clone(),
        &mut builder,
        io_elements.last().as_ref().unwrap(),
    );

    let command_buffer = Arc::new(builder.build().unwrap());
    (command_buffer, input_io, output_io)
}
