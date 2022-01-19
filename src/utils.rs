use std::sync::Arc;
use std::{fs::File, io::BufWriter, path::Path};

use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use vulkano::device::{Device, Queue};
use vulkano::format::Format;
use vulkano::image::{ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage};
use vulkano::sync::{self, GpuFuture};

use crate::endpoints::image_download::ImageDownload;
use crate::processing_elements::output::Output;
use crate::processing_elements::{IoFragment, PipeInput, PipeOutput, ProcessingElement};

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

    let (ct, _depth) = reader.output_color_type();

    // currently we only support RGBA images since RGB images cannot be
    // optimally represented by the raspberry
    match ct {
        png::ColorType::Rgba => (),
        _ => panic!("RGBA format required!"),
    }

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
        format => {
            println!("Cannot save format {:?}", format.type_color());
            return;
        }
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

pub fn cv_pipeline_sequential<I, O>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    input: &I,
    elements: &[&dyn ProcessingElement],
    output: &O,
) -> (Arc<PrimaryAutoCommandBuffer>, IoFragment, IoFragment)
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

    let dummy = IoFragment::none();
    let input_io = input.build(device.clone(), queue.clone(), &mut builder, &dummy);

    let mut io_fragments = vec![input_io.clone()];

    for pe in elements {
        io_fragments.push(pe.build(
            device.clone(),
            queue.clone(),
            &mut builder,
            io_fragments.last().as_ref().unwrap(),
        ));
    }

    let output_io = output.build(
        device.clone(),
        queue.clone(),
        &mut builder,
        io_fragments.last().as_ref().unwrap(),
    );

    let command_buffer = Arc::new(builder.build().unwrap());
    (command_buffer, input_io, output_io)
}

pub fn cv_pipeline_sequential_debug<I, O>(
    device: Arc<Device>,
    queue: Arc<Queue>,
    input: &I,
    elements: &[&dyn ProcessingElement],
    output: &O,
) -> DebugPipeline
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

    let dummy = IoFragment::none();
    let input_io = input.build(device.clone(), queue.clone(), &mut builder, &dummy);

    let mut io_fragments = vec![input_io.clone()];

    for pe in elements {
        io_fragments.push(pe.build(
            device.clone(),
            queue.clone(),
            &mut builder,
            io_fragments.last().as_ref().unwrap(),
        ));
    }

    let output_io = output.build(
        device.clone(),
        queue.clone(),
        &mut builder,
        io_fragments.last().as_ref().unwrap(),
    );

    // create generic outputs for each io element in the pipeline
    let generic_output = Output::new();
    let generic_output_ios: Vec<_> = io_fragments
        .iter()
        .map(|io| generic_output.build(device.clone(), queue.clone(), &mut builder, io))
        .collect();

    // finalize the command buffer
    let command_buffer = Arc::new(builder.build().unwrap());

    // create individual command buffers
    let mut indiv_io_elements = vec![input_io.clone()];
    let mut stage_labels = vec![];
    let mut individual_cbs = vec![];
    for pe in elements {
        let mut builder = vulkano::command_buffer::AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            vulkano::command_buffer::CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        let io = pe.build(
            device.clone(),
            queue.clone(),
            &mut builder,
            indiv_io_elements.last().as_ref().unwrap(),
        );
        stage_labels.push(io.label.to_string());

        indiv_io_elements.push(io);
        individual_cbs.push(Arc::new(builder.build().unwrap()));
    }

    DebugPipeline {
        cb: command_buffer,
        input: input_io,
        output: output_io,
        debug_outputs: generic_output_ios,
        individual_cbs,
        stage_labels,
    }
}

pub struct DebugPipeline {
    pub cb: Arc<PrimaryAutoCommandBuffer>,
    pub input: IoFragment,
    pub output: IoFragment,
    pub debug_outputs: Vec<IoFragment>,
    pub individual_cbs: Vec<Arc<PrimaryAutoCommandBuffer>>,

    stage_labels: Vec<String>,
}

impl DebugPipeline {
    pub fn dispatch(&self, device: Arc<Device>, queue: Arc<Queue>) {
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), self.cb.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap();

        future.wait(None).unwrap();
    }

    pub fn time(&self, device: Arc<Device>, queue: Arc<Queue>) {
        for (i, cb) in self.individual_cbs.iter().enumerate() {
            let future = sync::now(device.clone())
                .then_execute(queue.clone(), cb.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap();

            let started = std::time::Instant::now();
            future.wait(None).unwrap();
            let dt = std::time::Instant::now() - started;

            println!(
                "🠶 ({}) '{}' took {} μs",
                i,
                self.stage_labels[i],
                dt.as_micros()
            );
        }
    }

    pub fn save_all(&self, device: Arc<Device>, queue: Arc<Queue>, dir: &str, prefix: &str) {
        self.dispatch(device, queue);
        std::fs::create_dir_all(dir).unwrap();
        for (i, io) in self.debug_outputs.iter().enumerate() {
            let download = ImageDownload::from_io(io.clone()).unwrap();
            download.save_output_buffer(&format!("{}/{}{}.png", dir, prefix, i))
        }
    }
}
