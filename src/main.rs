use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage},
    descriptor_set::PersistentDescriptorSet,
    format::Format,
    image::{view::ImageView, ImageDimensions, StorageImage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, GpuFuture},
};

mod processing_elements;
mod utils;
mod vk_init;

fn main() {
    // v3d specs/properties: https://vulkan.gpuinfo.org/displayreport.php?id=13073#properties

    let (img_info, img_data) = utils::load_image("Large_Scaled_Forest_Lizard.png");

    // init device
    let (device, mut queues) = vk_init::init();

    let queue = queues.next().unwrap();

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/shaders/convelution.comp.glsl",
        }
    }

    mod cs_hsv {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/shaders/rgb_to_hsv.comp.glsl",
        }
    }

    let pipeline = {
        let shader = cs::load(device.clone()).unwrap();
        ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &cs::SpecializationConstants {},
            None,
            |_| {},
        )
        .unwrap()
    };

    // http://vulkan.gpuinfo.org/listlineartilingformats.php?platform=linux
    let src_image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: img_info.width,
            height: img_info.height,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.family()),
    )
    .unwrap();
    let src_img_view = ImageView::new(src_image.clone()).unwrap();

    let dst_image = StorageImage::new(
        device.clone(),
        ImageDimensions::Dim2d {
            width: img_info.width,
            height: img_info.height,
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(queue.family()),
    )
    .unwrap();
    let dst_img_view = ImageView::new(dst_image.clone()).unwrap();

    // setup layout
    let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
    let mut set_builder = PersistentDescriptorSet::start(layout.clone());

    set_builder.add_image(src_img_view.clone()).unwrap();
    set_builder.add_image(dst_img_view.clone()).unwrap();

    let set = set_builder.build().unwrap();

    let buf = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, img_data)
        .unwrap();

    // build command buffer
    let mut builder = AutoCommandBufferBuilder::primary(
        device.clone(),
        queue.family(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let push_constants = cs::ty::PushConstants {
        kernel: [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0],
        offset: 1.0,
        denom: 0.5,
    };

    builder
        .copy_buffer_to_image(buf.clone(), src_image.clone())
        .unwrap()
        .bind_pipeline_compute(pipeline.clone())
        .bind_descriptor_sets(
            PipelineBindPoint::Compute,
            pipeline.layout().clone(),
            0,
            set.clone(),
        )
        .push_constants(pipeline.layout().clone(), 0, push_constants)
        .dispatch([img_info.width / 16, img_info.height / 16, 1])
        .unwrap()
        .copy_image_to_buffer(dst_image.clone(), buf.clone())
        .unwrap();

    let command_buffer = builder.build().unwrap();

    // exec command buffer
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();

    // check data
    future.wait(None).unwrap();

    let buffer_content = buf.read().unwrap();
    utils::write_image("output.png", &buffer_content);
}
