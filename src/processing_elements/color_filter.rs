use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageAccess, StorageImage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

use crate::utils::{create_storage_image, ImageInfo};

use super::ProcessingElement;

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/color_filter.comp.glsl",
    }
}

pub struct ColorFilter {
    input_img: Option<Arc<StorageImage>>,
    output_img: Option<Arc<StorageImage>>,
}

impl ColorFilter {
    pub fn new() -> Self {
        Self {
            input_img: None,
            output_img: None,
        }
    }
}

impl ProcessingElement for ColorFilter {
    fn input_image(&self) -> Option<Arc<StorageImage>> {
        self.input_img.clone()
    }

    fn output_image(&self) -> Option<Arc<StorageImage>> {
        self.output_img.clone()
    }

    fn build(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input: &dyn ProcessingElement,
    ) {
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

        // input image
        let input_img = input.output_image().unwrap();

        // output image
        let output_img = create_storage_image(
            device.clone(),
            queue.clone(),
            &ImageInfo::from_image(&input_img, Format::R8_UNORM),
        );

        // setup layout
        let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        let input_img_view = ImageView::new(input_img.clone()).unwrap();
        let output_img_view = ImageView::new(output_img.clone()).unwrap();

        set_builder.add_image(input_img_view).unwrap();
        set_builder.add_image(output_img_view).unwrap();

        let set = set_builder.build().unwrap();

        // build command buffer
        let push_constants = cs::ty::PushConstants {
            rgb_min: [0.0, 0.0, 0.0],
            rgb_max: [0.5, 1.0, 0.5],
            _dummy0: [0, 0, 0, 0],
        };

        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .push_constants(pipeline.layout().clone(), 0, push_constants)
            .dispatch([
                input_img.dimensions().width() / 16,
                input_img.dimensions().height() / 16,
                1,
            ])
            .unwrap();

        self.input_img = Some(input_img);
        self.output_img = Some(output_img);
    }
}
