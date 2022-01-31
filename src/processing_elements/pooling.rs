use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    device::{Device, Queue},
    image::{view::ImageView, ImageAccess},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

use crate::utils;

use super::{Io, IoFragment, ProcessingElement};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/pooling_2x2.comp.glsl",
    }
}

pub enum Operation {
    Min,
    Max,
}

pub struct Pooling {
    op: Operation,
}

impl Pooling {
    pub fn new(op: Operation) -> Self {
        Self { op }
    }
}

impl ProcessingElement for Pooling {
    fn build(
        &self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input: &IoFragment,
    ) -> IoFragment {
        let local_size = 16;

        let pipeline = {
            let shader = cs::load(device.clone()).unwrap();
            ComputePipeline::new(
                device.clone(),
                shader.entry_point("main").unwrap(),
                &cs::SpecializationConstants {
                    constant_0: local_size,
                    constant_1: local_size,
                    min_max: match self.op {
                        Operation::Min => 0,
                        Operation::Max => 1,
                    },
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        };

        // input image
        let input_img = input.output_image().unwrap();

        // output image
        let output_img = utils::create_storage_image(
            device.clone(),
            queue.clone(),
            &utils::ImageInfo {
                width: input_img.dimensions().width() / 2,
                height: input_img.dimensions().height() / 2,
                format: input_img.format(),
            },
        );

        // setup layout
        let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        let input_img_view = ImageView::new(input_img.clone()).unwrap();
        let output_img_view = ImageView::new(output_img.clone()).unwrap();

        set_builder.add_image(input_img_view).unwrap();
        set_builder.add_image(output_img_view).unwrap();

        let set = set_builder.build().unwrap();

        builder
            .bind_pipeline_compute(pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline.layout().clone(),
                0,
                set.clone(),
            )
            // .push_constants(pipeline.layout().clone(), 0, push_constants)
            .dispatch(utils::workgroups(
                &input_img.dimensions().width_height(),
                &[local_size, local_size],
            ))
            .unwrap();

        IoFragment {
            input: Io::Image(input_img),
            output: Io::Image(output_img),
            label: "Morphology".to_string(),
        }
    }
}
