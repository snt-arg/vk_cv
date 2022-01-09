use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    device::{Device, Queue},
    image::{view::ImageView, ImageAccess, StorageImage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

use crate::utils;

use super::ProcessingElement;

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/morphology3x3.comp.glsl",
    }
}

pub enum Operation {
    Erode,
    Dilate,
}

pub struct Morphology {
    input_img: Option<Arc<StorageImage>>,
    output_img: Option<Arc<StorageImage>>,
    op: Operation,
}

impl Morphology {
    pub fn new(op: Operation) -> Self {
        Self {
            input_img: None,
            output_img: None,
            op,
        }
    }
}

impl ProcessingElement for Morphology {
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
        let local_size = 16;

        let pipeline = {
            let shader = cs::load(device.clone()).unwrap();
            ComputePipeline::new(
                device.clone(),
                shader.entry_point("main").unwrap(),
                &cs::SpecializationConstants {
                    constant_0: local_size,
                    constant_1: local_size,
                    erode_dilate: match self.op {
                        Operation::Erode => 0,
                        Operation::Dilate => 1,
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
        let output_img =
            utils::create_storage_image(device.clone(), queue.clone(), &(&input_img).into());

        // setup layout
        let layout = pipeline.layout().descriptor_set_layouts().get(0).unwrap();
        let mut set_builder = PersistentDescriptorSet::start(layout.clone());

        let input_img_view = ImageView::new(input_img.clone()).unwrap();
        let output_img_view = ImageView::new(output_img.clone()).unwrap();

        set_builder.add_image(input_img_view).unwrap();
        set_builder.add_image(output_img_view).unwrap();

        let set = set_builder.build().unwrap();

        // build command buffer
        // let push_constants = cs::ty::PushConstants {
        //     kernel: [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0],
        //     offset: 1.0,
        //     denom: 0.5,
        // };

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

        self.input_img = Some(input_img);
        self.output_img = Some(output_img);
    }
}