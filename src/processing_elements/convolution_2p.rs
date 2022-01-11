use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    device::{Device, Queue},
    image::{view::ImageView, ImageAccess},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

use crate::utils;

use super::{Io, IoElement, ProcessingElement};

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/convolution_2p_3x3.comp.glsl",
    }
}

pub struct Convolution2Pass {}

impl Convolution2Pass {
    pub fn new() -> Self {
        Self {}
    }
}

impl ProcessingElement for Convolution2Pass {
    fn build(
        &mut self,
        device: Arc<Device>,
        queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        input: &IoElement,
    ) -> IoElement {
        let local_size = 32;

        // shader for the first pass
        let pipeline_1p = {
            let shader = cs::load(device.clone()).unwrap();
            ComputePipeline::new(
                device.clone(),
                shader.entry_point("main").unwrap(),
                &cs::SpecializationConstants {
                    constant_0: local_size,
                    constant_1: local_size,
                    offset: 0.5,
                    m1: 1.0,
                    m2: 2.0,
                    m3: 1.0,
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        };

        // input image
        let input_img = input.output_image().unwrap();

        // shader for the second pass
        let pipeline_2p = {
            let shader = cs::load(device.clone()).unwrap();
            ComputePipeline::new(
                device.clone(),
                shader.entry_point("main").unwrap(),
                &cs::SpecializationConstants {
                    constant_0: local_size,
                    constant_1: local_size,
                    v_pass: 1,
                    offset: 0.5,
                    m4: 1.0,
                    m5: 0.0,
                    m6: -1.0,
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        };

        // output image for first pass
        let intermediate_img =
            utils::create_storage_image(device.clone(), queue.clone(), &(&input_img).into());
        // output image for second pass
        let output_img =
            utils::create_storage_image(device.clone(), queue.clone(), &(&input_img).into());

        // setup layout
        let input_img_view = ImageView::new(input_img.clone()).unwrap();
        let intermediate_img_view = ImageView::new(intermediate_img.clone()).unwrap();
        let output_img_view = ImageView::new(output_img.clone()).unwrap();

        let layout_1p = pipeline_1p
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();
        let mut set_builder = PersistentDescriptorSet::start(layout_1p.clone());
        set_builder.add_image(input_img_view).unwrap();
        set_builder
            .add_image(intermediate_img_view.clone())
            .unwrap();

        let set_1p = set_builder.build().unwrap();

        let layout_2p = pipeline_2p
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();
        let mut set_builder = PersistentDescriptorSet::start(layout_2p.clone());
        set_builder.add_image(intermediate_img_view).unwrap();
        set_builder.add_image(output_img_view).unwrap();

        let set_2p = set_builder.build().unwrap();

        // build command buffer
        builder
            .bind_pipeline_compute(pipeline_1p.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline_1p.layout().clone(),
                0,
                set_1p.clone(),
            )
            // .push_constants(pipeline.layout().clone(), 0, push_constants)
            .dispatch(utils::workgroups(
                &input_img.dimensions().width_height(),
                &[local_size, local_size],
            ))
            .unwrap()
            .bind_pipeline_compute(pipeline_2p.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline_2p.layout().clone(),
                0,
                set_2p.clone(),
            )
            // .push_constants(pipeline.layout().clone(), 0, push_constants)
            .dispatch(utils::workgroups(
                &input_img.dimensions().width_height(),
                &[local_size, local_size],
            ))
            .unwrap();

        IoElement {
            input: Io::Image(input_img),
            output: Io::Image(output_img),
        }
    }
}
