use std::sync::Arc;
use vulkano::{
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    descriptor_set::PersistentDescriptorSet,
    device::{Device, Queue},
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageCreateFlags, ImageDimensions, ImageUsage, StorageImage,
    },
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

use super::ProcessingElement;

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/convolution_2p_3x3.comp.glsl",
    }
}

pub struct Convolution2Pass {
    input_img: Arc<StorageImage>,
    output_img: Arc<StorageImage>,
    command_buffer: Arc<PrimaryAutoCommandBuffer>,
}

impl Convolution2Pass {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>, input_img: Arc<StorageImage>) -> Self {
        let local_size = 16;

        // shader for the first pass
        let pipeline_1p = {
            let shader = cs::load(device.clone()).unwrap();
            ComputePipeline::new(
                device.clone(),
                shader.entry_point("main").unwrap(),
                &cs::SpecializationConstants {
                    constant_0: local_size,
                    constant_1: local_size,
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        };

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
                    ..Default::default()
                },
                None,
                |_| {},
            )
            .unwrap()
        };

        let usage = ImageUsage {
            transfer_source: true,
            transfer_destination: true,
            storage: true,
            ..ImageUsage::none()
        };

        let intermediate_img = StorageImage::with_usage(
            device.clone(),
            ImageDimensions::Dim2d {
                width: input_img.dimensions().width(),
                height: input_img.dimensions().height(),
                array_layers: 1,
            },
            Format::R8_UNORM,
            usage,
            ImageCreateFlags::none(),
            Some(queue.family()),
        )
        .unwrap();

        let output_img = StorageImage::with_usage(
            device.clone(),
            ImageDimensions::Dim2d {
                width: input_img.dimensions().width(),
                height: input_img.dimensions().height(),
                array_layers: 1,
            },
            Format::R8_UNORM,
            usage,
            ImageCreateFlags::none(),
            Some(queue.family()),
        )
        .unwrap();

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
        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        // let push_constants = cs::ty::PushConstants {
        //     kernel: [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 2.0],
        //     offset: 1.0,
        //     denom: 0.5,
        // };

        builder
            .bind_pipeline_compute(pipeline_1p.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline_1p.layout().clone(),
                0,
                set_1p.clone(),
            )
            // .push_constants(pipeline.layout().clone(), 0, push_constants)
            .dispatch([
                input_img.dimensions().width() / local_size,
                input_img.dimensions().height() / local_size,
                1,
            ])
            .unwrap()
            .bind_pipeline_compute(pipeline_2p.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                pipeline_2p.layout().clone(),
                0,
                set_2p.clone(),
            )
            // .push_constants(pipeline.layout().clone(), 0, push_constants)
            .dispatch([
                input_img.dimensions().width() / local_size,
                input_img.dimensions().height() / local_size,
                1,
            ])
            .unwrap();

        let command_buffer = Arc::new(builder.build().unwrap());

        Self {
            input_img,
            output_img,
            command_buffer,
        }
    }

    pub fn output_image(&self) -> Arc<StorageImage> {
        self.output_img.clone()
    }
}

impl ProcessingElement for Convolution2Pass {
    fn command_buffer(&self) -> Arc<PrimaryAutoCommandBuffer> {
        self.command_buffer.clone()
    }
}
