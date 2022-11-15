use vulkano::{
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    image::{view::ImageView, ImageAccess},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
};

use crate::{utils, vk_init::VkContext};

use super::{AutoCommandBufferBuilder, Io, IoFragment, ProcessingElement};

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
        &self,
        ctx: &VkContext,
        builder: &mut AutoCommandBufferBuilder,
        input: &IoFragment,
    ) -> IoFragment {
        let local_size = 32;

        // shader for the first pass
        let pipeline_1p = {
            let shader = cs::load(ctx.device.clone()).unwrap();
            ComputePipeline::new(
                ctx.device.clone(),
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
            let shader = cs::load(ctx.device.clone()).unwrap();
            ComputePipeline::new(
                ctx.device.clone(),
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
        let intermediate_img = utils::create_storage_image(ctx, &(&input_img).into());
        // output image for second pass
        let output_img = utils::create_storage_image(ctx, &(&input_img).into());

        // setup layout
        let input_img_view = ImageView::new_default(input_img.clone()).unwrap();
        let intermediate_img_view = ImageView::new_default(intermediate_img.clone()).unwrap();
        let output_img_view = ImageView::new_default(output_img.clone()).unwrap();

        let layout_1p = pipeline_1p.layout().set_layouts().get(0).unwrap();

        let set_1p = PersistentDescriptorSet::new(
            &ctx.memory.descriptor_set_allocator,
            layout_1p.clone(),
            [
                WriteDescriptorSet::image_view(0, input_img_view),
                WriteDescriptorSet::image_view(1, intermediate_img_view.clone()),
            ],
        )
        .unwrap();

        let layout_2p = pipeline_2p.layout().set_layouts().get(0).unwrap();

        let set_2p = PersistentDescriptorSet::new(
            &ctx.memory.descriptor_set_allocator,
            layout_2p.clone(),
            [
                WriteDescriptorSet::image_view(0, intermediate_img_view),
                WriteDescriptorSet::image_view(1, output_img_view),
            ],
        )
        .unwrap();

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

        IoFragment {
            input: Io::Image(input_img),
            output: Io::Image(output_img.clone()),
            label: utils::basic_label("Convolution two passes", &output_img),
        }
    }
}
