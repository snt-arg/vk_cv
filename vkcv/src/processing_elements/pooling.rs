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
        ctx: &VkContext,
        builder: &mut AutoCommandBufferBuilder,
        input: &IoFragment,
    ) -> IoFragment {
        let local_size = 16;

        let pipeline = {
            let shader = cs::load(ctx.device.clone()).unwrap();
            ComputePipeline::new(
                ctx.device.clone(),
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
        let output_size = (
            input_img.dimensions().width() / 2,
            input_img.dimensions().height() / 2,
        );
        let output_img = utils::create_storage_image(
            ctx,
            &utils::ImageInfo {
                width: output_size.0,
                height: output_size.1,
                format: input_img.format(),
            },
        );

        // setup layout
        let layout = pipeline.layout().set_layouts().get(0).unwrap();
        let input_img_view = ImageView::new_default(input_img.clone()).unwrap();
        let output_img_view = ImageView::new_default(output_img.clone()).unwrap();

        let set = PersistentDescriptorSet::new(
            &ctx.memory.descriptor_set_allocator,
            layout.clone(),
            [
                WriteDescriptorSet::image_view(0, input_img_view),
                WriteDescriptorSet::image_view(1, output_img_view),
            ],
        )
        .unwrap();

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
            output: Io::Image(output_img.clone()),
            label: utils::basic_label("Pooling", &output_img),
        }
    }
}
