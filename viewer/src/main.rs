use eframe::egui;
use egui::{Color32, ColorImage};
use egui_extras::RetainedImage;
use pipeline::Config;
use vkcv::utils::image_to_rgba8;

mod pipeline;

fn main() {
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(1400.0, 1000.0)),
        ..Default::default()
    };

    eframe::run_native(
        "vkcv viewer",
        options,
        Box::new(|_cc| Box::new(MyApp::default())),
    );
}

struct MyApp {
    image: RetainedImage,
    pipeline: pipeline::Pipeline,
    pipeline_cfg: Config,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            image: RetainedImage::from_color_image(
                "",
                ColorImage::new([32, 32], Color32::from_rgb(0, 0, 0)),
            ),
            pipeline: pipeline::Pipeline::new(),
            pipeline_cfg: Default::default(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let mut recreate_pipeline = false;

            let res = self.pipeline.fetch_and_process();

            ui.heading("pipeline output");
            ui.label(format!(
                "{}x{}x{}, {:?}",
                res.image.info.width,
                res.image.info.height,
                res.image.info.bytes_per_pixel(),
                res.image.info.format
            ));

            ui.horizontal(|ui| {
                self.image = RetainedImage::from_color_image(
                    "",
                    ColorImage::from_rgba_unmultiplied(
                        [
                            res.image.info.width as usize,
                            res.image.info.height as usize,
                        ],
                        &res.image.buffer,
                    ),
                );
                self.image.show(ui);

                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.label("MSV min");
                        if ui
                            .color_edit_button_hsva(&mut self.pipeline_cfg.hsv_min)
                            .changed()
                        {
                            recreate_pipeline = true;
                        }
                    });

                    ui.horizontal(|ui| {
                        ui.label("MSV max");
                        if ui
                            .color_edit_button_hsva(&mut self.pipeline_cfg.hsv_max)
                            .changed()
                        {
                            recreate_pipeline = true;
                        }
                    });
                });
            });

            match res.target_pos {
                Some(point) => ui.heading(format!(
                    "X: {:.2}, Y: {:.2}, Z: {:.2}, Area: {}, dt= {:.2}ms",
                    point[0],
                    point[1],
                    point[2],
                    res.area,
                    res.dt.as_secs_f64() / 1e3
                )),
                None => ui.heading("Nothing detected"),
            };

            ui.separator();

            egui::ScrollArea::new([false, true]).show(ui, |ui| {
                egui::Grid::new("grid").striped(true).show(ui, |ui| {
                    for (i, dl) in self.pipeline.download.iter().enumerate() {
                        ui.vertical(|ui| {
                            let input_image = dl.transferred_image();
                            ui.label(format!(
                                "{}x{}x{}, {:?}",
                                input_image.info().width,
                                input_image.info().height,
                                input_image.info().bytes_per_pixel(),
                                input_image.info().format
                            ));
                            let (info, rgba) =
                                image_to_rgba8(input_image.info(), input_image.buffer_content());
                            let input_image = RetainedImage::from_color_image(
                                "",
                                ColorImage::from_rgba_unmultiplied(
                                    [info.width as usize, info.height as usize],
                                    &rgba,
                                ),
                            );
                            input_image.show(ui);
                        });

                        if i % 2 == 1 {
                            ui.end_row();
                        }
                    }
                })
            });

            if recreate_pipeline {
                self.pipeline.reconfigure(&self.pipeline_cfg);
            }

            ctx.request_repaint();
        });
    }
}
