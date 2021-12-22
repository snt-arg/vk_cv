use std::{fs::File, io::BufWriter, path::Path};

pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl ImageInfo {
    pub fn bytes_count(&self) -> u32 {
        self.width * self.height * self.depth
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

    (
        ImageInfo {
            width: oi.width,
            height: oi.height,
            depth: 4,
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
    if img_info.depth == 4 {
        encoder.set_color(png::ColorType::Rgba);
    } else if img_info.depth == 1 {
        encoder.set_color(png::ColorType::Grayscale);
    }

    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(data).unwrap();
}
