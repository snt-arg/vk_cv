use crate::utils::ImageInfo;

#[derive(Debug)]
pub struct OwnedImage {
    pub info: ImageInfo,
    pub buffer: Vec<u8>,
}

pub fn draw_centroid(frame: &mut OwnedImage, centroid: &[f32; 2], width: f32) {
    let bytes = &mut frame.buffer;
    let stride = frame.info.stride() as i32;
    let bpp = frame.info.bytes_per_pixel() as i32;

    // draw a cross at the position of the centroid
    let size = 16;
    let cx = centroid[0] as i32;
    let cy = centroid[1] as i32;
    let w = (width / 2.0).ceil() as i32;

    // vertical line
    for y in cy - size..cy + size {
        for x in cx - w..cx + w {
            let x = x.clamp(0, (frame.info.width - 1) as i32);
            let y = y.clamp(0, (frame.info.height - 1) as i32);
            let offset = x * bpp + y * stride;
            bytes[offset as usize] = 255;
        }
    }

    // horizontal line
    for x in cx - size..cx + size {
        for y in cy - w..cy + w {
            let x = x.clamp(0, (frame.info.width - 1) as i32);
            let y = y.clamp(0, (frame.info.height - 1) as i32);
            let offset = x * bpp + y * stride;
            bytes[offset as usize] = 255;
        }
    }
}
