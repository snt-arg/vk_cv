use crate::processing_elements::IoElement;

pub struct ImageUpload {
    io: IoElement,
}

impl ImageUpload {
    pub fn new(io: IoElement) -> Self {
        Self { io }
    }

    pub fn copy_input_data(&self, data: &[u8]) {
        if let Ok(mut lock) = self.io.input_buffer().as_mut().unwrap().write() {
            let len = lock.len();
            if len < data.len() {
                lock.copy_from_slice(&data[0..len]);
            } else {
                lock.copy_from_slice(data);
            }
        }
    }
}
