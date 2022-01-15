use crate::processing_elements::IoFragment;

pub struct ImageUpload {
    io: IoFragment,
}

impl ImageUpload {
    pub fn new(io: IoFragment) -> Self {
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
