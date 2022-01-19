use crate::processing_elements::{Io, IoFragment};

pub struct ImageUpload {
    io: IoFragment,
}

impl ImageUpload {
    pub fn from_io(io: IoFragment) -> Result<Self, &'static str> {
        match &io.input {
            Io::Buffer(_) => Ok(Self { io }),
            _ => Err("Input needs to be a buffer"),
        }
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
