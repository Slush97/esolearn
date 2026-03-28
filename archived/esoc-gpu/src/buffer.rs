// SPDX-License-Identifier: MIT OR Apache-2.0
//! Dynamic GPU buffer that grows but never shrinks.

/// A GPU buffer that auto-reallocates when data exceeds capacity.
pub struct DynamicBuffer {
    buffer: wgpu::Buffer,
    capacity: u64,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl DynamicBuffer {
    /// Create a new dynamic buffer with an initial capacity in bytes.
    pub fn new(
        device: &wgpu::Device,
        label: &'static str,
        initial_capacity: u64,
        usage: wgpu::BufferUsages,
    ) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: initial_capacity,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity: initial_capacity,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            label,
        }
    }

    /// Write data to the buffer, reallocating if needed.
    pub fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) {
        let size = data.len() as u64;
        if size > self.capacity {
            // Grow to at least 2× or the needed size
            let new_capacity = (self.capacity * 2).max(size);
            self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: new_capacity,
                usage: self.usage,
                mapped_at_creation: false,
            });
            self.capacity = new_capacity;
        }
        if !data.is_empty() {
            queue.write_buffer(&self.buffer, 0, data);
        }
    }

    /// Get the underlying wgpu buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Current capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }
}
