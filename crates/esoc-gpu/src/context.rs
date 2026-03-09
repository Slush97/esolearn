// SPDX-License-Identifier: MIT OR Apache-2.0
//! GPU context: device, queue, texture format.

use std::sync::Arc;

/// Holds the wgpu device, queue, and preferred texture format.
pub struct GpuContext {
    /// The wgpu device.
    pub device: Arc<wgpu::Device>,
    /// The command queue.
    pub queue: Arc<wgpu::Queue>,
    /// Preferred output texture format.
    pub format: wgpu::TextureFormat,
}

impl GpuContext {
    /// Create a GPU context for headless (offscreen) rendering.
    pub fn new_headless() -> Option<Self> {
        pollster::block_on(Self::new_headless_async())
    }

    async fn new_headless_async() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("esoc-gpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    ..Default::default()
                },
                None,
            )
            .await
            .ok()?;

        Some(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        })
    }

    /// Create a GPU context from an existing surface (for windowed rendering).
    pub fn from_surface(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            device,
            queue,
            format,
        }
    }
}
