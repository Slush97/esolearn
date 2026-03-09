pub mod shape;

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::backend::DeviceBackend;
use shape::Shape;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

/// Unique identifier for a tensor, used as key in autograd tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

impl TensorId {
    fn next() -> Self {
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// N-dimensional tensor backed by a device-specific storage.
/// Backend is passed as parameter to ops, not stored in the tensor.
#[derive(Clone, Debug)]
pub struct Tensor<B: DeviceBackend> {
    pub id: TensorId,
    pub data: B::Storage,
    pub shape: Shape,
}

impl<B: DeviceBackend> Tensor<B> {
    pub fn new(data: B::Storage, shape: Shape) -> Self {
        Self {
            id: TensorId::next(),
            data,
            shape,
        }
    }

    pub fn zeros(shape: Shape) -> Self {
        let data = B::zeros(&shape);
        Self::new(data, shape)
    }

    pub fn ones(shape: Shape) -> Self {
        let data = B::ones(&shape);
        Self::new(data, shape)
    }

    /// Creates a tensor from a `Vec<f32>` and a [`Shape`].
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal `shape.numel()`.
    pub fn from_vec(data: Vec<f32>, shape: Shape) -> Self {
        assert_eq!(
            data.len(),
            shape.numel(),
            "Tensor::from_vec: data length {} != shape numel {}",
            data.len(),
            shape.numel()
        );
        let storage = B::from_vec(data, &shape);
        Self::new(storage, shape)
    }

    pub fn to_vec(&self) -> Vec<f32> {
        B::to_vec(&self.data)
    }

    /// Consume the tensor and return the underlying data as `Vec<f32>`.
    /// Avoids cloning when `Storage = Vec<f32>` (e.g. CpuBackend).
    pub fn into_vec(self) -> Vec<f32> {
        B::into_vec(self.data)
    }

    /// Borrow the tensor data as a `&[f32]` slice without cloning.
    /// Returns `Cow::Borrowed` on CpuBackend, `Cow::Owned` on GPU backends.
    pub fn as_slice(&self) -> std::borrow::Cow<'_, [f32]> {
        B::as_slice(&self.data)
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }
}
