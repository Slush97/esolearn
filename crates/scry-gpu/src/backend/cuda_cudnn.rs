// SPDX-License-Identifier: MIT OR Apache-2.0
//! cuDNN convolution forward path for the CUDA backend.
//!
//! Holds an [`Arc<Cudnn>`] handle bound to the same stream as cuBLAS, plus a
//! shape-keyed cache of descriptors and pre-selected algorithms. Skips the
//! im2col→cuBLAS lowering kernel that the default path uses, going through
//! cuDNN's fused implicit-GEMM (or Winograd / FFT, picked per shape via the
//! cuDNN heuristic API).
//!
//! Cache discipline: descriptors and algorithm choice are computed on the
//! first call for a given `(input shape, kernel, stride, pad)` and reused
//! thereafter — descriptor creation is cheap but `pick_algorithm` /
//! `get_workspace_size` make extra cuDNN round-trips that are pure overhead
//! at steady state.
//!
//! # Thread safety
//!
//! `cudarc::cudnn::Cudnn` is conservatively marked `!Send`/`!Sync` because
//! cuDNN handles serve a single thread at a time. We wrap the whole state
//! (handle, descriptor cache, workspace) in a [`Mutex`] and unsafely impl
//! `Send`/`Sync` for the wrapper — the lock guarantees only one thread
//! touches the handle at a time, which is the property cuDNN actually wants.
//! cuDNN 8+ is documented as thread-safe under that discipline.
//!
//! # NCHW only
//!
//! Inputs and weights are interpreted as NCHW (filter `[C_out, C_in, K_h, K_w]`).
//! That matches `scry-llm`'s f32-row-major Conv2d weight layout.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::cudnn::safe::{ConvDescriptor, ConvForward, Cudnn, FilterDescriptor, TensorDescriptor};
use cudarc::cudnn::sys;
use cudarc::driver::{CudaSlice, CudaStream};

use crate::backend::cuda::CudaBuffer;
use crate::error::{backend_err, BackendOp, GpuError, Result};

/// Cache key — every shape/conv-param combination gets its own descriptors
/// and algo. We see the same set of shapes loop after loop in inference, so
/// the cache shrinks per-call cuDNN setup to a HashMap lookup.
#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub struct Conv2dKey {
    pub n: u32,
    pub c_in: u32,
    pub h_in: u32,
    pub w_in: u32,
    pub c_out: u32,
    pub k_h: u32,
    pub k_w: u32,
    pub pad_h: u32,
    pub pad_w: u32,
    pub stride_h: u32,
    pub stride_w: u32,
}

struct ConvCacheEntry {
    x_desc: TensorDescriptor<f32>,
    w_desc: FilterDescriptor<f32>,
    y_desc: TensorDescriptor<f32>,
    conv_desc: ConvDescriptor<f32>,
    algo: sys::cudnnConvolutionFwdAlgo_t,
    workspace_bytes: usize,
    h_out: u32,
    w_out: u32,
}

struct CudnnInner {
    cudnn: Arc<Cudnn>,
    cache: HashMap<Conv2dKey, ConvCacheEntry>,
    /// Workspace grows monotonically — kept across calls so we don't free
    /// and re-alloc whenever the largest required size goes up.
    workspace: Option<CudaSlice<u8>>,
}

/// Mutex-protected cuDNN state. The `Send`/`Sync` impl is sound because all
/// access goes through the mutex, satisfying cuDNN's "one-thread-at-a-time
/// per handle" rule.
pub(crate) struct CudnnState {
    inner: Mutex<CudnnInner>,
}

// SAFETY: cuDNN handles and the descriptors derived from them are documented
// as thread-safe so long as concurrent access is serialized. The Mutex around
// CudnnInner provides that serialization. cudarc's safe Cudnn type chose to
// be `!Send`/`!Sync` conservatively, but the underlying handle is a plain
// pointer that can move between threads under our locking discipline.
unsafe impl Send for CudnnState {}
unsafe impl Sync for CudnnState {}

impl CudnnState {
    pub(crate) fn create(stream: Arc<CudaStream>) -> Result<Self> {
        let cudnn = Cudnn::new(stream).map_err(|e| backend_err(BackendOp::CuDnn, e))?;
        Ok(Self {
            inner: Mutex::new(CudnnInner {
                cudnn,
                cache: HashMap::new(),
                workspace: None,
            }),
        })
    }

    /// Run a conv2d forward pass: `output = conv(input, filter)`.
    ///
    /// All shapes in element counts (NOT bytes). Buffers must be sized as
    /// `n*c_in*h_in*w_in`, `c_out*c_in*k_h*k_w`, `n*c_out*h_out*w_out` `f32`s.
    /// The output spatial dims are computed from `(h_in + 2*pad_h - k_h) / stride_h + 1`
    /// (and same for w) — the helper returns them so the caller doesn't have
    /// to recompute.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv2d_forward_async(
        &self,
        stream: &Arc<CudaStream>,
        input: &CudaBuffer,
        filter: &CudaBuffer,
        output: &mut CudaBuffer,
        key: Conv2dKey,
    ) -> Result<(u32, u32)> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| backend_err(BackendOp::MutexPoisoned, "cudnn"))?;

        // Build/cache descriptors and pick the algorithm on first call for
        // this shape. Steady-state path is just a HashMap lookup.
        if !inner.cache.contains_key(&key) {
            let cudnn = Arc::clone(&inner.cudnn);
            let entry = build_cache_entry(&cudnn, key)?;
            inner.cache.insert(key, entry);
        }

        let (needed, h_out, w_out) = {
            let entry = &inner.cache[&key];
            (entry.workspace_bytes, entry.h_out, entry.w_out)
        };

        // Grow workspace if needed. Workspace is scratch-only — cuDNN never
        // requires it to be zeroed, so we use the unsafe `alloc` to skip the
        // `cuMemsetD8Async` round-trip the safe `alloc_zeros` would do.
        let need_grow = inner.workspace.as_ref().is_none_or(|w| w.len() < needed);
        if need_grow && needed > 0 {
            // SAFETY: cuDNN treats the workspace as undefined-contents scratch;
            // it overwrites every byte it reads.
            let buf = unsafe {
                stream
                    .alloc::<u8>(needed)
                    .map_err(|e| backend_err(BackendOp::CreateBuffer, e))?
            };
            inner.workspace = Some(buf);
        }

        // Reinterpret the type-erased u8 buffers as f32 views — cuDNN's
        // `ConvForward::launch` is generic over `DevicePtr<f32>`, which is
        // implemented for `CudaView<'_, f32>`/`CudaViewMut<'_, f32>`. Same
        // bytes, different type-tag.
        let in_len = (key.n * key.c_in * key.h_in * key.w_in) as usize;
        let filter_len = (key.c_out * key.c_in * key.k_h * key.k_w) as usize;
        let out_len = (key.n * key.c_out * h_out * w_out) as usize;

        // SAFETY: the byte buffers are sized for f32 elements and aligned to
        // 4 bytes (CUDA returns 256-byte-aligned allocations).
        let input_view = unsafe { input.inner.transmute::<f32>(in_len) }
            .ok_or_else(|| transmute_err("cudnn input"))?;
        let filter_view = unsafe { filter.inner.transmute::<f32>(filter_len) }
            .ok_or_else(|| transmute_err("cudnn filter"))?;
        let mut output_view = unsafe { output.inner.transmute_mut::<f32>(out_len) }
            .ok_or_else(|| transmute_err("cudnn output"))?;

        let CudnnInner {
            cache, workspace, ..
        } = &mut *inner;
        let entry = &cache[&key];
        let forward = ConvForward {
            conv: &entry.conv_desc,
            x: &entry.x_desc,
            w: &entry.w_desc,
            y: &entry.y_desc,
        };

        // SAFETY: descriptors describe the buffer types/shapes; workspace is
        // sized for the picked algo; cuDNN handle and underlying buffers
        // share this stream.
        unsafe {
            forward
                .launch(
                    entry.algo,
                    workspace.as_mut(),
                    (1.0_f32, 0.0_f32),
                    &input_view,
                    &filter_view,
                    &mut output_view,
                )
                .map_err(|e| backend_err(BackendOp::CuDnn, e))?;
        }

        Ok((h_out, w_out))
    }
}

fn transmute_err(name: &str) -> GpuError {
    GpuError::BackendUnavailable(format!("{name} buffer too small for f32 view"))
}

fn build_cache_entry(cudnn: &Arc<Cudnn>, key: Conv2dKey) -> Result<ConvCacheEntry> {
    let h_out = (key.h_in + 2 * key.pad_h).saturating_sub(key.k_h) / key.stride_h + 1;
    let w_out = (key.w_in + 2 * key.pad_w).saturating_sub(key.k_w) / key.stride_w + 1;

    let x_desc = cudnn
        .create_4d_tensor::<f32>(
            sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [
                key.n as i32,
                key.c_in as i32,
                key.h_in as i32,
                key.w_in as i32,
            ],
        )
        .map_err(|e| backend_err(BackendOp::CuDnn, e))?;

    let w_desc = cudnn
        .create_4d_filter::<f32>(
            sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [
                key.c_out as i32,
                key.c_in as i32,
                key.k_h as i32,
                key.k_w as i32,
            ],
        )
        .map_err(|e| backend_err(BackendOp::CuDnn, e))?;

    let y_desc = cudnn
        .create_4d_tensor::<f32>(
            sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [key.n as i32, key.c_out as i32, h_out as i32, w_out as i32],
        )
        .map_err(|e| backend_err(BackendOp::CuDnn, e))?;

    let conv_desc = cudnn
        .create_conv2d::<f32>(
            [key.pad_h as i32, key.pad_w as i32],
            [key.stride_h as i32, key.stride_w as i32],
            [1, 1],
            sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )
        .map_err(|e| backend_err(BackendOp::CuDnn, e))?;

    let forward = ConvForward {
        conv: &conv_desc,
        x: &x_desc,
        w: &w_desc,
        y: &y_desc,
    };
    let algo = forward
        .pick_algorithm()
        .map_err(|e| backend_err(BackendOp::CuDnn, e))?;
    let workspace_bytes = forward
        .get_workspace_size(algo)
        .map_err(|e| backend_err(BackendOp::CuDnn, e))?;

    Ok(ConvCacheEntry {
        x_desc,
        w_desc,
        y_desc,
        conv_desc,
        algo,
        workspace_bytes,
        h_out,
        w_out,
    })
}
