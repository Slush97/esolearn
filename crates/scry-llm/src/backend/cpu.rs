use rayon::prelude::*;

use crate::backend::{DeviceBackend, MathBackend};
use crate::tensor::shape::Shape;

/// CPU reference backend. Correctness over performance.
pub struct CpuBackend;

/// Minimum number of rows (row-tiles) before engaging rayon for matmul.
#[cfg(not(any(feature = "blas", feature = "mkl")))]
const MATMUL_PAR_THRESHOLD: usize = 128;
/// Minimum total element count before parallelizing any operation.
/// Rayon dispatch overhead (~50-100µs on 16 threads) requires at least this
/// much compute to amortize. At ~1-2ns/element for softmax/gelu/add, this
/// threshold corresponds to ~130-260µs of sequential work.
const PAR_THRESHOLD: usize = 262_144;

impl DeviceBackend for CpuBackend {
    type Storage = Vec<f32>;
    type Stream = ();
    #[cfg(feature = "quantize")]
    type I8Storage = Vec<i8>;

    #[cfg(feature = "quantize")]
    fn i8_from_vec(data: Vec<i8>) -> Vec<i8> { data }
    #[cfg(feature = "quantize")]
    fn i8_to_vec(storage: &Vec<i8>) -> Vec<i8> { storage.clone() }

    fn zeros(shape: &Shape) -> Vec<f32> {
        vec![0.0; shape.numel()]
    }

    fn ones(shape: &Shape) -> Vec<f32> {
        vec![1.0; shape.numel()]
    }

    fn from_vec(data: Vec<f32>, _shape: &Shape) -> Vec<f32> {
        data
    }

    fn to_vec(storage: &Vec<f32>) -> Vec<f32> {
        storage.clone()
    }

    fn into_vec(storage: Vec<f32>) -> Vec<f32> {
        storage
    }

    fn as_slice(storage: &Vec<f32>) -> std::borrow::Cow<'_, [f32]> {
        std::borrow::Cow::Borrowed(storage.as_slice())
    }

    fn clone_storage(storage: &Vec<f32>) -> Vec<f32> {
        storage.clone()
    }
}

impl MathBackend for CpuBackend {
    fn matmul_bias(
        a: &Vec<f32>,
        b: &Vec<f32>,
        bias: &Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Vec<f32> {
        matmul_bias_impl(a, b, bias, m, k, n, trans_a, trans_b)
    }

    fn matmul(
        a: &Vec<f32>,
        b: &Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Vec<f32> {
        matmul_tiled(a, b, m, k, n, trans_a, trans_b)
    }

    fn add_inplace(dst: &mut Vec<f32>, src: &Vec<f32>) {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d += s;
        }
    }

    fn add(
        a: &Vec<f32>,
        b: &Vec<f32>,
        a_shape: &Shape,
        b_shape: &Shape,
        out_shape: &Shape,
    ) -> Vec<f32> {
        let out_numel = out_shape.numel();

        // Fast path: same shape — direct elementwise add (no broadcast math)
        if a_shape == b_shape {
            if out_numel >= PAR_THRESHOLD {
                return a.par_iter().zip(b.par_iter()).map(|(&x, &y)| x + y).collect();
            }
            return a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
        }

        // Fast path: row broadcast [N,M] + [1,M] — add same row to every row
        let a_dims = a_shape.dims();
        let b_dims = b_shape.dims();
        let out_dims = out_shape.dims();
        if out_dims.len() == 2 {
            let (rows, cols) = (out_dims[0], out_dims[1]);
            // b is [1, M] or [M] broadcast over [N, M]
            if a_dims == out_dims && b.len() == cols
                && (b_dims == [1, cols] || b_dims == [cols])
            {
                let mut result = vec![0.0f32; out_numel];
                for r in 0..rows {
                    let row_start = r * cols;
                    for c in 0..cols {
                        result[row_start + c] = a[row_start + c] + b[c];
                    }
                }
                return result;
            }
            // a is [1, M] or [M] broadcast over [N, M]
            if b_dims == out_dims && a.len() == cols
                && (a_dims == [1, cols] || a_dims == [cols])
            {
                let mut result = vec![0.0f32; out_numel];
                for r in 0..rows {
                    let row_start = r * cols;
                    for c in 0..cols {
                        result[row_start + c] = a[c] + b[row_start + c];
                    }
                }
                return result;
            }
            // Column broadcast: [N,M] + [N,1]
            if a_dims == out_dims && b_dims == [rows, 1] {
                let mut result = vec![0.0f32; out_numel];
                for r in 0..rows {
                    let row_start = r * cols;
                    let bv = b[r];
                    for c in 0..cols {
                        result[row_start + c] = a[row_start + c] + bv;
                    }
                }
                return result;
            }
            if b_dims == out_dims && a_dims == [rows, 1] {
                let mut result = vec![0.0f32; out_numel];
                for r in 0..rows {
                    let row_start = r * cols;
                    let av = a[r];
                    for c in 0..cols {
                        result[row_start + c] = av + b[row_start + c];
                    }
                }
                return result;
            }
        }

        // Generic broadcast fallback
        let mut result = vec![0.0; out_numel];
        let a_strides = a_shape.broadcast_strides(out_shape);
        let b_strides = b_shape.broadcast_strides(out_shape);
        let out_strides = out_shape.strides();
        let ndim = out_shape.ndim();

        for idx in 0..out_numel {
            let mut remaining = idx;
            let mut a_offset = 0;
            let mut b_offset = 0;
            for d in 0..ndim {
                let coord = remaining / out_strides[d];
                remaining %= out_strides[d];
                a_offset += coord * a_strides[d];
                b_offset += coord * b_strides[d];
            }
            result[idx] = a[a_offset] + b[b_offset];
        }
        result
    }

    fn reshape_for_heads_from_host(
        data: &[f32],
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> Vec<f32> {
        // Index host slice directly — zero clones for CpuBackend.
        if batch == 1 && seq == 1 {
            return data.to_vec();
        }
        let d_model = n_heads * d_head;
        let total = batch * n_heads * seq * d_head;
        let mut out = vec![0.0f32; total];
        for b in 0..batch {
            for h in 0..n_heads {
                for s in 0..seq {
                    for d in 0..d_head {
                        out[(b * n_heads + h) * seq * d_head + s * d_head + d] =
                            data[(b * seq + s) * d_model + h * d_head + d];
                    }
                }
            }
        }
        out
    }

    fn reshape_for_heads(
        storage: &Vec<f32>,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> Vec<f32> {
        // When B=1, S=1 the reshape is an identity permutation — skip the transpose.
        if batch == 1 && seq == 1 {
            return storage.clone();
        }
        // Index storage directly — no clone needed since Storage = Vec<f32>
        let d_model = n_heads * d_head;
        let total = batch * n_heads * seq * d_head;
        let mut out = vec![0.0f32; total];
        for b in 0..batch {
            for h in 0..n_heads {
                for s in 0..seq {
                    for d in 0..d_head {
                        out[(b * n_heads + h) * seq * d_head + s * d_head + d] =
                            storage[(b * seq + s) * d_model + h * d_head + d];
                    }
                }
            }
        }
        out
    }

    fn reshape_from_heads(
        storage: &Vec<f32>,
        batch: usize,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> Vec<f32> {
        // When B=1, S=1 the reshape is an identity permutation — skip the transpose.
        if batch == 1 && seq == 1 {
            return storage.clone();
        }
        // Index storage directly — no clone needed since Storage = Vec<f32>
        let d_model = n_heads * d_head;
        let total = batch * seq * d_model;
        let mut out = vec![0.0f32; total];
        for b in 0..batch {
            for h in 0..n_heads {
                for s in 0..seq {
                    for d in 0..d_head {
                        out[(b * seq + s) * d_model + h * d_head + d] =
                            storage[(b * n_heads + h) * seq * d_head + s * d_head + d];
                    }
                }
            }
        }
        out
    }

    fn split_qkv_reshape_heads(
        qkv: &Vec<f32>,
        seq: usize,
        n_heads: usize,
        d_head: usize,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let d_model = n_heads * d_head;
        let head_len = n_heads * seq * d_head;
        // Index qkv directly — no clone needed since Storage = Vec<f32>
        let mut q = vec![0.0f32; head_len];
        let mut k = vec![0.0f32; head_len];
        let mut v = vec![0.0f32; head_len];
        for s in 0..seq {
            let row = s * 3 * d_model;
            for h in 0..n_heads {
                for d in 0..d_head {
                    let dst = (h * seq + s) * d_head + d;
                    let src_col = h * d_head + d;
                    q[dst] = qkv[row + src_col];
                    k[dst] = qkv[row + d_model + src_col];
                    v[dst] = qkv[row + 2 * d_model + src_col];
                }
            }
        }
        (q, k, v)
    }

    fn scaled_softmax(input: &Vec<f32>, scale: f32, shape: &Shape) -> Vec<f32> {
        let dims = shape.dims();
        let last = *dims.last().unwrap();
        let mut output = vec![0.0f32; input.len()];

        let process_row = |out_row: &mut [f32], in_row: &[f32]| {
            let max_val = in_row.iter().copied().map(|x| x * scale).fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for i in 0..last {
                let e = f64::from((in_row[i] * scale - max_val).exp());
                out_row[i] = e as f32;
                sum += e;
            }
            for i in 0..last {
                out_row[i] = (f64::from(out_row[i]) / sum) as f32;
            }
        };

        if input.len() >= PAR_THRESHOLD {
            output
                .par_chunks_mut(last)
                .zip(input.par_chunks(last))
                .for_each(|(o, i)| process_row(o, i));
        } else {
            output
                .chunks_mut(last)
                .zip(input.chunks(last))
                .for_each(|(o, i)| process_row(o, i));
        }
        output
    }

    fn softmax(input: &Vec<f32>, shape: &Shape) -> Vec<f32> {
        let dims = shape.dims();
        let last = *dims.last().unwrap();
        let mut output = vec![0.0f32; input.len()];

        let process_row = |out_row: &mut [f32], in_row: &[f32]| {
            let max_val = in_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for i in 0..last {
                let e = f64::from((in_row[i] - max_val).exp());
                out_row[i] = e as f32;
                sum += e;
            }
            for i in 0..last {
                out_row[i] = (f64::from(out_row[i]) / sum) as f32;
            }
        };

        if input.len() >= PAR_THRESHOLD {
            output
                .par_chunks_mut(last)
                .zip(input.par_chunks(last))
                .for_each(|(o, i)| process_row(o, i));
        } else {
            output
                .chunks_mut(last)
                .zip(input.chunks(last))
                .for_each(|(o, i)| process_row(o, i));
        }
        output
    }

    fn layernorm_inference(
        input: &Vec<f32>,
        gamma: &Vec<f32>,
        beta: &Vec<f32>,
        shape: &Shape,
        eps: f32,
    ) -> Vec<f32> {
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let mut output = vec![0.0f32; input.len()];

        let process_row = |i: usize, out_row: &mut [f32]| {
            let start = i * d;
            let slice = &input[start..start + d];

            let mean = slice.iter().map(|&x| f64::from(x)).sum::<f64>() / d as f64;
            let var = slice
                .iter()
                .map(|&x| {
                    let diff = f64::from(x) - mean;
                    diff * diff
                })
                .sum::<f64>()
                / d as f64;
            let rstd = 1.0 / (var + f64::from(eps)).sqrt();

            for j in 0..d {
                let norm = (f64::from(slice[j]) - mean) * rstd;
                out_row[j] = (norm * f64::from(gamma[j]) + f64::from(beta[j])) as f32;
            }
        };

        if input.len() >= PAR_THRESHOLD {
            output
                .par_chunks_mut(d)
                .enumerate()
                .for_each(|(i, out_row)| {
                    process_row(i, out_row);
                });
        } else {
            output
                .chunks_mut(d)
                .enumerate()
                .for_each(|(i, out_row)| {
                    process_row(i, out_row);
                });
        }

        output
    }

    fn layernorm(
        input: &Vec<f32>,
        gamma: &Vec<f32>,
        beta: &Vec<f32>,
        shape: &Shape,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let rows = input.len() / d;
        let mut output = vec![0.0f32; input.len()];
        let mut means = vec![0.0f32; rows];
        let mut rstds = vec![0.0f32; rows];

        let process_row = |i: usize, out_row: &mut [f32], mean_out: &mut f32, rstd_out: &mut f32| {
            let start = i * d;
            let slice = &input[start..start + d];

            let mean = slice.iter().map(|&x| f64::from(x)).sum::<f64>() / d as f64;
            *mean_out = mean as f32;

            let var = slice
                .iter()
                .map(|&x| {
                    let diff = f64::from(x) - mean;
                    diff * diff
                })
                .sum::<f64>()
                / d as f64;

            let rstd = 1.0 / (var + f64::from(eps)).sqrt();
            *rstd_out = rstd as f32;

            for j in 0..d {
                let norm = (f64::from(slice[j]) - mean) * rstd;
                out_row[j] = (norm * f64::from(gamma[j]) + f64::from(beta[j])) as f32;
            }
        };

        if input.len() >= PAR_THRESHOLD {
            output
                .par_chunks_mut(d)
                .zip(means.par_iter_mut().zip(rstds.par_iter_mut()))
                .enumerate()
                .for_each(|(i, (out_row, (mean_out, rstd_out)))| {
                    process_row(i, out_row, mean_out, rstd_out);
                });
        } else {
            output
                .chunks_mut(d)
                .zip(means.iter_mut().zip(rstds.iter_mut()))
                .enumerate()
                .for_each(|(i, (out_row, (mean_out, rstd_out)))| {
                    process_row(i, out_row, mean_out, rstd_out);
                });
        }

        (output, means, rstds)
    }

    fn gelu(input: &Vec<f32>) -> Vec<f32> {
        if input.len() >= PAR_THRESHOLD {
            input.par_iter().map(|&x| gelu_scalar(x)).collect()
        } else {
            input.iter().map(|&x| gelu_scalar(x)).collect()
        }
    }

    fn embedding(weight: &Vec<f32>, indices: &[usize], vocab: usize, dim: usize) -> Vec<f32> {
        assert!(
            indices.iter().all(|&i| i < vocab),
            "embedding index out of range: max index {}, vocab size {}",
            indices.iter().copied().max().unwrap_or(0),
            vocab
        );
        let mut output = vec![0.0; indices.len() * dim];
        for (i, &idx) in indices.iter().enumerate() {
            let src = &weight[idx * dim..(idx + 1) * dim];
            output[i * dim..(i + 1) * dim].copy_from_slice(src);
        }
        output
    }

    fn sum(input: &Vec<f32>) -> f32 {
        let s: f64 = input.iter().map(|&x| f64::from(x)).sum();
        s as f32
    }

    fn mul_elementwise(a: &Vec<f32>, b: &Vec<f32>) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
    }

    fn scale(a: &Vec<f32>, scalar: f32) -> Vec<f32> {
        a.iter().map(|&x| x * scalar).collect()
    }

    fn matmul_strided_batched(
        a: &Vec<f32>,
        b: &Vec<f32>,
        batch_count: usize,
        m: usize,
        k: usize,
        n: usize,
        trans_a: bool,
        trans_b: bool,
    ) -> Vec<f32> {
        matmul_strided_batched_impl(a, b, batch_count, m, k, n, trans_a, trans_b)
    }

    fn concat_rows(
        a: &Vec<f32>,
        b: &Vec<f32>,
        _a_rows: usize,
        _b_rows: usize,
        _cols: usize,
    ) -> Vec<f32> {
        let mut out = Vec::with_capacity(a.len() + b.len());
        out.extend_from_slice(a);
        out.extend_from_slice(b);
        out
    }

    // ---- INT8 quantized matmul ----

    #[cfg(feature = "quantize")]
    fn matmul_i8_f32(
        a: &Vec<f32>,
        b_q: &Vec<i8>,
        scale: f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        matmul_i8_f32_tiled(a, b_q, scale, m, k, n)
    }

    #[cfg(feature = "quantize")]
    fn matmul_i8_f32_bias(
        a: &Vec<f32>,
        b_q: &Vec<i8>,
        scale: f32,
        bias: &Vec<f32>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut c = matmul_i8_f32_tiled(a, b_q, scale, m, k, n);
        for row in 0..m {
            for col in 0..n {
                c[row * n + col] += bias[col];
            }
        }
        c
    }

    // ---- Llama-specific ops ----

    fn rmsnorm(
        input: &Vec<f32>,
        weight: &Vec<f32>,
        shape: &Shape,
        eps: f32,
    ) -> Vec<f32> {
        let dims = shape.dims();
        let d = *dims.last().unwrap();
        let n = input.len() / d;
        let mut output = vec![0.0f32; input.len()];

        for i in 0..n {
            let start = i * d;
            let slice = &input[start..start + d];

            // mean(x^2)
            let mean_sq = slice
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                / d as f64;

            let rstd = 1.0 / (mean_sq + f64::from(eps)).sqrt();

            for j in 0..d {
                output[start + j] = (f64::from(slice[j]) * rstd * f64::from(weight[j])) as f32;
            }
        }

        output
    }

    fn rope(
        input: &Vec<f32>,
        shape: &Shape,
        pos: usize,
        head_dim: usize,
        theta: f32,
    ) -> Vec<f32> {
        let dims = shape.dims();
        let total_elements = input.len();
        let last_dim = *dims.last().unwrap();
        let n_rows = total_elements / last_dim;
        let mut output = input.clone();

        let theta_f64 = f64::from(theta);

        for row in 0..n_rows {
            let row_start = row * last_dim;
            // Apply RoPE to each pair of dimensions within each head
            let n_heads_in_row = last_dim / head_dim;
            for h in 0..n_heads_in_row {
                let head_start = row_start + h * head_dim;
                for i in 0..head_dim / 2 {
                    let freq = 1.0 / theta_f64.powf(2.0 * i as f64 / head_dim as f64);
                    let angle = pos as f64 * freq;
                    let cos_val = angle.cos() as f32;
                    let sin_val = angle.sin() as f32;

                    let idx0 = head_start + 2 * i;
                    let idx1 = head_start + 2 * i + 1;
                    let x0 = input[idx0];
                    let x1 = input[idx1];
                    output[idx0] = x0 * cos_val - x1 * sin_val;
                    output[idx1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }

        output
    }

    fn rope_with_freqs_preloaded(
        input: &Vec<f32>,
        seq: usize,
        n_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freqs: &Vec<f32>,
    ) -> Vec<f32> {
        let total_dim = n_heads * head_dim;
        let mut output = input.clone();
        for s in 0..seq {
            let pos = (start_pos + s) as f64;
            let row_start = s * total_dim;
            for h in 0..n_heads {
                let head_start = row_start + h * head_dim;
                for (i, &freq) in freqs.iter().enumerate() {
                    let angle = pos * f64::from(freq);
                    let cos_val = angle.cos() as f32;
                    let sin_val = angle.sin() as f32;
                    let idx0 = head_start + 2 * i;
                    let idx1 = head_start + 2 * i + 1;
                    let x0 = input[idx0];
                    let x1 = input[idx1];
                    output[idx0] = x0 * cos_val - x1 * sin_val;
                    output[idx1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
        output
    }

    fn swiglu(gate: &Vec<f32>, up: &Vec<f32>) -> Vec<f32> {
        if gate.len() >= PAR_THRESHOLD {
            gate.par_iter()
                .zip(up.par_iter())
                .map(|(&g, &u)| silu_scalar(g) * u)
                .collect()
        } else {
            gate.iter()
                .zip(up.iter())
                .map(|(&g, &u)| silu_scalar(g) * u)
                .collect()
        }
    }

    fn repeat_kv(
        input: &Vec<f32>,
        n_kv_heads: usize,
        n_q_heads: usize,
        seq: usize,
        d_head: usize,
    ) -> Vec<f32> {
        let n_rep = n_q_heads / n_kv_heads;
        if n_rep == 1 {
            return input.clone();
        }

        let head_size = seq * d_head;
        let mut output = vec![0.0f32; n_q_heads * head_size];

        for kv_h in 0..n_kv_heads {
            let src_start = kv_h * head_size;
            let src = &input[src_start..src_start + head_size];
            for r in 0..n_rep {
                let dst_start = (kv_h * n_rep + r) * head_size;
                output[dst_start..dst_start + head_size].copy_from_slice(src);
            }
        }

        output
    }

    fn gather_reshape_repeat_kv(
        cache: &Vec<f32>,
        _max_seq: usize,
        cached_len: usize,
        n_kv_heads: usize,
        n_q_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let n_rep = n_q_heads / n_kv_heads;
        let kv_dim = n_kv_heads * head_dim;
        let total = n_q_heads * cached_len * head_dim;
        let mut out = vec![0.0f32; total];

        for q_head in 0..n_q_heads {
            let kv_head = q_head / n_rep;
            for s in 0..cached_len {
                let dst = (q_head * cached_len + s) * head_dim;
                let src = s * kv_dim + kv_head * head_dim;
                out[dst..dst + head_dim].copy_from_slice(&cache[src..src + head_dim]);
            }
        }

        out
    }
}

// ---- Helper functions ----

/// Fast vector-matrix multiply for single-row matmuls (m == 1).
///
/// Computes `out[j] = sum_p vec[p] * mat[p * n + j]` for row-major `mat[k, n]`.
/// Falls back to a tight scalar loop that auto-vectorizes well on x86-64.
/// Avoids BLAS dispatch overhead which dominates at small dimensions (e.g. 384).
/// Minimum output columns before parallelizing GEMV with rayon.
/// Below this threshold, thread spawn overhead exceeds the parallelism benefit.
const GEMV_PAR_THRESHOLD: usize = 8192;

/// Column chunk size for parallel GEMV — sized to fit in L2 cache.
/// Each thread processes a contiguous slice of output columns.
const GEMV_PAR_CHUNK: usize = 4096;

fn gemv_f32(vec: &[f32], mat: &[f32], k: usize, n: usize) -> Vec<f32> {
    // Parallel path: split output columns across threads for large n.
    // The logit projection [1,384]×[384,51865] is bandwidth-bound at ~1ms;
    // splitting across cores reduces wall time proportionally.
    if n >= GEMV_PAR_THRESHOLD {
        let mut out = vec![0.0f32; n];
        out.par_chunks_mut(GEMV_PAR_CHUNK)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let col_start = chunk_idx * GEMV_PAR_CHUNK;
                let chunk_n = chunk.len();
                let k4 = k & !3;
                for p in (0..k4).step_by(4) {
                    let v0 = vec[p];
                    let v1 = vec[p + 1];
                    let v2 = vec[p + 2];
                    let v3 = vec[p + 3];
                    let base0 = p * n + col_start;
                    let base1 = (p + 1) * n + col_start;
                    let base2 = (p + 2) * n + col_start;
                    let base3 = (p + 3) * n + col_start;
                    for j in 0..chunk_n {
                        chunk[j] += v0 * mat[base0 + j]
                            + v1 * mat[base1 + j]
                            + v2 * mat[base2 + j]
                            + v3 * mat[base3 + j];
                    }
                }
                for p in k4..k {
                    let v = vec[p];
                    let base = p * n + col_start;
                    for j in 0..chunk_n {
                        chunk[j] += v * mat[base + j];
                    }
                }
            });
        return out;
    }

    // Sequential path for small n
    let mut out = vec![0.0f32; n];
    let k4 = k & !3;
    for p in (0..k4).step_by(4) {
        let v0 = vec[p];
        let v1 = vec[p + 1];
        let v2 = vec[p + 2];
        let v3 = vec[p + 3];
        let row0 = &mat[p * n..];
        let row1 = &mat[(p + 1) * n..];
        let row2 = &mat[(p + 2) * n..];
        let row3 = &mat[(p + 3) * n..];
        for j in 0..n {
            out[j] += v0 * row0[j] + v1 * row1[j] + v2 * row2[j] + v3 * row3[j];
        }
    }
    for p in k4..k {
        let v = vec[p];
        let row = &mat[p * n..];
        for j in 0..n {
            out[j] += v * row[j];
        }
    }
    out
}

/// GEMV for `vec[1, k] @ mat_T[n, k]` (trans_b = true, row-major).
/// Equivalent to: for each j, dot(vec, mat_T[j*k..]).
/// Uses 4-wide accumulator unrolling for ILP (matches `gemv_f32`'s pattern).
#[inline]
fn gemv_trans_b_f32(vec: &[f32], mat_t: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    gemv_trans_b_into(vec, mat_t, k, n, &mut out);
    out
}

/// GEMV for `vec[1, k] @ mat_T[n, k]` writing directly into `out` (no allocation).
#[inline]
fn gemv_trans_b_into(vec: &[f32], mat_t: &[f32], k: usize, n: usize, out: &mut [f32]) {
    let k4 = k & !3;
    for j in 0..n {
        let row = &mat_t[j * k..];
        let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for p in (0..k4).step_by(4) {
            a0 += vec[p]     * row[p];
            a1 += vec[p + 1] * row[p + 1];
            a2 += vec[p + 2] * row[p + 2];
            a3 += vec[p + 3] * row[p + 3];
        }
        let mut acc = a0 + a1 + a2 + a3;
        for p in k4..k { acc += vec[p] * row[p]; }
        out[j] = acc;
    }
}

/// GEMV for `vec[1, k] @ mat[k, n]` writing directly into `out` (no allocation).
#[inline]
fn gemv_into(vec: &[f32], mat: &[f32], k: usize, n: usize, out: &mut [f32]) {
    for j in 0..n { out[j] = 0.0; }
    let k4 = k & !3;
    for p in (0..k4).step_by(4) {
        let v0 = vec[p];
        let v1 = vec[p + 1];
        let v2 = vec[p + 2];
        let v3 = vec[p + 3];
        let row0 = &mat[p * n..];
        let row1 = &mat[(p + 1) * n..];
        let row2 = &mat[(p + 2) * n..];
        let row3 = &mat[(p + 3) * n..];
        for j in 0..n {
            out[j] += v0 * row0[j] + v1 * row1[j] + v2 * row2[j] + v3 * row3[j];
        }
    }
    for p in k4..k {
        let v = vec[p];
        let row = &mat[p * n..];
        for j in 0..n {
            out[j] += v * row[j];
        }
    }
}

/// Compute one row-tile of C for the tiled matmul.
#[cfg(not(any(feature = "blas", feature = "mkl")))]
fn matmul_tile_rows(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
    i_tile: usize,
    i_end: usize,
) -> Vec<f64> {
    const T: usize = 32;
    let rows = i_end - i_tile;
    let mut local = vec![0.0f64; rows * n];

    let mut p_tile = 0;
    while p_tile < k {
        let p_end = (p_tile + T).min(k);
        let mut j_tile = 0;
        while j_tile < n {
            let j_end = (j_tile + T).min(n);

            for i in i_tile..i_end {
                let li = i - i_tile;
                for p in p_tile..p_end {
                    let a_val = if trans_a {
                        f64::from(a[p * m + i])
                    } else {
                        f64::from(a[i * k + p])
                    };
                    for j in j_tile..j_end {
                        let b_val = if trans_b {
                            f64::from(b[j * k + p])
                        } else {
                            f64::from(b[p * n + j])
                        };
                        local[li * n + j] += a_val * b_val;
                    }
                }
            }

            j_tile += T;
        }
        p_tile += T;
    }
    local
}

/// Tiled matmul with optional rayon parallelism.
fn matmul_tiled(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    // Fast path: single-row matmul → GEMV (avoids BLAS dispatch overhead).
    // For m=1, GEMV is always faster than sgemm regardless of n — the n dimension
    // is just independent dot products, perfectly sequential and cache-friendly.
    // BLAS sgemm dispatch overhead alone is 50-100µs, and for Whisper's logit
    // projection [1,384]×[51865,384]^T this caused 0.6-11ms variance vs ~0.8ms
    // consistent with GEMV.
    if m == 1 && !trans_a {
        if trans_b {
            return gemv_trans_b_f32(a, b, k, n);
        } else {
            return gemv_f32(a, b, k, n);
        }
    }

    #[cfg(feature = "dnnl")]
    {
        return matmul_dnnl(a, b, m, k, n, trans_a, trans_b, 0.0, &[]);
    }

    #[cfg(any(feature = "blas", feature = "mkl"))]
    {
        return matmul_cblas(a, b, m, k, n, trans_a, trans_b);
    }

    #[cfg(not(any(feature = "blas", feature = "mkl", feature = "dnnl")))]
    {
        matmul_tiled_fallback(a, b, m, k, n, trans_a, trans_b)
    }
}

#[cfg(not(any(feature = "blas", feature = "mkl")))]
fn matmul_tiled_fallback(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    const T: usize = 32;
    let i_tiles: Vec<usize> = (0..m).step_by(T).collect();

    let row_chunks: Vec<Vec<f64>> = if m >= MATMUL_PAR_THRESHOLD {
        i_tiles
            .par_iter()
            .map(|&i_tile| {
                let i_end = (i_tile + T).min(m);
                matmul_tile_rows(a, b, m, k, n, trans_a, trans_b, i_tile, i_end)
            })
            .collect()
    } else {
        i_tiles
            .iter()
            .map(|&i_tile| {
                let i_end = (i_tile + T).min(m);
                matmul_tile_rows(a, b, m, k, n, trans_a, trans_b, i_tile, i_end)
            })
            .collect()
    };

    let mut c = Vec::with_capacity(m * n);
    for chunk in &row_chunks {
        c.extend(chunk.iter().map(|&x| x as f32));
    }
    c
}

/// oneDNN FFI bindings for `dnnl_sgemm`.
#[cfg(feature = "dnnl")]
mod dnnl_ffi {
    // dnnl_dim_t is i64 on 64-bit platforms
    type DnnlDim = i64;

    // dnnl_status_t
    #[allow(non_camel_case_types, dead_code)]
    type DnnlStatus = i32;

    unsafe extern "C" {
        pub fn dnnl_sgemm(
            transa: std::ffi::c_char,
            transb: std::ffi::c_char,
            m: DnnlDim,
            n: DnnlDim,
            k: DnnlDim,
            alpha: f32,
            a: *const f32,
            lda: DnnlDim,
            b: *const f32,
            ldb: DnnlDim,
            beta: f32,
            c: *mut f32,
            ldc: DnnlDim,
        ) -> DnnlStatus;
    }
}

/// Pin BLAS to single-threaded mode on first use.
///
/// BLAS libraries (OpenBLAS, MKL) spawn their own thread pools by default,
/// which contends with rayon's thread pool. Since we manage parallelism at
/// the batch level ourselves, BLAS should run single-threaded per call.
///
/// Respects `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` if set explicitly.
#[cfg(any(feature = "blas", feature = "mkl"))]
fn init_blas_threading() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // Only override if the user hasn't set the env var themselves.
        #[cfg(all(feature = "blas", not(feature = "mkl")))]
        if std::env::var("OPENBLAS_NUM_THREADS").is_err() {
            extern "C" {
                fn openblas_set_num_threads(num: i32);
            }
            unsafe { openblas_set_num_threads(1) };
        }
        #[cfg(feature = "mkl")]
        if std::env::var("MKL_NUM_THREADS").is_err() {
            extern "C" {
                fn mkl_set_num_threads(num: i32);
            }
            unsafe { mkl_set_num_threads(1) };
        }
    });
}

/// BLAS-accelerated matmul via cblas_sgemm.
#[cfg(any(feature = "blas", feature = "mkl"))]
fn matmul_cblas(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
    init_blas_threading();

    let mut c = vec![0.0f32; m * n];

    let (transa, lda) = if trans_a {
        (CBLAS_TRANSPOSE::CblasTrans, m as i32)
    } else {
        (CBLAS_TRANSPOSE::CblasNoTrans, k as i32)
    };

    let (transb, ldb) = if trans_b {
        (CBLAS_TRANSPOSE::CblasTrans, k as i32)
    } else {
        (CBLAS_TRANSPOSE::CblasNoTrans, n as i32)
    };

    unsafe {
        cblas_sgemm(
            CBLAS_LAYOUT::CblasRowMajor,
            transa,
            transb,
            m as i32,   // M
            n as i32,   // N
            k as i32,   // K
            1.0,        // alpha
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            0.0,        // beta
            c.as_mut_ptr(),
            n as i32,   // ldc
        );
    }

    c
}

/// `GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044_715 * x^3)))`
fn gelu_scalar(x: f32) -> f32 {
    let x64 = f64::from(x);
    let sqrt_2_over_pi: f64 = (2.0 / std::f64::consts::PI).sqrt();
    let inner = sqrt_2_over_pi * (x64 + 0.044_715 * x64 * x64 * x64);
    (0.5 * x64 * (1.0 + inner.tanh())) as f32
}

/// `SiLU(x) = x / (1 + exp(-x))`
fn silu_scalar(x: f32) -> f32 {
    let x64 = f64::from(x);
    (x64 / (1.0 + (-x64).exp())) as f32
}

/// Tiled i8×f32 matmul with on-the-fly dequantization.
///
/// `A` is `[m, k]` f32, `B_q` is `[k, n]` i8. Result is `[m, n]` f32.
/// Each i8 element is dequantized as `b_q[idx] as f32 * scale`.
#[cfg(feature = "quantize")]
fn matmul_i8_f32_tiled(
    a: &[f32],
    b_q: &[i8],
    scale: f32,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    const T: usize = 32;
    let mut c = vec![0.0f32; m * n];
    let scale_f64 = f64::from(scale);

    for i_tile in (0..m).step_by(T) {
        let i_end = (i_tile + T).min(m);
        for p_tile in (0..k).step_by(T) {
            let p_end = (p_tile + T).min(k);
            for j_tile in (0..n).step_by(T) {
                let j_end = (j_tile + T).min(n);
                for i in i_tile..i_end {
                    for p in p_tile..p_end {
                        let a_val = f64::from(a[i * k + p]);
                        for j in j_tile..j_end {
                            let b_val = f64::from(b_q[p * n + j]) * scale_f64;
                            c[i * n + j] += (a_val * b_val) as f32;
                        }
                    }
                }
            }
        }
    }
    c
}

/// Fused matmul + bias: C = A @ B + bias (bias broadcast along rows).
fn matmul_bias_impl(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    // Fast path: single-row matmul+bias → GEMV + bias add (small matrices only)
    if m == 1 && !trans_a && k * n <= 1_000_000 {
        let mut c = if trans_b {
            gemv_trans_b_f32(a, b, k, n)
        } else {
            gemv_f32(a, b, k, n)
        };
        for j in 0..n {
            c[j] += bias[j];
        }
        return c;
    }

    #[cfg(feature = "dnnl")]
    {
        return matmul_dnnl(a, b, m, k, n, trans_a, trans_b, 1.0, bias);
    }

    #[cfg(any(feature = "blas", feature = "mkl"))]
    {
        return matmul_bias_cblas(a, b, bias, m, k, n, trans_a, trans_b);
    }

    #[cfg(not(any(feature = "blas", feature = "mkl", feature = "dnnl")))]
    {
        let mut c = matmul_tiled_fallback(a, b, m, k, n, trans_a, trans_b);
        for row in 0..m {
            for col in 0..n {
                c[row * n + col] += bias[col];
            }
        }
        c
    }
}

/// oneDNN sgemm: supports both plain matmul and fused matmul+bias via beta parameter.
/// When `bias` is non-empty and `beta == 1.0`, C is pre-filled with broadcast bias
/// so that `C = alpha*A*B + beta*C` fuses the bias add.
///
/// Note: unlike standard Fortran BLAS, `dnnl_sgemm` is **row-major**.
#[cfg(feature = "dnnl")]
fn matmul_dnnl(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
    beta: f32,
    bias: &[f32],
) -> Vec<f32> {
    let transa_c: std::ffi::c_char = if trans_a { b'T' as _ } else { b'N' as _ };
    let transb_c: std::ffi::c_char = if trans_b { b'T' as _ } else { b'N' as _ };

    // Row-major leading dimensions (same convention as cblas_sgemm CblasRowMajor)
    let lda = if trans_a { m as i64 } else { k as i64 };
    let ldb = if trans_b { k as i64 } else { n as i64 };
    let ldc = n as i64;

    let mut c = if !bias.is_empty() {
        let mut c = Vec::with_capacity(m * n);
        for _ in 0..m {
            c.extend_from_slice(bias);
        }
        c
    } else {
        vec![0.0f32; m * n]
    };

    unsafe {
        let status = dnnl_ffi::dnnl_sgemm(
            transa_c,
            transb_c,
            m as i64,
            n as i64,
            k as i64,
            1.0,         // alpha
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            beta,
            c.as_mut_ptr(),
            ldc,
        );
        debug_assert_eq!(status, 0, "dnnl_sgemm failed with status {status}");
    }

    c
}

/// BLAS-accelerated fused matmul + bias via beta=1.0.
#[cfg(any(feature = "blas", feature = "mkl"))]
fn matmul_bias_cblas(
    a: &[f32],
    b: &[f32],
    bias: &[f32],
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
    init_blas_threading();

    // Pre-fill C with broadcast bias so sgemm's beta=1.0 adds it for free
    let mut c = Vec::with_capacity(m * n);
    for _ in 0..m {
        c.extend_from_slice(bias);
    }

    let (transa, lda) = if trans_a {
        (CBLAS_TRANSPOSE::CblasTrans, m as i32)
    } else {
        (CBLAS_TRANSPOSE::CblasNoTrans, k as i32)
    };

    let (transb, ldb) = if trans_b {
        (CBLAS_TRANSPOSE::CblasTrans, k as i32)
    } else {
        (CBLAS_TRANSPOSE::CblasNoTrans, n as i32)
    };

    unsafe {
        cblas_sgemm(
            CBLAS_LAYOUT::CblasRowMajor,
            transa,
            transb,
            m as i32,
            n as i32,
            k as i32,
            1.0,        // alpha
            a.as_ptr(),
            lda,
            b.as_ptr(),
            ldb,
            1.0,        // beta = 1.0 → C = alpha*A*B + 1.0*C (adds pre-filled bias)
            c.as_mut_ptr(),
            n as i32,
        );
    }

    c
}

/// Zero-copy strided batched matmul — passes slice offsets directly instead of cloning.
fn matmul_strided_batched_impl(
    a: &[f32],
    b: &[f32],
    batch_count: usize,
    m: usize,
    k: usize,
    n: usize,
    trans_a: bool,
    trans_b: bool,
) -> Vec<f32> {
    let a_stride = m * k;
    let b_stride = k * n;
    let c_stride = m * n;

    // Fast path: batched GEMV when m == 1 (decoder attention/projections)
    // Writes directly into pre-allocated output — no per-head Vec allocation.
    if m == 1 && !trans_a {
        let mut c = vec![0.0f32; batch_count * n];
        for i in 0..batch_count {
            let a_slice = &a[i * k..(i + 1) * k];
            let b_slice = &b[i * b_stride..];
            let out_slice = &mut c[i * n..(i + 1) * n];
            if trans_b {
                gemv_trans_b_into(a_slice, b_slice, k, n, out_slice);
            } else {
                gemv_into(a_slice, b_slice, k, n, out_slice);
            }
        }
        return c;
    }

    #[cfg(feature = "dnnl")]
    {
        let mut c = vec![0.0f32; batch_count * c_stride];
        let transa_c: std::ffi::c_char = if trans_a { b'T' as _ } else { b'N' as _ };
        let transb_c: std::ffi::c_char = if trans_b { b'T' as _ } else { b'N' as _ };
        let lda = if trans_a { m as i64 } else { k as i64 };
        let ldb = if trans_b { k as i64 } else { n as i64 };
        let ldc = n as i64;

        if batch_count >= 2 {
            // Parallel across batches — each dnnl_sgemm writes to its own
            // non-overlapping slice so this is safe.
            c.par_chunks_mut(c_stride)
                .enumerate()
                .for_each(|(i, out_chunk)| {
                    unsafe {
                        dnnl_ffi::dnnl_sgemm(
                            transa_c,
                            transb_c,
                            m as i64,
                            n as i64,
                            k as i64,
                            1.0,
                            a[i * a_stride..].as_ptr(),
                            lda,
                            b[i * b_stride..].as_ptr(),
                            ldb,
                            0.0,
                            out_chunk.as_mut_ptr(),
                            ldc,
                        );
                    }
                });
        } else {
            unsafe {
                dnnl_ffi::dnnl_sgemm(
                    transa_c,
                    transb_c,
                    m as i64,
                    n as i64,
                    k as i64,
                    1.0,
                    a.as_ptr(),
                    lda,
                    b.as_ptr(),
                    ldb,
                    0.0,
                    c.as_mut_ptr(),
                    ldc,
                );
            }
        }
        return c;
    }

    #[cfg(any(feature = "blas", feature = "mkl"))]
    {
        use cblas_sys::{cblas_sgemm, CBLAS_LAYOUT, CBLAS_TRANSPOSE};
        init_blas_threading();

        let mut c = vec![0.0f32; batch_count * c_stride];

        let (transa, lda) = if trans_a {
            (CBLAS_TRANSPOSE::CblasTrans, m as i32)
        } else {
            (CBLAS_TRANSPOSE::CblasNoTrans, k as i32)
        };
        let (transb, ldb) = if trans_b {
            (CBLAS_TRANSPOSE::CblasTrans, k as i32)
        } else {
            (CBLAS_TRANSPOSE::CblasNoTrans, n as i32)
        };

        // Sequential across batches — avoids rayon/BLAS thread contention.
        // cblas_sgemm is already highly optimized per call; nesting it inside
        // rayon's thread pool causes scheduling overhead that exceeds the
        // parallelism benefit, especially for the small per-head matrices
        // typical in attention (e.g., [50,64]×[64,50]).
        for i in 0..batch_count {
            let out_offset = i * c_stride;
            unsafe {
                cblas_sgemm(
                    CBLAS_LAYOUT::CblasRowMajor,
                    transa,
                    transb,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a[i * a_stride..].as_ptr(),
                    lda,
                    b[i * b_stride..].as_ptr(),
                    ldb,
                    0.0,
                    c[out_offset..].as_mut_ptr(),
                    n as i32,
                );
            }
        }
        return c;
    }

    #[cfg(not(any(feature = "blas", feature = "mkl", feature = "dnnl")))]
    {
        if batch_count >= 2 {
            let mut c = vec![0.0f32; batch_count * c_stride];
            c.par_chunks_mut(c_stride)
                .enumerate()
                .for_each(|(i, out_chunk)| {
                    let a_slice = &a[i * a_stride..(i + 1) * a_stride];
                    let b_slice = &b[i * b_stride..(i + 1) * b_stride];
                    let tile = matmul_tiled_fallback(a_slice, b_slice, m, k, n, trans_a, trans_b);
                    out_chunk.copy_from_slice(&tile);
                });
            c
        } else {
            matmul_tiled_fallback(a, b, m, k, n, trans_a, trans_b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = CpuBackend::matmul(&a, &b, 2, 2, 2, false, false);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn softmax_basic() {
        let input = vec![1.0, 2.0, 3.0];
        let shape = Shape::new(&[1, 3]);
        let output = CpuBackend::softmax(&input, &shape);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn gelu_zero() {
        assert!((gelu_scalar(0.0)).abs() < 1e-7);
    }

    #[test]
    fn rmsnorm_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let shape = Shape::new(&[1, 4]);
        let output = CpuBackend::rmsnorm(&input, &weight, &shape, 1e-5);
        // mean(x^2) = (1+4+9+16)/4 = 7.5, rstd = 1/sqrt(7.5) ≈ 0.3651
        let expected_rstd = 1.0 / (7.5f64 + 1e-5).sqrt();
        for i in 0..4 {
            let expected = (i + 1) as f64 * expected_rstd;
            assert!((f64::from(output[i]) - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn swiglu_basic() {
        let gate = vec![0.0, 1.0, -1.0];
        let up = vec![1.0, 1.0, 1.0];
        let output = CpuBackend::swiglu(&gate, &up);
        // silu(0) = 0, silu(1) ≈ 0.7311, silu(-1) ≈ -0.2689
        assert!(output[0].abs() < 1e-6);
        assert!((output[1] - 0.7311).abs() < 0.001);
        assert!((output[2] - (-0.2689)).abs() < 0.001);
    }

    #[test]
    fn repeat_kv_passthrough() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1 head, seq=2, d_head=2
        let output = CpuBackend::repeat_kv(&input, 1, 1, 2, 2);
        assert_eq!(output, input);
    }

    #[test]
    fn repeat_kv_doubles() {
        let input = vec![1.0, 2.0, 3.0, 4.0]; // 1 kv_head, seq=2, d_head=2
        let output = CpuBackend::repeat_kv(&input, 1, 2, 2, 2);
        assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_basic() {
        // Just verify it doesn't panic and modifies the values
        let input = vec![1.0, 0.0, 1.0, 0.0]; // 1 row, head_dim=4
        let shape = Shape::new(&[1, 4]);
        let output = CpuBackend::rope(&input, &shape, 1, 4, 10000.0);
        // At pos=1, first pair rotated by theta=1.0*1/(10000^0) = 1.0
        // cos(1) ≈ 0.5403, sin(1) ≈ 0.8415
        assert!((output[0] - 0.5403_f32).abs() < 0.001);
        assert!((output[1] - 0.8415_f32).abs() < 0.001);
    }

    #[test]
    fn matmul_bias_matches_separate() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // [3, 2]
        let bias = vec![10.0, 20.0]; // [2]
        let fused = CpuBackend::matmul_bias(&a, &b, &bias, 2, 3, 2, false, false);
        let separate = CpuBackend::matmul(&a, &b, 2, 3, 2, false, false);
        for i in 0..4 {
            let row = i / 2;
            let col = i % 2;
            let expected = separate[i] + bias[col];
            assert!(
                (fused[i] - expected).abs() < 1e-5,
                "mismatch at [{row},{col}]: fused={}, expected={expected}",
                fused[i]
            );
        }
    }

    #[test]
    fn scaled_softmax_matches_separate() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let shape = Shape::new(&[2, 3]);
        let scale = 0.5;
        let fused = CpuBackend::scaled_softmax(&input, scale, &shape);
        let scaled: Vec<f32> = input.iter().map(|&x| x * scale).collect();
        let reference = CpuBackend::softmax(&scaled, &shape);
        for (i, (&f, &r)) in fused.iter().zip(reference.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-6,
                "mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    #[test]
    fn layernorm_inference_matches() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gamma = vec![1.0, 1.0, 1.0];
        let beta = vec![0.0, 0.0, 0.0];
        let shape = Shape::new(&[2, 3]);
        let (reference, _, _) = CpuBackend::layernorm(&input, &gamma, &beta, &shape, 1e-5);
        let fused = CpuBackend::layernorm_inference(&input, &gamma, &beta, &shape, 1e-5);
        for (i, (&f, &r)) in fused.iter().zip(reference.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-6,
                "mismatch at {i}: fused={f}, reference={r}"
            );
        }
    }

    #[test]
    fn split_qkv_reshape_heads_matches_manual() {
        // seq=2, n_heads=2, d_head=3, d_model=6
        let seq = 2;
        let n_heads = 2;
        let d_head = 3;
        let d_model = n_heads * d_head;
        // qkv: [seq, 3*d_model]
        let qkv: Vec<f32> = (0..seq * 3 * d_model).map(|i| i as f32).collect();

        let (q_heads, k_heads, v_heads) =
            CpuBackend::split_qkv_reshape_heads(&qkv, seq, n_heads, d_head);

        // Manual reference: split then reshape
        let mut q_flat = vec![0.0f32; seq * d_model];
        let mut k_flat = vec![0.0f32; seq * d_model];
        let mut v_flat = vec![0.0f32; seq * d_model];
        for s in 0..seq {
            let row = s * 3 * d_model;
            let dst = s * d_model;
            q_flat[dst..dst + d_model].copy_from_slice(&qkv[row..row + d_model]);
            k_flat[dst..dst + d_model].copy_from_slice(&qkv[row + d_model..row + 2 * d_model]);
            v_flat[dst..dst + d_model].copy_from_slice(&qkv[row + 2 * d_model..row + 3 * d_model]);
        }
        let q_ref = CpuBackend::reshape_for_heads(
            &q_flat, 1, seq, n_heads, d_head,
        );
        let k_ref = CpuBackend::reshape_for_heads(
            &k_flat, 1, seq, n_heads, d_head,
        );
        let v_ref = CpuBackend::reshape_for_heads(
            &v_flat, 1, seq, n_heads, d_head,
        );
        assert_eq!(q_heads, q_ref, "Q mismatch");
        assert_eq!(k_heads, k_ref, "K mismatch");
        assert_eq!(v_heads, v_ref, "V mismatch");
    }

    #[test]
    fn matmul_strided_batched_matches_default() {
        // 3 batches of [2,4] @ [4,3]
        let batch = 3;
        let m = 2;
        let k = 4;
        let n = 3;
        let a: Vec<f32> = (0..batch * m * k).map(|i| (i as f32) * 0.1).collect();
        let b: Vec<f32> = (0..batch * k * n).map(|i| (i as f32) * 0.1).collect();

        let result = CpuBackend::matmul_strided_batched(&a, &b, batch, m, k, n, false, false);

        // Reference: individual matmuls
        for i in 0..batch {
            let a_slice = &a[i * m * k..(i + 1) * m * k];
            let b_slice = &b[i * k * n..(i + 1) * k * n];
            let ref_c = CpuBackend::matmul(
                &a_slice.to_vec(), &b_slice.to_vec(), m, k, n, false, false,
            );
            for j in 0..m * n {
                assert!(
                    (result[i * m * n + j] - ref_c[j]).abs() < 1e-4,
                    "batch {i}, idx {j}: got {}, expected {}",
                    result[i * m * n + j],
                    ref_c[j]
                );
            }
        }
    }

    #[test]
    fn gather_reshape_repeat_kv_matches_3step() {
        // Simulate a cache [max_seq=4, n_kv=2, hd=2] with cached_len=3
        // n_q_heads=4 (n_rep=2)
        let max_seq = 4;
        let n_kv = 2;
        let n_q = 4;
        let hd = 2;
        let kv_dim = n_kv * hd;
        let cached_len = 3;

        // Fill cache: row s, col c = (s * kv_dim + c) as f32
        let mut cache = vec![0.0f32; max_seq * kv_dim];
        for s in 0..max_seq {
            for c in 0..kv_dim {
                cache[s * kv_dim + c] = (s * kv_dim + c) as f32;
            }
        }

        // Fused path
        let fused = CpuBackend::gather_reshape_repeat_kv(&cache, max_seq, cached_len, n_kv, n_q, hd);

        // 3-step reference path
        let gathered = CpuBackend::gather_rows(&cache, max_seq, kv_dim, 0, cached_len);
        let reshaped = CpuBackend::reshape_for_heads(&gathered, 1, cached_len, n_kv, hd);
        let reference = CpuBackend::repeat_kv(&reshaped, n_kv, n_q, cached_len, hd);

        assert_eq!(fused.len(), reference.len());
        for (i, (&f, &r)) in fused.iter().zip(reference.iter()).enumerate() {
            assert!(
                (f - r).abs() < 1e-6,
                "mismatch at index {i}: fused={f}, reference={r}"
            );
        }
    }

    #[cfg(feature = "quantize")]
    #[test]
    fn matmul_i8_f32_matches_f32_matmul() {
        use crate::quantize::quantize_symmetric;

        let m = 4;
        let k = 8;
        let n = 3;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 - 0.6).collect();

        // f32 reference
        let ref_c = CpuBackend::matmul(&a, &b, m, k, n, false, false);

        // Quantize B and compute via i8 path
        let (b_q, meta) = quantize_symmetric(&b);
        let i8_c = CpuBackend::matmul_i8_f32(&a, &b_q, meta.scale, m, k, n);

        // Tolerance: quantization introduces error proportional to absmax/127 * k
        let b_absmax = b.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let tolerance = b_absmax / 127.0 * k as f32 * 0.5; // generous bound
        for i in 0..m * n {
            assert!(
                (ref_c[i] - i8_c[i]).abs() < tolerance,
                "mismatch at {i}: ref={}, i8={}, tol={tolerance}",
                ref_c[i], i8_c[i]
            );
        }
    }

    #[cfg(feature = "quantize")]
    #[test]
    fn matmul_i8_f32_bias_matches_f32() {
        use crate::quantize::quantize_symmetric;

        let m = 2;
        let k = 4;
        let n = 3;
        let a: Vec<f32> = (0..m * k).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| i as f32 * 0.05).collect();
        let bias: Vec<f32> = vec![1.0, 2.0, 3.0];

        let ref_c = CpuBackend::matmul_bias(&a, &b, &bias, m, k, n, false, false);
        let (b_q, meta) = quantize_symmetric(&b);
        let i8_c = CpuBackend::matmul_i8_f32_bias(&a, &b_q, meta.scale, &bias, m, k, n);

        let b_absmax = b.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let tolerance = b_absmax / 127.0 * k as f32 * 0.5;
        for i in 0..m * n {
            assert!(
                (ref_c[i] - i8_c[i]).abs() < tolerance,
                "mismatch at {i}: ref={}, i8={}, tol={tolerance}",
                ref_c[i], i8_c[i]
            );
        }
    }
}
