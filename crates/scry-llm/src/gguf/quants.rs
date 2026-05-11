// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dequantization routines for GGUF tensor formats.
//!
//! Each `dequant_*` function consumes the on-disk bytes for one tensor
//! (laid out as `n_blocks` contiguous blocks of [`GgufDtype::block_size_bytes`])
//! and produces `n_elems` `f32` values in row-major order.
//!
//! Reference: llama.cpp `ggml-quants.c`, functions `dequantize_row_q4_K`,
//! `dequantize_row_q5_K`, `dequantize_row_q8_0`.

use super::format::GgufDtype;

pub(super) fn dequantize(
    dtype: GgufDtype,
    bytes: &[u8],
    n_elems: usize,
) -> Result<Vec<f32>, String> {
    match dtype {
        GgufDtype::F32 => Ok(dequant_f32(bytes, n_elems)),
        GgufDtype::F16 => Ok(dequant_f16(bytes, n_elems)),
        GgufDtype::BF16 => Ok(dequant_bf16(bytes, n_elems)),
        GgufDtype::Q8_0 => dequant_q8_0(bytes, n_elems),
        GgufDtype::Q4K => dequant_q4_k(bytes, n_elems),
        GgufDtype::Q5K => dequant_q5_k(bytes, n_elems),
        GgufDtype::Q6K => dequant_q6_k(bytes, n_elems),
        GgufDtype::Other(id) => Err(format!("dequant for ggml_type={id} not implemented")),
    }
}

fn dequant_f32(bytes: &[u8], n_elems: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_elems);
    for chunk in bytes.chunks_exact(4).take(n_elems) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

fn dequant_f16(bytes: &[u8], n_elems: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_elems);
    for chunk in bytes.chunks_exact(2).take(n_elems) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(f16_to_f32(bits));
    }
    out
}

fn dequant_bf16(bytes: &[u8], n_elems: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_elems);
    for chunk in bytes.chunks_exact(2).take(n_elems) {
        // bf16 is the high 16 bits of an IEEE-754 binary32; pad with zero
        // mantissa bits to reconstruct.
        let bits = (u32::from(chunk[1]) << 24) | (u32::from(chunk[0]) << 16);
        out.push(f32::from_bits(bits));
    }
    out
}

fn dequant_q8_0(bytes: &[u8], n_elems: usize) -> Result<Vec<f32>, String> {
    const BLOCK: usize = 32;
    const BLOCK_BYTES: usize = 34;
    if n_elems % BLOCK != 0 {
        return Err(format!("Q8_0: {n_elems} elements not a multiple of 32"));
    }
    let n_blocks = n_elems / BLOCK;
    if bytes.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q8_0: truncated, need {} bytes, have {}",
            n_blocks * BLOCK_BYTES,
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(n_elems);
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let scale_bits = u16::from_le_bytes([bytes[base], bytes[base + 1]]);
        let scale = f16_to_f32(scale_bits);
        for i in 0..BLOCK {
            let q = bytes[base + 2 + i] as i8;
            out.push(scale * f32::from(q));
        }
    }
    Ok(out)
}

fn dequant_q4_k(bytes: &[u8], n_elems: usize) -> Result<Vec<f32>, String> {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 144;
    if n_elems % QK != 0 {
        return Err(format!("Q4_K: {n_elems} elements not a multiple of 256"));
    }
    let n_blocks = n_elems / QK;
    if bytes.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q4_K: truncated, need {} bytes, have {}",
            n_blocks * BLOCK_BYTES,
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(n_elems);
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([bytes[base + 2], bytes[base + 3]]));
        let scales = &bytes[base + 4..base + 16]; // 12 bytes
        let qs = &bytes[base + 16..base + 144]; // 128 bytes
                                                // Process the 256 elements in four chunks of 64. Each chunk uses
                                                // two sub-block indices (is, is+1) reading low and high nibbles of
                                                // the same 32-byte qs slice.
        let mut is: usize = 0;
        for chunk in 0..4 {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let (sc1, m1) = get_scale_min_k4(is + 1, scales);
            let d1 = d * f32::from(sc0);
            let m1f = dmin * f32::from(m0);
            let d2 = d * f32::from(sc1);
            let m2f = dmin * f32::from(m1);
            let q_off = chunk * 32;
            for l in 0..32 {
                out.push(d1 * f32::from(qs[q_off + l] & 0x0F) - m1f);
            }
            for l in 0..32 {
                out.push(d2 * f32::from(qs[q_off + l] >> 4) - m2f);
            }
            is += 2;
        }
    }
    Ok(out)
}

/// Unpack the 6-bit scale + 6-bit min for sub-block `j` (0..8) from the
/// 12-byte packed `scales` array of a Q4_K / Q5_K block.
///
/// Matches `llama.cpp/ggml-quants.c::get_scale_min_k4`:
/// ```text
/// j < 4 : d = q[j]   & 63;          m = q[j+4] & 63
/// j ≥ 4 : d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
///         m = (q[j+4] >> 4 ) | ((q[j  ] >> 6) << 4)
/// ```
/// The high 2 bits of `q[0..4]` carry the high bits of `d[4..8]`;
/// the high 2 bits of `q[4..8]` carry the high bits of `m[4..8]`;
/// `q[8..12]` carry the low 4 bits of both `d[4..8]` (low nibble) and
/// `m[4..8]` (high nibble).
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        let d = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        let m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (d, m)
    }
}

fn dequant_q5_k(bytes: &[u8], n_elems: usize) -> Result<Vec<f32>, String> {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 176;
    if n_elems % QK != 0 {
        return Err(format!("Q5_K: {n_elems} elements not a multiple of 256"));
    }
    let n_blocks = n_elems / QK;
    if bytes.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q5_K: truncated, need {} bytes, have {}",
            n_blocks * BLOCK_BYTES,
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(n_elems);
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        let d = f16_to_f32(u16::from_le_bytes([bytes[base], bytes[base + 1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([bytes[base + 2], bytes[base + 3]]));
        let scales = &bytes[base + 4..base + 16]; // 12 bytes
        let qh = &bytes[base + 16..base + 48]; // 32 bytes — one high bit per element
        let qs = &bytes[base + 48..base + 176]; // 128 bytes — 4-bit lows
                                                // qh[l] holds high bits for 8 elements (one per chunk-lane).
                                                // For chunk c (0..4), low lane reads bit `2*c`, high lane reads bit `2*c + 1`.
        let mut is: usize = 0;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for chunk in 0..4 {
            let (sc0, m0) = get_scale_min_k4(is, scales);
            let (sc1, m1) = get_scale_min_k4(is + 1, scales);
            let d1 = d * f32::from(sc0);
            let m1f = dmin * f32::from(m0);
            let d2 = d * f32::from(sc1);
            let m2f = dmin * f32::from(m1);
            let q_off = chunk * 32;
            for l in 0..32 {
                let low = qs[q_off + l] & 0x0F;
                let high = if qh[l] & u1 != 0 { 16 } else { 0 };
                out.push(d1 * f32::from(low + high) - m1f);
            }
            for l in 0..32 {
                let low = qs[q_off + l] >> 4;
                let high = if qh[l] & u2 != 0 { 16 } else { 0 };
                out.push(d2 * f32::from(low + high) - m2f);
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
    Ok(out)
}

fn dequant_q6_k(bytes: &[u8], n_elems: usize) -> Result<Vec<f32>, String> {
    const QK: usize = 256;
    const BLOCK_BYTES: usize = 210;
    if n_elems % QK != 0 {
        return Err(format!("Q6_K: {n_elems} elements not a multiple of 256"));
    }
    let n_blocks = n_elems / QK;
    if bytes.len() < n_blocks * BLOCK_BYTES {
        return Err(format!(
            "Q6_K: truncated, need {} bytes, have {}",
            n_blocks * BLOCK_BYTES,
            bytes.len()
        ));
    }
    let mut out = vec![0.0f32; n_elems];
    for b in 0..n_blocks {
        let base = b * BLOCK_BYTES;
        // Layout: ql[128] | qh[64] | scales[16, i8] | f16 d
        let ql = &bytes[base..base + 128];
        let qh = &bytes[base + 128..base + 192];
        let sc_raw = &bytes[base + 192..base + 208];
        let d = f16_to_f32(u16::from_le_bytes([bytes[base + 208], bytes[base + 209]]));
        let out_off = b * QK;
        // Two 128-element chunks per block; within each chunk, four 32-element
        // lanes interleave low/high nibbles of ql with pairs of bits from qh.
        // The signed quant is recentered by -32 so q ∈ [-32, +31].
        for chunk in 0..2 {
            let ql_off = chunk * 64;
            let qh_off = chunk * 32;
            let sc_off = chunk * 8;
            let y_off = out_off + chunk * 128;
            for l in 0..32_usize {
                let is = l / 16; // 0 for l<16, 1 for l>=16 (selects scale sub-block)
                let q1 =
                    i32::from((ql[ql_off + l] & 0x0F) | (((qh[qh_off + l] >> 0) & 0x03) << 4)) - 32;
                let q2 =
                    i32::from((ql[ql_off + l + 32] & 0x0F) | (((qh[qh_off + l] >> 2) & 0x03) << 4))
                        - 32;
                let q3 =
                    i32::from((ql[ql_off + l] >> 4) | (((qh[qh_off + l] >> 4) & 0x03) << 4)) - 32;
                let q4 =
                    i32::from((ql[ql_off + l + 32] >> 4) | (((qh[qh_off + l] >> 6) & 0x03) << 4))
                        - 32;
                let s0 = f32::from(sc_raw[sc_off + is] as i8);
                let s1 = f32::from(sc_raw[sc_off + is + 2] as i8);
                let s2 = f32::from(sc_raw[sc_off + is + 4] as i8);
                let s3 = f32::from(sc_raw[sc_off + is + 6] as i8);
                out[y_off + l] = d * s0 * q1 as f32;
                out[y_off + l + 32] = d * s1 * q2 as f32;
                out[y_off + l + 64] = d * s2 * q3 as f32;
                out[y_off + l + 96] = d * s3 * q4 as f32;
            }
        }
    }
    Ok(out)
}

/// IEEE-754 binary16 → binary32 conversion. Handles subnormals, infinities,
/// and NaNs correctly.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = u32::from(bits & 0x3FF);
    let bits32 = match exp {
        0 => {
            // Subnormal or zero. Renormalize to f32's hidden-bit form.
            //
            // f16 subnormal value = mant * 2^-24 with 1 ≤ mant ≤ 0x3FF.
            // Left-shift `m` until bit 10 is set (the f16 hidden-bit position);
            // each shift contributes one factor of 2 to the magnitude, so the
            // final unbiased f32 exponent is -14 - shifts and the new
            // 10-bit mantissa is `m & 0x3FF`.
            //
            // (We previously initialised `e = -1` and adjusted by `127 - 15`,
            // which is the *normal* f16 → f32 bias offset; that produced
            // values 4× too small for subnormals — Q6_K blocks store `d` as
            // a subnormal f16 for small-magnitude weights and exercise this
            // path, so the bug was latent until M14.)
            if mant == 0 {
                sign
            } else {
                let mut shifts: i32 = 0;
                let mut m = mant;
                while m & 0x400 == 0 {
                    m <<= 1;
                    shifts += 1;
                }
                m &= 0x3FF;
                let exp32 = ((127 - 14 - shifts) as u32) << 23;
                sign | exp32 | (m << 13)
            }
        }
        0x1F => sign | 0x7F80_0000 | (mant << 13),
        _ => sign | ((u32::from(exp) + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_known_values() {
        // 0.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // 1.0
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        // -1.0
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        // 2.0
        assert_eq!(f16_to_f32(0x4000), 2.0);
        // 0.5
        assert_eq!(f16_to_f32(0x3800), 0.5);
        // smallest positive normal: 2^-14
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
    }

    #[test]
    fn f16_subnormals() {
        // Smallest positive subnormal: 0x0001 = 1 * 2^-24
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        // Negative smallest subnormal
        assert_eq!(f16_to_f32(0x8001), -(2f32.powi(-24)));
        // Mid subnormal: 0x0101 = 257 * 2^-24 ≈ 1.5318e-5
        // (This is the failure-causing value from Q6_K block 0's `d`.)
        let v = f16_to_f32(0x0101);
        let expected = 257.0 * 2f32.powi(-24);
        assert!(
            (v - expected).abs() < 1e-12,
            "0x0101 → {v}, expected {expected}"
        );
        // Largest subnormal: 0x03FF = 1023 * 2^-24
        assert_eq!(f16_to_f32(0x03FF), 1023.0 * 2f32.powi(-24));
        // Just below smallest normal: 0x03FF < 0x0400
        assert!(f16_to_f32(0x03FF) < f16_to_f32(0x0400));
    }

    #[test]
    fn bf16_roundtrip_through_buffer() {
        // 1.0 in bf16 is 0x3F80 (= 1.0_f32 with the low 16 mantissa bits chopped).
        // Stored little-endian: [0x80, 0x3F].
        let out = dequant_bf16(&[0x80, 0x3F], 1);
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn q8_0_basic_block() {
        // One block: scale = 1.0 (f16: 0x3C00 = [0x00, 0x3C]) + 32 quants.
        let mut bytes = vec![0x00, 0x3C];
        bytes.extend((0..32_i8).map(|i| (i * 2) as u8));
        let out = dequant_q8_0(&bytes, 32).unwrap();
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, f32::from(i as i8 * 2));
        }
    }

    #[test]
    fn get_scale_min_k4_known_layout() {
        // All scales = 63, all mins = 0:
        //   bytes 0..4 = 0xFF (low 6 bits = 63 for d[0..4]; top 2 bits = 0b11 for d[4..8])
        //   bytes 4..8 = 0x00 (m[0..4] = 0 and m[4..8] high bits = 0)
        //   bytes 8..12 = 0x0F (d[4..8] low nibble = 0xF; m[4..8] high nibble = 0)
        let scales = [0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0x0F, 0x0F, 0x0F, 0x0F];
        for j in 0..8 {
            let (d, m) = get_scale_min_k4(j, &scales);
            assert_eq!(d, 63, "sub-block {j} scale");
            assert_eq!(m, 0, "sub-block {j} min");
        }

        // All scales = 0, all mins = 63: flip the role of high bits.
        //   bytes 0..4 = 0 (no high bits, low bits = 0)
        //   bytes 4..8 = 0xFF (low 6 = 63 for m[0..4]; top 2 = 0b11 for m[4..8])
        //   bytes 8..12 = 0xF0 (d[4..8] low nibble = 0; m[4..8] high nibble = 0xF)
        let scales = [0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xF0, 0xF0, 0xF0, 0xF0];
        for j in 0..8 {
            let (d, m) = get_scale_min_k4(j, &scales);
            assert_eq!(d, 0, "sub-block {j} scale");
            assert_eq!(m, 63, "sub-block {j} min");
        }
    }

    #[test]
    fn q4_k_synthetic_block() {
        // Construct a one-superblock Q4_K tensor with:
        //   d = 1.0, dmin = 0.0  → output = sc[j] * (q & 0xF) for each element
        //   all sub-block scales = 63 (encoded as above)
        //   all sub-block mins   = 0
        //   all qs bytes = 0xF0  → low nibble = 0, high nibble = 0xF
        // Expected per element (256 total):
        //   sub-block 0,2,4,6 (low-nibble lanes): 63 * 0 = 0
        //   sub-block 1,3,5,7 (high-nibble lanes): 63 * 15 = 945
        let mut bytes = Vec::with_capacity(144);
        // d = 1.0 (f16 0x3C00, LE = [0x00, 0x3C])
        bytes.extend_from_slice(&[0x00, 0x3C]);
        // dmin = 0.0
        bytes.extend_from_slice(&[0x00, 0x00]);
        // scales: all scales=63, all mins=0
        bytes.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0, 0x0F, 0x0F, 0x0F, 0x0F]);
        // qs: 128 bytes of 0xF0
        bytes.extend(std::iter::repeat_n(0xF0_u8, 128));
        assert_eq!(bytes.len(), 144);

        let out = dequant_q4_k(&bytes, 256).unwrap();
        assert_eq!(out.len(), 256);
        // The dequant order is: for each chunk of 64 elements, low-nibble
        // lane (32) then high-nibble lane (32). So elements 0..32 = 0,
        // 32..64 = 945, 64..96 = 0, 96..128 = 945, etc.
        for (i, &v) in out.iter().enumerate() {
            let expected = if (i / 32) % 2 == 0 { 0.0 } else { 945.0 };
            assert_eq!(v, expected, "elem {i}");
        }
    }

    #[test]
    fn q4_k_with_min_subtraction() {
        // d = 0.0, dmin = 1.0, all sub-block scales = 0, all mins = 7, q = anything
        //   → output = 0 - 1.0 * 7 = -7 for every element
        let mut bytes = Vec::with_capacity(144);
        bytes.extend_from_slice(&[0x00, 0x00]); // d = 0.0
        bytes.extend_from_slice(&[0x00, 0x3C]); // dmin = 1.0
                                                // All scales = 0, all mins = 7 (= 0b000111, low 6 bits only).
                                                //   bytes 0..4 = 0 (d[0..4] = 0, d[4..8] high bits = 0)
                                                //   bytes 4..8 = 7 (m[0..4] = 7, m[4..8] high bits = 0)
                                                //   bytes 8..12 = 0x70 (d[4..8] low nibble = 0; m[4..8] high nibble = 0x7)
        bytes.extend_from_slice(&[0, 0, 0, 0, 7, 7, 7, 7, 0x70, 0x70, 0x70, 0x70]);
        bytes.extend(std::iter::repeat_n(0xAB_u8, 128));

        let out = dequant_q4_k(&bytes, 256).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, -7.0, "elem {i}");
        }
    }

    #[test]
    fn q5_k_synthetic_block() {
        // d = 1.0, dmin = 0.0; all scales = 1, all mins = 0; all qs = 0
        // (so low 4 bits = 0); all qh = 0xFF (every high bit set).
        // Every element gets weight = 0 + 16 = 16, scale = 1 → output = 16.
        let mut bytes = Vec::with_capacity(176);
        bytes.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
        bytes.extend_from_slice(&[0x00, 0x00]); // dmin = 0.0
                                                // All scales = 1, all mins = 0 (6-bit value 1).
                                                //   bytes 0..4 = 1 (d[0..4] = 1; d[4..8] high bits = 0)
                                                //   bytes 4..8 = 0 (m[0..4] = 0; m[4..8] high bits = 0)
                                                //   bytes 8..12 = 0x01 (d[4..8] low nibble = 1; m[4..8] high nibble = 0)
        bytes.extend_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 0x01, 0x01, 0x01, 0x01]);
        // qh: every high bit set
        bytes.extend(std::iter::repeat_n(0xFF_u8, 32));
        // qs: all zero (low 4 bits of weight = 0)
        bytes.extend(std::iter::repeat_n(0_u8, 128));
        assert_eq!(bytes.len(), 176);

        let out = dequant_q5_k(&bytes, 256).unwrap();
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v, 16.0, "elem {i}");
        }
    }

    #[test]
    fn q5_k_high_bit_lane_routing() {
        // Verifies qh bit routing: only bit 0 of every qh byte set → only
        // chunk 0's low lane gets the +16 bonus. d=1, scales=1, mins=0,
        // qs=0 → first 32 elements = 16, rest = 0.
        let mut bytes = Vec::with_capacity(176);
        bytes.extend_from_slice(&[0x00, 0x3C]); // d = 1.0
        bytes.extend_from_slice(&[0x00, 0x00]); // dmin = 0.0
        bytes.extend_from_slice(&[1, 1, 1, 1, 0, 0, 0, 0, 0x01, 0x01, 0x01, 0x01]);
        bytes.extend(std::iter::repeat_n(0x01_u8, 32)); // only bit 0
        bytes.extend(std::iter::repeat_n(0_u8, 128));

        let out = dequant_q5_k(&bytes, 256).unwrap();
        for i in 0..32 {
            assert_eq!(out[i], 16.0, "chunk 0 low elem {i}");
        }
        for i in 32..256 {
            assert_eq!(out[i], 0.0, "elem {i} should be untouched");
        }
    }

    #[test]
    fn q8_0_negative_quants() {
        // scale = 0.5 (f16: 0x3800 = [0x00, 0x38]), q[0] = -1 → -0.5.
        let mut bytes = vec![0x00, 0x38];
        bytes.push(0xFF); // -1 as i8
        bytes.extend(std::iter::repeat_n(0_u8, 31));
        let out = dequant_q8_0(&bytes, 32).unwrap();
        assert_eq!(out[0], -0.5);
        for v in &out[1..] {
            assert_eq!(*v, 0.0);
        }
    }
}
