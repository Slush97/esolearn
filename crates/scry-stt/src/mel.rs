use std::f64::consts::PI;
use std::sync::OnceLock;

use rayon::prelude::*;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Whisper uses 80 mel bands, 400-sample window (25ms at 16kHz), 160-sample hop (10ms).
pub const WHISPER_N_MELS: usize = 80;
pub const WHISPER_HOP_LENGTH: usize = 160;
pub const WHISPER_N_FFT: usize = 400;
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;
/// Maximum audio chunk duration: 30 seconds.
pub const WHISPER_CHUNK_SAMPLES: usize = WHISPER_SAMPLE_RATE as usize * 30; // 480_000

/// Pad or trim audio samples to exactly one Whisper chunk (30 seconds at 16kHz).
///
/// This matches the Python Whisper pipeline which zero-pads raw audio *before*
/// computing the mel spectrogram, ensuring silence frames receive the correct
/// normalized values.
pub fn pad_or_trim_audio(samples: &[f32]) -> Vec<f32> {
    if samples.len() >= WHISPER_CHUNK_SAMPLES {
        samples[..WHISPER_CHUNK_SAMPLES].to_vec()
    } else {
        let mut out = Vec::with_capacity(WHISPER_CHUNK_SAMPLES);
        out.extend_from_slice(samples);
        out.resize(WHISPER_CHUNK_SAMPLES, 0.0);
        out
    }
}

/// Compute the log-mel spectrogram of an audio signal.
///
/// Input: `samples` — mono PCM audio at 16kHz (f32).
/// Output: `[n_mels, n_frames]` row-major f32 array, where
///         `n_frames = (samples.len() + n_fft) / hop_length`.
///
/// **Important:** For Whisper inference, pad or trim the audio to 30 seconds
/// (480,000 samples) *before* calling this function, using [`pad_or_trim_audio`].
/// This ensures the mel spectrogram has exactly 3000 frames and that silence
/// frames receive correct normalized values.
///
/// This matches the output of `whisper.log_mel_spectrogram()` in the Python reference.
pub fn log_mel_spectrogram(samples: &[f32]) -> MelSpectrogram {
    let n_fft = WHISPER_N_FFT;
    let hop = WHISPER_HOP_LENGTH;
    let n_mels = WHISPER_N_MELS;
    let n_freq = n_fft / 2 + 1; // 201
    let pad = n_fft / 2; // 200 — matches torch.stft center=True

    // Reflect-pad the signal by n_fft/2 on each side (center=True behaviour).
    let padded = reflect_pad(samples, pad);

    // Match torch.stft frame count: (padded_len - n_fft) / hop + 1
    let stft_frames = (padded.len() - n_fft) / hop + 1;
    // Whisper drops the last STFT frame: `stft[..., :-1]`
    let n_frames = stft_frames.saturating_sub(1).max(1);

    // Cached Hann window and mel filterbank (parsed/computed once, reused forever).
    static HANN: OnceLock<Vec<f64>> = OnceLock::new();
    static FILTERS: OnceLock<Vec<[f64; 201]>> = OnceLock::new();
    let window = HANN.get_or_init(|| hann_window(n_fft));
    let filters = FILTERS.get_or_init(|| {
        let raw = whisper_mel_filterbank();
        raw.into_iter()
            .map(|row| {
                let mut arr = [0.0f64; 201];
                arr.copy_from_slice(&row);
                arr
            })
            .collect()
    });

    // Stage 1: parallel STFT → power spectra (flat buffer [n_frames * n_freq]).
    let mut power_buf = vec![0.0f64; n_frames * n_freq];
    power_buf
        .par_chunks_mut(n_freq)
        .enumerate()
        .for_each(|(frame_idx, power_out)| {
            let start = frame_idx * hop;
            // Each thread gets its own FftPlanner + frame buffer.
            let mut planner = FftPlanner::<f64>::new();
            let fft = planner.plan_fft_forward(n_fft);
            let mut frame = vec![Complex::new(0.0, 0.0); n_fft];
            for i in 0..n_fft {
                let sample = if start + i < padded.len() {
                    f64::from(padded[start + i])
                } else {
                    0.0
                };
                frame[i] = Complex::new(sample * window[i], 0.0);
            }
            fft.process(&mut frame);
            for i in 0..n_freq {
                power_out[i] = frame[i].norm_sqr();
            }
        });

    // Stage 2: parallel mel filterbank application + log (over mel bands).
    let mut mel_spec = vec![0.0f32; n_mels * n_frames];
    mel_spec
        .par_chunks_mut(n_frames)
        .enumerate()
        .for_each(|(mel_idx, mel_row)| {
            let filter = &filters[mel_idx];
            for frame_idx in 0..n_frames {
                let power_row = &power_buf[frame_idx * n_freq..][..n_freq];
                let mut energy = 0.0f64;
                for (freq_idx, &coeff) in filter.iter().enumerate() {
                    energy += coeff * power_row[freq_idx];
                }
                mel_row[frame_idx] = energy.max(1e-10).log10() as f32;
            }
        });

    // Normalize: scale to match Whisper's reference implementation
    let max_val = mel_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let clamp_min = max_val - 8.0;
    for v in &mut mel_spec {
        *v = (*v).max(clamp_min);
        *v = (*v + 4.0) / 4.0;
    }

    MelSpectrogram {
        data: mel_spec,
        n_mels,
        n_frames,
    }
}

/// A computed mel spectrogram, stored row-major as `[n_mels, n_frames]`.
#[derive(Clone, Debug)]
pub struct MelSpectrogram {
    /// Row-major data: `data[mel * n_frames + frame]`.
    pub data: Vec<f32>,
    /// Number of mel frequency bands (typically 80).
    pub n_mels: usize,
    /// Number of time frames.
    pub n_frames: usize,
}

impl MelSpectrogram {
    /// Pad or truncate to exactly `target_frames` time frames.
    /// Whisper expects 3000 frames for a 30-second chunk.
    pub fn pad_or_truncate(&self, target_frames: usize) -> Self {
        let mut data = vec![0.0f32; self.n_mels * target_frames];
        let copy_frames = self.n_frames.min(target_frames);
        for mel in 0..self.n_mels {
            let src_start = mel * self.n_frames;
            let dst_start = mel * target_frames;
            data[dst_start..dst_start + copy_frames]
                .copy_from_slice(&self.data[src_start..src_start + copy_frames]);
        }
        Self {
            data,
            n_mels: self.n_mels,
            n_frames: target_frames,
        }
    }
}

/// Reflect-pad a signal by `pad` samples on each side.
///
/// Matches `torch.nn.functional.pad(x, (pad, pad), mode='reflect')` which is
/// what `torch.stft(..., center=True)` applies internally.
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let n = samples.len();
    if n == 0 {
        return vec![0.0; pad * 2];
    }
    let mut out = Vec::with_capacity(n + 2 * pad);
    // Left reflection: samples[pad], samples[pad-1], ..., samples[1]
    for i in (1..=pad).rev() {
        out.push(samples[i.min(n - 1)]);
    }
    out.extend_from_slice(samples);
    // Right reflection: samples[n-2], samples[n-3], ...
    for i in 0..pad {
        let idx = if n >= 2 { n - 2 - i.min(n - 2) } else { 0 };
        out.push(samples[idx]);
    }
    out
}

/// Hann window of length `n`.
fn hann_window(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let x = (PI * i as f64) / n as f64;
            x.sin().powi(2)
        })
        .collect()
}

/// Load Whisper's reference mel filterbank from the embedded binary.
///
/// The binary contains 80×201 f32 values in little-endian row-major order,
/// extracted from `openai/whisper`'s `mel_filters.npz` (the 80-band variant).
fn whisper_mel_filterbank() -> Vec<Vec<f64>> {
    static BYTES: &[u8] = include_bytes!("mel_filters_80.bin");
    const N_MELS: usize = 80;
    const N_FREQ: usize = 201; // n_fft/2 + 1 where n_fft = 400
    assert_eq!(BYTES.len(), N_MELS * N_FREQ * 4);

    let mut filters = Vec::with_capacity(N_MELS);
    for mel in 0..N_MELS {
        let mut row = Vec::with_capacity(N_FREQ);
        for freq in 0..N_FREQ {
            let offset = (mel * N_FREQ + freq) * 4;
            let bytes: [u8; 4] = BYTES[offset..offset + 4].try_into().unwrap();
            row.push(f64::from(f32::from_le_bytes(bytes)));
        }
        filters.push(row);
    }
    filters
}

/// Compute mel filterbank matrix `[n_mels, n_freq]` from scratch using the Slaney formula.
///
/// Returns a vector of `n_mels` filters, each a vector of `n_freq` coefficients.
/// **Note:** This is kept for testing/reference. The main spectrogram path uses
/// `whisper_mel_filterbank()` which loads Whisper's exact pre-computed coefficients.
#[cfg(test)]
fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32) -> Vec<Vec<f64>> {
    let n_freq = n_fft / 2 + 1;
    let sr = f64::from(sample_rate);

    // Mel scale: Hz → mel and mel → Hz
    let hz_to_mel = |f: f64| -> f64 { 2595.0 * (1.0 + f / 700.0).log10() };
    let mel_to_hz = |m: f64| -> f64 { 700.0 * (10.0_f64.powf(m / 2595.0) - 1.0) };

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(sr / 2.0);

    // n_mels + 2 evenly spaced points in mel scale
    let n_points = n_mels + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64)
        .collect();
    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz points to FFT bin indices (fractional)
    let fft_bins: Vec<f64> = hz_points
        .iter()
        .map(|&f| f * n_fft as f64 / sr)
        .collect();

    // Build triangular filters
    let mut filters = vec![vec![0.0f64; n_freq]; n_mels];
    for m in 0..n_mels {
        let left = fft_bins[m];
        let center = fft_bins[m + 1];
        let right = fft_bins[m + 2];

        for k in 0..n_freq {
            let kf = k as f64;
            if kf >= left && kf <= center {
                let denom = center - left;
                if denom > 0.0 {
                    filters[m][k] = (kf - left) / denom;
                }
            } else if kf > center && kf <= right {
                let denom = right - center;
                if denom > 0.0 {
                    filters[m][k] = (right - kf) / denom;
                }
            }
        }

        // Slaney normalization: scale by 2 / (hz_high - hz_low)
        let hz_width = hz_points[m + 2] - hz_points[m];
        if hz_width > 0.0 {
            let norm = 2.0 / hz_width;
            for v in &mut filters[m] {
                *v *= norm;
            }
        }
    }

    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mel_spectrogram_shape() {
        // 1 second of silence at 16kHz
        let samples = vec![0.0f32; 16_000];
        let mel = log_mel_spectrogram(&samples);
        assert_eq!(mel.n_mels, 80);
        // With center padding: padded_len = 16000 + 400 = 16400
        // stft_frames = (16400 - 400) / 160 + 1 = 101, then drop last → 100
        assert_eq!(mel.n_frames, 100);
        assert_eq!(mel.data.len(), 80 * 100);
    }

    #[test]
    fn mel_spectrogram_30s_gives_3000_frames() {
        // Pad audio to 30s (Whisper's expected input), verify exactly 3000 frames.
        let samples = vec![0.0f32; 16_000];
        let audio = pad_or_trim_audio(&samples);
        let mel = log_mel_spectrogram(&audio);
        assert_eq!(mel.n_frames, 3000);
        assert_eq!(mel.data.len(), 80 * 3000);
    }

    #[test]
    fn mel_spectrogram_matches_python_reference() {
        // 3 seconds of 440Hz sine wave at 0.5 amplitude, padded to 30s.
        // Compare against Python whisper reference values.
        let sr = WHISPER_SAMPLE_RATE as usize;
        let samples: Vec<f32> = (0..3 * sr)
            .map(|i| {
                let t = i as f32 / sr as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            })
            .collect();
        let audio = pad_or_trim_audio(&samples);
        let mel = log_mel_spectrogram(&audio);

        assert_eq!(mel.n_frames, 3000);

        // Python reference values for this exact input:
        //   min=-0.561795, max=1.438205, mean=-0.551421
        //   silence region value: -0.5618
        let min = mel.data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = mel.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean = mel.data.iter().sum::<f32>() / mel.data.len() as f32;

        eprintln!("Rust mel: min={min:.6}, max={max:.6}, mean={mean:.6}");
        eprintln!("Python:   min=-0.561795, max=1.438205, mean=-0.551421");
        eprintln!("Mel[0,:10]: {:?}", &mel.data[..10]);
        // Python: [0.983, 0.471, -0.562, -0.562, ...]
        eprintln!("Mel[0,2500]: {}", mel.data[2500]);
        // Python: -0.5618 (silence floor)

        // Allow some tolerance for float differences
        assert!((min - (-0.5618)).abs() < 0.05,
            "mel min {min} too far from Python reference -0.5618");
        assert!((max - 1.4382).abs() < 0.05,
            "mel max {max} too far from Python reference 1.4382");
    }

    #[test]
    fn mel_spectrogram_pad_truncate() {
        let samples = vec![0.0f32; 16_000];
        let mel = log_mel_spectrogram(&samples);
        let padded = mel.pad_or_truncate(3000);
        assert_eq!(padded.n_frames, 3000);
        assert_eq!(padded.data.len(), 80 * 3000);
    }

    #[test]
    fn hann_window_endpoints() {
        let w = hann_window(400);
        assert_eq!(w.len(), 400);
        // Hann window is ~0 at endpoints
        assert!(w[0].abs() < 1e-10);
        // Peak at center
        assert!((w[200] - 1.0).abs() < 0.01);
    }

    #[test]
    fn mel_filterbank_shape() {
        let filters = mel_filterbank(80, 400, 16_000);
        assert_eq!(filters.len(), 80);
        assert_eq!(filters[0].len(), 201); // n_fft/2 + 1
    }

    #[test]
    fn mel_no_nan() {
        // Sine wave at 440Hz
        let samples: Vec<f32> = (0..16_000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();
        let mel = log_mel_spectrogram(&samples);
        assert!(!mel.data.iter().any(|v| v.is_nan()));
    }
}
