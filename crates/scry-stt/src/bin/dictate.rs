//! scry-dictate — persistent push-to-talk STT daemon.
//!
//! Keeps Whisper model loaded in memory and listens on a Unix socket for
//! `start`/`stop` commands. Records audio via cpal, transcribes, and types
//! the result with `wtype`.
//!
//! Build:
//!   cargo build --release -p scry-stt --features dictate
//!
//! Run:
//!   scry-dictate --model ~/.config/scry/whisper-base

use std::io::{Read as _, Write};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rusqlite::Connection;

use scry_llm::backend::cpu::CpuBackend as Backend;
use scry_llm::tensor::shape::Shape;
use scry_llm::tensor::Tensor;

use scry_stt::checkpoint::load_whisper_checkpoint;
use scry_stt::decode::{greedy_decode, DecodeConfig};
use scry_stt::mel::{log_mel_spectrogram, pad_or_trim_audio, WHISPER_SAMPLE_RATE};
use scry_stt::model::WhisperModel;
use scry_stt::model::config::WhisperConfig;
use scry_stt::tokenizer::WhisperTokenizer;

const SOCKET_PATH: &str = "/tmp/scry-dictate.sock";
const WAYBAR_PATH: &str = "/.cache/scry-dictate-waybar.json";
const MIN_RECORDING_SECS: f64 = 1.0;

// ── Stats database ──────────────────────────────────────────────────────

struct StatsDb {
    conn: Connection,
}

struct TodayStats {
    count: u64,
    total_words: u64,
    avg_inference_ms: f64,
}

impl StatsDb {
    fn open() -> rusqlite::Result<Self> {
        let data_dir = data_dir();
        std::fs::create_dir_all(&data_dir).ok();
        let db_path = data_dir.join("stats.db");
        let conn = Connection::open(db_path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS transcriptions (
                id            INTEGER PRIMARY KEY,
                timestamp     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S','now','localtime')),
                duration_s    REAL    NOT NULL,
                text          TEXT    NOT NULL,
                word_count    INTEGER NOT NULL,
                token_count   INTEGER NOT NULL,
                inference_ms  REAL    NOT NULL
            );",
        )?;
        Ok(Self { conn })
    }

    fn insert(
        &self,
        duration_s: f64,
        text: &str,
        token_count: usize,
        inference_ms: f64,
    ) -> rusqlite::Result<()> {
        let word_count = text.split_whitespace().count();
        self.conn.execute(
            "INSERT INTO transcriptions (duration_s, text, word_count, token_count, inference_ms)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![duration_s, text, word_count, token_count, inference_ms],
        )?;
        Ok(())
    }

    fn today_stats(&self) -> TodayStats {
        self.conn
            .query_row(
                "SELECT COUNT(*), COALESCE(SUM(word_count),0), COALESCE(AVG(inference_ms),0)
                 FROM transcriptions
                 WHERE date(timestamp) = date('now','localtime')",
                [],
                |row| {
                    Ok(TodayStats {
                        count: row.get(0)?,
                        total_words: row.get(1)?,
                        avg_inference_ms: row.get(2)?,
                    })
                },
            )
            .unwrap_or(TodayStats {
                count: 0,
                total_words: 0,
                avg_inference_ms: 0.0,
            })
    }
}

fn data_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
        PathBuf::from(xdg).join("scry")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".local/share/scry")
    } else {
        PathBuf::from("/tmp/scry")
    }
}

// ── Transcription result ────────────────────────────────────────────────

struct TranscribeResult {
    text: String,
    token_count: usize,
    inference_ms: f64,
}

// ── Waybar ──────────────────────────────────────────────────────────────

enum WaybarState {
    Idle,
    Recording,
    Transcribing,
}

fn waybar_path() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(WAYBAR_PATH.trim_start_matches('/'))
    } else {
        PathBuf::from("/tmp/scry-dictate-waybar.json")
    }
}

fn waybar_update(state: WaybarState, stats_db: &StatsDb) {
    let (icon, tooltip, class) = match state {
        WaybarState::Idle => {
            let stats = stats_db.today_stats();
            let tip = if stats.count > 0 {
                format!(
                    "Today: {} transcriptions, {} words, avg {:.0}ms",
                    stats.count, stats.total_words, stats.avg_inference_ms
                )
            } else {
                "STT ready".to_string()
            };
            (String::new(), tip, "")
        }
        WaybarState::Recording => (
            "\u{f036c}".to_string(), // nf-md-microphone 󰍬
            "Recording...".to_string(),
            "recording",
        ),
        WaybarState::Transcribing => (
            "\u{f110a}".to_string(), // nf-md-text_recognition 󱄊
            "Transcribing...".to_string(),
            "transcribing",
        ),
    };

    let json = format!(
        r#"{{"text": "{icon}", "tooltip": "{}", "class": "{class}"}}"#,
        tooltip.replace('"', "\\\""),
    );
    let path = waybar_path();
    if let Ok(mut f) = std::fs::File::create(&path) {
        let _ = f.write_all(json.as_bytes());
    }
    // Use .status() instead of .spawn() to reap the child process and avoid
    // zombies — the daemon is long-lived so un-waited children accumulate.
    let _ = std::process::Command::new("pkill")
        .args(["-RTMIN+8", "waybar"])
        .status();
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Linearly resample audio from `src_rate` to `dst_rate`.
fn resample(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate {
        return samples.to_vec();
    }
    let ratio = f64::from(dst_rate) / f64::from(src_rate);
    let out_len = (samples.len() as f64 * ratio) as usize;
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 / ratio;
            let idx = src_pos as usize;
            let frac = (src_pos - idx as f64) as f32;
            let s0 = samples[idx.min(samples.len() - 1)];
            let s1 = samples[(idx + 1).min(samples.len() - 1)];
            s0 + frac * (s1 - s0)
        })
        .collect()
}

fn notify(summary: &str, body: &str) {
    // Use .status() instead of .spawn() to reap the child process and avoid
    // zombies — the daemon is long-lived so un-waited children accumulate.
    let _ = std::process::Command::new("notify-send")
        .args(["-a", "scry-dictate", "-t", "2000", summary, body])
        .status();
}

fn wtype(text: &str) {
    // Enter suppress submap to block modifier keys during text injection.
    // Without this, if the user is still holding Super (or presses it during
    // transcription), wtype keystrokes become Super+<key> and trigger WM shortcuts.
    let _ = std::process::Command::new("hyprctl")
        .args(["dispatch", "submap", "scry_suppress"])
        .output();

    let _ = std::process::Command::new("wtype")
        .args(["--", text])
        .status();

    // Reset submap back to global.
    let _ = std::process::Command::new("hyprctl")
        .args(["dispatch", "submap", "reset"])
        .output();
}

fn detect_config(model_arg: Option<&str>) -> PathBuf {
    if let Some(p) = model_arg {
        return PathBuf::from(p);
    }
    let config_dir = dirs_next().join("whisper-base");
    if config_dir.join("model.safetensors").exists() {
        return config_dir;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/whisper-base")
}

fn dirs_next() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        PathBuf::from(xdg).join("scry")
    } else if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".config/scry")
    } else {
        PathBuf::from("/tmp/scry")
    }
}

fn infer_config(model_dir: &Path) -> WhisperConfig {
    let dir_name = model_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    if dir_name.contains("tiny") {
        WhisperConfig::tiny()
    } else if dir_name.contains("small") {
        WhisperConfig::small()
    } else if dir_name.contains("medium") {
        WhisperConfig::medium()
    } else if dir_name.contains("large") {
        WhisperConfig::large_v3()
    } else {
        WhisperConfig::base()
    }
}

struct AudioDevice {
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
}

impl AudioDevice {
    fn new() -> Self {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .expect("No input device available");
        let config = device
            .default_input_config()
            .expect("No default input config");
        Self { device, config }
    }

    fn sample_rate(&self) -> u32 {
        self.config.sample_rate().0
    }

    fn channels(&self) -> usize {
        self.config.channels() as usize
    }
}

/// Read a single command from a Unix stream (up to first newline, max 64 bytes).
fn read_command(mut stream: &std::os::unix::net::UnixStream) -> Option<String> {
    // Set a short read timeout so we don't block forever on a broken client.
    let _ = stream.set_read_timeout(Some(Duration::from_millis(100)));
    let mut buf = [0u8; 64];
    let n = stream.read(&mut buf).ok()?;
    if n == 0 {
        return None;
    }
    let s = std::str::from_utf8(&buf[..n]).ok()?;
    Some(s.trim().to_lowercase())
}

#[derive(PartialEq, Eq)]
enum State {
    Idle,
    Recording,
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_arg = args
        .iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str);

    let model_dir = detect_config(model_arg);
    let model_path = model_dir.join("model.safetensors");
    let tokenizer_path = model_dir.join("tokenizer.json");

    if !model_path.exists() {
        eprintln!("ERROR: Model not found at {}", model_path.display());
        eprintln!("Download whisper model into {}", model_dir.display());
        std::process::exit(1);
    }

    // Load model
    eprintln!(
        "scry-dictate: loading model from {}...",
        model_dir.display()
    );
    let t0 = Instant::now();
    let config = infer_config(&model_dir);
    let model =
        load_whisper_checkpoint::<Backend>(&model_path, &config).expect("Failed to load model");
    let tokenizer =
        WhisperTokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");
    eprintln!(
        "scry-dictate: model loaded in {:.0}ms ({})",
        t0.elapsed().as_secs_f64() * 1000.0,
        config.name
    );

    // Open stats database
    let stats_db = StatsDb::open().expect("Failed to open stats database");
    eprintln!("scry-dictate: stats db ready");

    // Set up audio device
    let audio = AudioDevice::new();
    eprintln!(
        "scry-dictate: audio device: {} Hz, {} ch",
        audio.sample_rate(),
        audio.channels()
    );

    let decode_config = DecodeConfig {
        max_tokens: 224,
        ..DecodeConfig::default()
    };

    // Clean up stale socket
    let _ = std::fs::remove_file(SOCKET_PATH);

    // Set up signal handler for clean shutdown
    let running = Arc::new(AtomicBool::new(true));
    let running_sig = Arc::clone(&running);
    let _ = unsafe { libc_sigaction(running_sig) };

    // Bind socket — use a short accept timeout so we can check the shutdown flag.
    let listener = UnixListener::bind(SOCKET_PATH).expect("Failed to bind Unix socket");
    // 200ms timeout: responsive to signals, negligible latency for commands.
    listener
        .set_nonblocking(false)
        .expect("Failed to set blocking");

    eprintln!("scry-dictate: listening on {SOCKET_PATH}");
    waybar_update(WaybarState::Idle, &stats_db);

    // State
    let mut state = State::Idle;
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let mut active_stream: Option<cpal::Stream> = None;
    let mut recording_start: Option<Instant> = None;

    while running.load(Ordering::Relaxed) {
        // Accept with a timeout so we can check the shutdown flag periodically.
        // std UnixListener doesn't have set_timeout, so we use nonblocking + sleep.
        // Actually, let's just set a SO_RCVTIMEO on the listener fd.
        set_accept_timeout(&listener, Duration::from_millis(200));

        let stream = match listener.accept() {
            Ok((s, _)) => s,
            Err(ref e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                continue;
            }
            Err(ref e) if e.raw_os_error() == Some(libc::EAGAIN) => {
                continue;
            }
            Err(e) => {
                eprintln!("scry-dictate: accept error: {e}");
                continue;
            }
        };

        let Some(cmd) = read_command(&stream) else {
            continue;
        };
        // Drop the connection immediately — ctl doesn't need a response.
        drop(stream);

        // Normalize "toggle" → "start" or "stop" based on current state.
        let cmd = match cmd.as_str() {
            "toggle" => {
                let mapped = if state == State::Idle { "start" } else { "stop" };
                eprintln!("scry-dictate: toggle → {mapped}");
                mapped.to_string()
            }
            other => other.to_string(),
        };

        match cmd.as_str() {
            "start" if state == State::Idle => {
                state = State::Recording;
                buffer.lock().unwrap().clear();

                let buf_clone = Arc::clone(&buffer);
                let channels = audio.channels();
                let max_samples = audio.sample_rate() as usize * 30 * channels;
                let stream_config: cpal::StreamConfig = audio.config.clone().into();

                let s = audio
                    .device
                    .build_input_stream(
                        &stream_config,
                        move |data: &[f32], _: &cpal::InputCallbackInfo| {
                            let mut buf = buf_clone.lock().unwrap();
                            let remaining = max_samples.saturating_sub(buf.len());
                            if remaining > 0 {
                                let take = data.len().min(remaining);
                                buf.extend_from_slice(&data[..take]);
                            }
                        },
                        |err| eprintln!("scry-dictate: audio error: {err}"),
                        None,
                    )
                    .expect("Failed to build input stream");
                s.play().expect("Failed to start recording");
                active_stream = Some(s);
                recording_start = Some(Instant::now());

                notify("Recording", "Listening...");
                waybar_update(WaybarState::Recording, &stats_db);
                eprintln!("scry-dictate: recording started");
            }

            "start" if state == State::Recording => {
                // Duplicate start — ignore (key repeat).
                eprintln!("scry-dictate: ignoring duplicate start");
            }

            "stop" if state == State::Recording => {
                drop(active_stream.take());
                let duration = recording_start
                    .take()
                    .map_or(0.0, |t| t.elapsed().as_secs_f64());

                let raw_samples = buffer.lock().unwrap().clone();
                eprintln!(
                    "scry-dictate: stopped ({:.1}s, {} samples)",
                    duration,
                    raw_samples.len()
                );

                if duration < MIN_RECORDING_SECS || raw_samples.is_empty() {
                    eprintln!("scry-dictate: too short, skipping");
                    notify("Skipped", "Recording too short");
                    waybar_update(WaybarState::Idle, &stats_db);
                    state = State::Idle;
                    continue;
                }

                waybar_update(WaybarState::Transcribing, &stats_db);

                let result = transcribe(
                    &raw_samples,
                    audio.channels(),
                    audio.sample_rate(),
                    &model,
                    &tokenizer,
                    &decode_config,
                );

                let text = result.text.trim().to_string();
                if text.is_empty() {
                    eprintln!("scry-dictate: no speech detected");
                    notify("No speech", "");
                } else {
                    eprintln!("scry-dictate: \"{text}\"");
                    if let Err(e) = stats_db.insert(
                        duration,
                        &text,
                        result.token_count,
                        result.inference_ms,
                    ) {
                        eprintln!("scry-dictate: stats insert error: {e}");
                    }
                    notify("Transcribed", &text);
                    wtype(&text);
                }

                waybar_update(WaybarState::Idle, &stats_db);
                state = State::Idle;
            }

            "stop" => {
                // Stop when idle — ignore (key release without matching press).
            }

            other => {
                eprintln!(
                    "scry-dictate: unknown command: {other} (state={})",
                    if state == State::Idle {
                        "idle"
                    } else {
                        "recording"
                    }
                );
            }
        }
    }

    // Cleanup
    let _ = std::fs::remove_file(SOCKET_PATH);
    let _ = std::fs::remove_file(waybar_path());
    eprintln!("scry-dictate: shutdown");
}

// ── Transcription ───────────────────────────────────────────────────────

fn transcribe(
    raw_samples: &[f32],
    channels: usize,
    sample_rate: u32,
    model: &WhisperModel<Backend>,
    tokenizer: &WhisperTokenizer,
    decode_config: &DecodeConfig,
) -> TranscribeResult {
    let t0 = Instant::now();

    let mono: Vec<f32> = if channels > 1 {
        raw_samples
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        raw_samples.to_vec()
    };

    let audio_16k = resample(&mono, sample_rate, WHISPER_SAMPLE_RATE);

    let peak = audio_16k.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak < 0.001 {
        eprintln!("scry-dictate: audio silent (peak={peak:.6})");
        return TranscribeResult {
            text: String::new(),
            token_count: 0,
            inference_ms: 0.0,
        };
    }

    let audio_chunk = pad_or_trim_audio(&audio_16k);
    let mel = log_mel_spectrogram(&audio_chunk);
    let mel_tensor =
        Tensor::<Backend>::from_vec(mel.data, Shape::new(&[mel.n_mels, mel.n_frames]));
    let encoder_output = model.encode(&mel_tensor);
    let tokens = greedy_decode(model, &encoder_output, decode_config);
    let text = tokenizer.decode(&tokens);

    let inference_ms = t0.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "scry-dictate: transcribed in {inference_ms:.0}ms ({} tokens)",
        tokens.len()
    );

    TranscribeResult {
        text,
        token_count: tokens.len(),
        inference_ms,
    }
}

// ── Low-level helpers ───────────────────────────────────────────────────

/// Set SO_RCVTIMEO on the listener socket so `accept()` times out.
fn set_accept_timeout(listener: &UnixListener, timeout: Duration) {
    use std::os::unix::io::AsRawFd;
    let fd = listener.as_raw_fd();
    let tv = libc::timeval {
        tv_sec: timeout.as_secs() as libc::time_t,
        tv_usec: timeout.subsec_micros() as libc::suseconds_t,
    };
    unsafe {
        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_RCVTIMEO,
            &raw const tv as *const libc::c_void,
            std::mem::size_of::<libc::timeval>() as libc::socklen_t,
        );
    }
}

/// Install SIGTERM/SIGINT handler to set `running` to false.
///
/// # Safety
/// Registers a signal handler using libc. The handler only performs an atomic
/// store which is async-signal-safe.
unsafe fn libc_sigaction(running: Arc<AtomicBool>) -> std::io::Result<()> {
    static mut RUNNING_PTR: *const AtomicBool = std::ptr::null();

    unsafe {
        // Intentionally leak the Arc so the raw pointer remains valid for
        // the process lifetime. The OS reclaims all memory on exit.
        RUNNING_PTR = Arc::into_raw(running);
    }

    extern "C" fn handler(_sig: libc::c_int) {
        unsafe {
            if !RUNNING_PTR.is_null() {
                (*RUNNING_PTR).store(false, Ordering::Relaxed);
            }
        }
    }

    let mut sa: libc::sigaction = unsafe { std::mem::zeroed() };
    sa.sa_sigaction = handler as *const () as usize;
    sa.sa_flags = 0;
    if unsafe { libc::sigaction(libc::SIGTERM, &raw const sa, std::ptr::null_mut()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe { libc::sigaction(libc::SIGINT, &raw const sa, std::ptr::null_mut()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}
