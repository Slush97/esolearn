// SPDX-License-Identifier: MIT OR Apache-2.0
//! Perf-regression gate for `bench/history.jsonl`.
//!
//! Reads the history file, filters entries that match the gate's pinned
//! config, and fails if the latest entry's `step_ms_median` is more than
//! `--max-regression` percent slower than the median of the trailing
//! `--window` baseline entries (excluding the candidate itself).
//!
//! Designed to be invoked from the pre-push git hook after `bench_sd
//! --json-out bench/history.jsonl` has appended a fresh entry.
//!
//! Exit codes: `0` ok (or insufficient history to gate yet), `1`
//! regression detected, `2` malformed input or filesystem error.
//!
//! ```bash
//! cargo run -p scry-diffusion --release --example check_perf_history -- \
//!     --history bench/history.jsonl
//! ```
//!
//! ## Pinned gate config (default)
//!
//! `size=64 steps=4 scheduler=ddim bf16_matmul=true no_cudnn=false`
//!
//! Override with `--size`, `--steps`, `--scheduler`, `--bf16-matmul=false`,
//! `--no-cudnn=true` if a different bucket should be gated. The candidate
//! is always the *last* line in the file that matches the bucket.

use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug)]
struct Args {
    history: PathBuf,
    size: u32,
    steps: u32,
    scheduler: String,
    bf16_matmul: bool,
    no_cudnn: bool,
    max_regression_pct: f64,
    window: usize,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut history = PathBuf::from("bench/history.jsonl");
        let mut size: u32 = 64;
        let mut steps: u32 = 4;
        let mut scheduler = String::from("ddim");
        let mut bf16_matmul = true;
        let mut no_cudnn = false;
        let mut max_regression_pct: f64 = 15.0;
        let mut window: usize = 10;

        let mut iter = std::env::args().skip(1);
        while let Some(a) = iter.next() {
            match a.as_str() {
                "--history" => history = PathBuf::from(iter.next().ok_or("--history needs path")?),
                "--size" => size = parse_next("--size", iter.next())?,
                "--steps" => steps = parse_next("--steps", iter.next())?,
                "--scheduler" => scheduler = iter.next().ok_or("--scheduler needs value")?,
                "--bf16-matmul" => bf16_matmul = parse_bool("--bf16-matmul", iter.next())?,
                "--no-cudnn" => no_cudnn = parse_bool("--no-cudnn", iter.next())?,
                "--max-regression" => {
                    max_regression_pct = parse_next("--max-regression", iter.next())?;
                }
                "--window" => window = parse_next("--window", iter.next())?,
                "-h" | "--help" => {
                    println!("{USAGE}");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown flag: {other}\n\n{USAGE}")),
            }
        }
        if window < 2 {
            return Err("--window must be >= 2".into());
        }
        if !(0.0..=1000.0).contains(&max_regression_pct) {
            return Err("--max-regression must be in [0, 1000]".into());
        }
        Ok(Self {
            history,
            size,
            steps,
            scheduler,
            bf16_matmul,
            no_cudnn,
            max_regression_pct,
            window,
        })
    }
}

fn parse_next<T: std::str::FromStr>(flag: &str, raw: Option<String>) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    raw.ok_or_else(|| format!("{flag} needs value"))?
        .parse::<T>()
        .map_err(|e| format!("{flag}: {e}"))
}

fn parse_bool(flag: &str, raw: Option<String>) -> Result<bool, String> {
    let v = raw.ok_or_else(|| format!("{flag} needs true|false"))?;
    match v.as_str() {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        other => Err(format!("{flag}: expected true|false, got {other}")),
    }
}

const USAGE: &str = "\
Usage: check_perf_history [OPTIONS]

Reads bench/history.jsonl, filters entries that match the pinned gate
config, and fails if the latest entry is >--max-regression percent slower
than the median of the trailing --window baseline entries.

Options:
  --history PATH        Path to history.jsonl (default: bench/history.jsonl)
  --size N              Gate bucket: image side (default: 64)
  --steps N             Gate bucket: denoise steps (default: 4)
  --scheduler NAME      Gate bucket: scheduler name (default: ddim)
  --bf16-matmul BOOL    Gate bucket: bf16 matmul on/off (default: true)
  --no-cudnn BOOL       Gate bucket: --no-cudnn was set (default: false)
  --max-regression PCT  Fail threshold, percent (default: 15.0)
  --window N            Trailing entries to use as baseline (default: 10)

Exit codes:
  0  no regression (or not enough history yet to gate)
  1  regression detected
  2  malformed input or filesystem error";

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };

    let raw = match std::fs::read_to_string(&args.history) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "error: cannot read {}: {e}\n\
                 hint: run `bench_sd --json-out bench/history.jsonl` first to seed it.",
                args.history.display()
            );
            return ExitCode::from(2);
        }
    };

    let mut matching: Vec<(usize, serde_json::Value)> = Vec::new();
    for (lineno, line) in raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!(
                    "error: {}:{}: malformed JSON: {e}",
                    args.history.display(),
                    lineno + 1
                );
                return ExitCode::from(2);
            }
        };
        if matches_bucket(&v, &args) {
            matching.push((lineno + 1, v));
        }
    }

    if matching.len() < 2 {
        println!(
            "[perf-gate] only {} matching entr{} for bucket size={} steps={} scheduler={} bf16={} no_cudnn={} \
             — need >=2 to gate. Recording, not gating.",
            matching.len(),
            if matching.len() == 1 { "y" } else { "ies" },
            args.size, args.steps, args.scheduler, args.bf16_matmul, args.no_cudnn,
        );
        return ExitCode::SUCCESS;
    }

    let candidate = matching.last().expect("non-empty per check above");
    let candidate_step = match candidate
        .1
        .get("step_ms_median")
        .and_then(serde_json::Value::as_f64)
    {
        Some(v) if v > 0.0 => v,
        _ => {
            eprintln!(
                "error: candidate entry at {}:{} missing or invalid step_ms_median",
                args.history.display(),
                candidate.0,
            );
            return ExitCode::from(2);
        }
    };

    let baseline_window: Vec<f64> = matching[..matching.len() - 1]
        .iter()
        .rev()
        .take(args.window)
        .filter_map(|(_, v)| v.get("step_ms_median").and_then(serde_json::Value::as_f64))
        .filter(|x| *x > 0.0)
        .collect();

    if baseline_window.is_empty() {
        println!("[perf-gate] no usable baseline entries in window — recording, not gating.");
        return ExitCode::SUCCESS;
    }

    let baseline_median = median(&baseline_window);
    let pct = (candidate_step - baseline_median) / baseline_median * 100.0;

    println!(
        "[perf-gate] bucket size={} steps={} scheduler={} bf16={} no_cudnn={}",
        args.size, args.steps, args.scheduler, args.bf16_matmul, args.no_cudnn,
    );
    println!(
        "[perf-gate] candidate {:.3} ms/step  baseline {:.3} ms/step (median of last {})  delta {:+.2}%  threshold +{:.2}%",
        candidate_step,
        baseline_median,
        baseline_window.len(),
        pct,
        args.max_regression_pct,
    );

    if pct > args.max_regression_pct {
        eprintln!(
            "[perf-gate] FAIL: regression {pct:+.2}% exceeds threshold +{:.2}%",
            args.max_regression_pct,
        );
        ExitCode::from(1)
    } else {
        println!("[perf-gate] OK");
        ExitCode::SUCCESS
    }
}

fn matches_bucket(v: &serde_json::Value, args: &Args) -> bool {
    let g = |k: &str| v.get(k);
    g("size").and_then(serde_json::Value::as_u64) == Some(u64::from(args.size))
        && g("steps").and_then(serde_json::Value::as_u64) == Some(u64::from(args.steps))
        && g("scheduler").and_then(serde_json::Value::as_str) == Some(args.scheduler.as_str())
        && g("bf16_matmul").and_then(serde_json::Value::as_bool) == Some(args.bf16_matmul)
        && g("no_cudnn").and_then(serde_json::Value::as_bool) == Some(args.no_cudnn)
}

fn median(xs: &[f64]) -> f64 {
    let mut v: Vec<f64> = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}
