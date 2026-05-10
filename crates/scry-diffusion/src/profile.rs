// SPDX-License-Identifier: MIT OR Apache-2.0
//! Section-level timing for the SD inference pipeline.
//!
//! When the `profile` feature is on, [`time_section`] wraps a closure with
//! `B::synchronize()` calls so the wall-clock timing actually reflects GPU
//! work (not just dispatch latency). Sections are aggregated in a
//! thread-local accumulator; [`print_summary`] sorts by total time and
//! prints percentages so the dominant cost is obvious at a glance.
//!
//! When the feature is off, every entry point compiles down to a direct
//! call of the inner closure — zero runtime overhead in production builds.
//!
//! Usage from inside a forward pass:
//! ```ignore
//! let out = profile::time_section("unet.down_blocks", || {
//!     run_down_blocks(&input)
//! });
//! ```
//!
//! Driver-side flow (see `examples/bench_sd.rs --profile`):
//! ```ignore
//! profile::reset();
//! pipeline.generate(&params)?;
//! profile::print_summary();
//! ```

#[cfg(feature = "profile")]
pub use enabled::*;
#[cfg(not(feature = "profile"))]
pub use disabled::*;

#[cfg(feature = "profile")]
mod enabled {
    use std::cell::RefCell;
    use std::time::{Duration, Instant};

    thread_local! {
        /// Per-section duration accumulator. Indexed by `&'static str` name
        /// so the hot path stores a pointer-sized key.
        static SECTIONS: RefCell<Vec<(&'static str, Duration)>> = const {
            RefCell::new(Vec::new())
        };
    }

    /// Best-effort GPU sync. CPU backend is sync-by-construction so this
    /// is a no-op there. We sync before AND after the closure so the
    /// closure's measurement window only contains its own kernels — any
    /// kernels still in flight from the prior section get attributed to
    /// the prior section, not this one.
    fn sync_gpu() {
        #[cfg(feature = "scry-gpu-cuda")]
        {
            let _ = scry_llm::backend::scry_gpu::ScryGpuBackend::synchronize();
        }
    }

    /// Time a closure, attributing its wall-clock duration to `name`.
    ///
    /// `name` must be `'static` so the aggregator can store it cheaply
    /// (no allocation in the hot path). Use string literals.
    pub fn time_section<R>(name: &'static str, f: impl FnOnce() -> R) -> R {
        sync_gpu();
        let t0 = Instant::now();
        let r = f();
        sync_gpu();
        let dt = t0.elapsed();
        SECTIONS.with(|s| s.borrow_mut().push((name, dt)));
        r
    }

    /// Clear the accumulator. Call once before the timed window starts
    /// (e.g. before `pipeline.generate(...)`).
    pub fn reset() {
        SECTIONS.with(|s| s.borrow_mut().clear());
    }

    /// Print a section breakdown — total time per name, sorted descending,
    /// with call count and percent of grand total. Designed for one-screen
    /// reading; no fancy formatting beyond fixed-width columns.
    pub fn print_summary() {
        SECTIONS.with(|s| {
            let v = s.borrow();
            if v.is_empty() {
                println!("\n=== profile summary: empty (feature on, no sections recorded) ===");
                return;
            }
            // Aggregate by name. Linear search is fine — section name set
            // is small (~20-30 entries).
            let mut totals: Vec<(&'static str, Duration, usize)> = Vec::new();
            for (n, d) in v.iter() {
                if let Some(entry) = totals.iter_mut().find(|(k, _, _)| *k == *n) {
                    entry.1 += *d;
                    entry.2 += 1;
                } else {
                    totals.push((n, *d, 1));
                }
            }
            totals.sort_by(|a, b| b.1.cmp(&a.1));
            let total: Duration = v.iter().map(|(_, d)| *d).sum();
            let total_ms = total.as_secs_f64() * 1000.0;
            println!(
                "\n=== profile summary: {} entries, {} unique sections, {:.2} ms grand total ===",
                v.len(),
                totals.len(),
                total_ms,
            );
            println!(
                "{:<32} {:>6} {:>10} {:>10} {:>7}",
                "section", "calls", "total ms", "avg ms", "% total"
            );
            for (name, dt, n) in &totals {
                let ms = dt.as_secs_f64() * 1000.0;
                #[allow(clippy::cast_precision_loss)]
                let avg = ms / *n as f64;
                let pct = if total_ms > 0.0 {
                    100.0 * ms / total_ms
                } else {
                    0.0
                };
                println!(
                    "{name:<32} {n:>6} {ms:>10.2} {avg:>10.3} {pct:>6.2}%"
                );
            }
        });
    }
}

#[cfg(not(feature = "profile"))]
mod disabled {
    /// No-op when the `profile` feature is off — the closure runs directly,
    /// nothing is recorded, no synchronization happens. Compiles down to
    /// the bare closure call after inlining.
    #[inline]
    pub fn time_section<R>(_name: &'static str, f: impl FnOnce() -> R) -> R {
        f()
    }

    /// No-op when the `profile` feature is off.
    #[inline]
    pub fn reset() {}

    /// No-op when the `profile` feature is off.
    #[inline]
    pub fn print_summary() {}
}
