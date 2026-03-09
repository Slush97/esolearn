#![allow(unsafe_code, missing_docs, dead_code, clippy::cast_possible_wrap)]

//! A minimal `GlobalAlloc` wrapper that tracks peak heap usage and allocation count.
//!
//! **Usage**: include this module from a test binary and mark the static as
//! `#[global_allocator]`. Each test binary is a separate process, so this
//! won't conflict with other binaries.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

/// Global allocation tracker wrapping the system allocator.
pub struct TrackingAllocator {
    inner: System,
}

/// Current number of live heap bytes.
static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
/// High-water mark of heap bytes since last reset.
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
/// Total number of `alloc` calls since last reset.
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
/// Total number of `dealloc` calls since last reset.
static DEALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

impl Default for TrackingAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackingAllocator {
    pub const fn new() -> Self {
        Self { inner: System }
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { self.inner.alloc(layout) };
        if !ptr.is_null() {
            let size = layout.size();
            let prev = CURRENT_BYTES.fetch_add(size, SeqCst);
            let new = prev + size;
            // Update peak via compare-and-swap loop.
            let mut peak = PEAK_BYTES.load(SeqCst);
            while new > peak {
                match PEAK_BYTES.compare_exchange_weak(peak, new, SeqCst, SeqCst) {
                    Ok(_) => break,
                    Err(actual) => peak = actual,
                }
            }
            ALLOC_COUNT.fetch_add(1, SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        CURRENT_BYTES.fetch_sub(layout.size(), SeqCst);
        DEALLOC_COUNT.fetch_add(1, SeqCst);
        unsafe { self.inner.dealloc(ptr, layout) };
    }
}

/// Snapshot of allocator statistics at a point in time.
#[derive(Debug, Clone, Copy)]
pub struct AllocSnapshot {
    pub current_bytes: usize,
    pub peak_bytes: usize,
    pub alloc_count: usize,
    pub dealloc_count: usize,
}

impl AllocSnapshot {
    /// Take a snapshot of the current allocator state.
    pub fn now() -> Self {
        Self {
            current_bytes: CURRENT_BYTES.load(SeqCst),
            peak_bytes: PEAK_BYTES.load(SeqCst),
            alloc_count: ALLOC_COUNT.load(SeqCst),
            dealloc_count: DEALLOC_COUNT.load(SeqCst),
        }
    }

    /// Reset counters and peak, then return a fresh snapshot.
    ///
    /// Note: not perfectly atomic across all counters, but good enough
    /// for single-threaded benchmark sections.
    pub fn reset() -> Self {
        let current = CURRENT_BYTES.load(SeqCst);
        PEAK_BYTES.store(current, SeqCst);
        ALLOC_COUNT.store(0, SeqCst);
        DEALLOC_COUNT.store(0, SeqCst);
        Self {
            current_bytes: current,
            peak_bytes: current,
            alloc_count: 0,
            dealloc_count: 0,
        }
    }

    /// Snapshot without resetting — safe for concurrent test threads.
    pub fn snapshot() -> Self {
        Self::now()
    }

    /// Compute the delta between two snapshots (self = after, other = before).
    pub fn delta_from(self, before: Self) -> AllocDelta {
        AllocDelta {
            peak_increase: self.peak_bytes.saturating_sub(before.current_bytes),
            alloc_count: self.alloc_count.saturating_sub(before.alloc_count),
            dealloc_count: self.dealloc_count.saturating_sub(before.dealloc_count),
            net_bytes: (self.current_bytes as isize) - (before.current_bytes as isize),
        }
    }
}

/// Difference between two allocation snapshots.
#[derive(Debug, Clone, Copy)]
pub struct AllocDelta {
    /// Peak heap increase above the starting point.
    pub peak_increase: usize,
    /// Number of `alloc()` calls in the window.
    pub alloc_count: usize,
    /// Number of `dealloc()` calls in the window.
    pub dealloc_count: usize,
    /// Net change in live bytes (positive = growth).
    pub net_bytes: isize,
}

/// Format bytes into a human-readable string (B / KB / MB / GB).
pub fn format_bytes(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Format an allocation count with commas.
pub fn format_count(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
