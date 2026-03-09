#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# fair_bench_runner.sh — Reproducible benchmark runner for scry-learn
#
# Collects hardware info, enforces single-thread, runs the benchmark,
# and saves structured results.
#
# Usage (local):   ./fair_bench_runner.sh
# Usage (cloud):   ./fair_bench_runner.sh --cloud
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRATE_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$SCRIPT_DIR/results"
CLOUD_MODE="${1:-}"
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
mkdir -p "$RESULTS_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  FAIR BENCHMARK RUNNER — scry-learn vs Rust ML ecosystem"
echo "  $(date -u)"
echo "═══════════════════════════════════════════════════════════════"

# ── 1. Hardware & Environment ────────────────────────────────────────
INFO_FILE="$RESULTS_DIR/env_${TIMESTAMP}.txt"
{
    echo "=== ENVIRONMENT ==="
    echo "Date:     $(date -u)"
    echo "Hostname: $(hostname)"
    echo "OS:       $(uname -srm)"
    echo ""
    echo "=== CPU ==="
    lscpu 2>/dev/null || echo "(lscpu not available)"
    echo ""
    echo "=== MEMORY ==="
    free -h 2>/dev/null || echo "(free not available)"
    echo ""
    echo "=== RUST ==="
    rustc --version
    cargo --version
    echo ""
    echo "=== RAYON ==="
    echo "RAYON_NUM_THREADS=1 (enforced)"
    echo ""
} | tee "$INFO_FILE"

# ── 2. Set environment ──────────────────────────────────────────────
export RAYON_NUM_THREADS=1

# ── 3. Cloud mode extras ────────────────────────────────────────────
BENCH_FLAGS="--bench fair_bench -p scry-learn"
if [[ "$CLOUD_MODE" == "--cloud" ]]; then
    echo "┌─ CLOUD MODE: extended scaling + process isolation"
    BENCH_FLAGS="$BENCH_FLAGS --features extended-bench"
    # CPU pinning (use cores 0-1 to avoid NUMA effects)
    if command -v taskset &>/dev/null; then
        TASKSET_PREFIX="taskset -c 0"
        echo "│  CPU pinned to core 0 via taskset"
    else
        TASKSET_PREFIX=""
        echo "│  taskset not available — no CPU pinning"
    fi
    # Priority
    if command -v nice &>/dev/null; then
        NICE_PREFIX="nice -n -10"
        echo "│  nice -n -10 for scheduling priority"
    else
        NICE_PREFIX=""
    fi
    echo "└─"
else
    TASKSET_PREFIX=""
    NICE_PREFIX=""
    echo "Local mode — standard settings"
fi

# ── 4. Compile ──────────────────────────────────────────────────────
echo ""
echo "─── Compiling (release) ─────────────────────────────"
cargo bench $BENCH_FLAGS --no-run 2>&1 | tail -5

# ── 5. Run ──────────────────────────────────────────────────────────
echo ""
echo "─── Running benchmarks ────────────────────────────────"
echo ""
BENCH_LOG="$RESULTS_DIR/bench_${TIMESTAMP}.log"

$NICE_PREFIX $TASKSET_PREFIX cargo bench $BENCH_FLAGS 2>&1 | tee "$BENCH_LOG"

# ── 6. Summary ──────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ Benchmark complete"
echo "  Env info: $INFO_FILE"
echo "  Results:  $BENCH_LOG"
echo "  Criterion HTML: target/criterion/"
echo "═══════════════════════════════════════════════════════════════"
