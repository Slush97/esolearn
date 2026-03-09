# ─────────────────────────────────────────────────────────────────
# scry-learn benchmark container
#
# Multi-stage build:
#   1. Rust builder: compile all benchmarks in release mode
#   2. Python stage: install ML baselines
#   3. Runner: minimal image with both compiled benches + Python
#
# Build:
#   docker build -f crates/scry-learn/benches/Dockerfile -t scry-bench .
#
# Run:
#   docker run --rm scry-bench                   # local mode
#   docker run --rm --privileged scry-bench --cloud  # cloud mode (CPU pinning)
# ─────────────────────────────────────────────────────────────────

# ── Stage 1: Rust builder ───────────────────────────────────────
FROM rust:1.83-bookworm AS rust-builder

WORKDIR /build
COPY . .

# Build all benchmarks in release/bench profile
RUN cargo bench --bench fair_bench -p scry-learn --no-run 2>&1 && \
    cargo bench --bench competitor_bench -p scry-learn --no-run 2>&1

# Build the quick_vitals test
RUN cargo test --test quick_vitals -p scry-learn --release --no-run 2>&1

# ── Stage 2: Python baselines ──────────────────────────────────
FROM python:3.11-slim-bookworm AS python-stage

RUN pip install --no-cache-dir \
    numpy \
    scikit-learn \
    xgboost \
    lightgbm

# ── Stage 3: Runner ────────────────────────────────────────────
FROM debian:bookworm-slim AS runner

# Install minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    python3 \
    python3-pip \
    util-linux \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages
COPY --from=python-stage /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=python-stage /usr/local/bin/python3 /usr/local/bin/python3

# Copy compiled Rust benchmarks
COPY --from=rust-builder /build/target/release/deps/fair_bench-* /usr/local/bin/
COPY --from=rust-builder /build/target/release/deps/competitor_bench-* /usr/local/bin/
COPY --from=rust-builder /build/target/release/deps/quick_vitals-* /usr/local/bin/

# Copy benchmark scripts and fixtures
COPY crates/scry-learn/benches/fair_bench_runner.sh /bench/
COPY crates/scry-learn/benches/python/ /bench/python/
COPY crates/scry-learn/tests/fixtures/ /bench/fixtures/

WORKDIR /bench

ENV RAYON_NUM_THREADS=1
ENV RUST_BACKTRACE=1

ENTRYPOINT ["/bench/fair_bench_runner.sh"]
CMD []
