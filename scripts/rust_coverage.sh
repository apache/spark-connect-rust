#!/usr/bin/env bash
#
# Measure Rust test coverage across the whole client, including the pyspark-rs PyO3
# layer (which has no Rust tests of its own - it is exercised by the Python suite).
#
# Flow (cargo-llvm-cov, single merged report):
#   1. Export the instrumentation env (RUSTFLAGS=-Cinstrument-coverage + a profile dir).
#   2. Run the Rust unit + integration tests of the pure-Rust crates. e2e_integration
#      tests self-gate on SPARK_REMOTE, so a running Connect server raises coverage of
#      the RPC paths (client.rs, bytes_codec.rs); without one they self-skip.
#   3. Build the pyspark-rs extension *with the same instrumentation*, drop it into the
#      skin, and run the Python suite (our e2e script + the official connect suite via
#      the transport plugin) against it - that is what exercises pyspark-rs's lines.
#   4. Merge everything into one report and enforce --fail-under-lines.
#
# Usage:
#   SPARK_REMOTE=sc://localhost:15002 SPARK_SOURCE=~/spark-source \
#   RUST_PY=/path/to/python scripts/rust_coverage.sh [FAIL_UNDER]
#
# FAIL_UNDER defaults to 100. The generated proto crate is excluded (no hand-written
# logic to test). Set COV_HTML=1 to also emit an HTML report under target/llvm-cov/html.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

FAIL_UNDER="${1:-100}"
RUST_PY="${RUST_PY:-python3}"
CRATES=(-p apache-spark-connect-core -p apache-spark-connect -p pyspark-rs)
# The generated protobuf crate has no hand-written logic worth covering.
IGNORE='--ignore-filename-regex=(/target/|spark-connect-proto/|/tests/|dispatch_generated\.rs)'

echo "==> Preparing instrumented build environment"
cargo llvm-cov clean --workspace
# Export RUSTFLAGS / LLVM_PROFILE_FILE / etc. into this shell so every subsequent
# cargo build and the extension build below share one coverage profile set.
source <(cargo llvm-cov show-env --export-prefix)

echo "==> Rust unit + integration tests (pure-Rust crates)"
# e2e_integration self-gates on SPARK_REMOTE; golden/unit tests always run.
cargo test -p apache-spark-connect-core -p apache-spark-connect

echo "==> Building instrumented pyspark-rs extension into the skin"
cargo build -p pyspark-rs
# Linux CI produces lib_pyspark.so; macOS produces lib_pyspark.dylib.
if [ -f target/debug/lib_pyspark.so ]; then
  cp target/debug/lib_pyspark.so python/pyspark/_pyspark.so
else
  cp target/debug/lib_pyspark.dylib python/pyspark/_pyspark.so
fi

echo "==> Python suite against the instrumented extension (exercises pyspark-rs)"
export RUST_PYSPARK_SO="$REPO/python/pyspark/_pyspark.so"
if [ -n "${SPARK_REMOTE:-}" ]; then
  export SPARK_CONNECT_TESTING_REMOTE="${SPARK_CONNECT_TESTING_REMOTE:-$SPARK_REMOTE}"
  # Our own end-to-end script drives the drop-in wrapper (session/df/functions/streaming).
  [ -f scripts/e2e_wrapper.py ] && PYTHONPATH="$REPO/python" "$RUST_PY" scripts/e2e_wrapper.py || true
  # The official connect suite through our transport drives the low-level bindings.
  if [ -n "${SPARK_SOURCE:-}" ]; then
    SPARK_CONNECT_TESTING_REMOTE="$SPARK_CONNECT_TESTING_REMOTE" \
      "$RUST_PY" scripts/run_official_tests.py --spark "$SPARK_SOURCE" --jobs 1 || true
  fi
else
  echo "   (SPARK_REMOTE unset - skipping server-driven Python coverage)"
fi

echo "==> Merged coverage report (fail under ${FAIL_UNDER}% lines)"
[ "${COV_HTML:-0}" = "1" ] && cargo llvm-cov report "${CRATES[@]}" "$IGNORE" --html
cargo llvm-cov report "${CRATES[@]}" "$IGNORE" --summary-only --fail-under-lines "$FAIL_UNDER"
