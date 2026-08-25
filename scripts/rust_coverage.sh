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

# Pin one target for the whole run when cross-building the extension: the Rust tests,
# the extension, and the merged report must all use the SAME architecture, or the
# report cannot map the (Python-driven) extension's profiles onto the test objects.
# In CI this is unset (native == the Python arch); locally set COV_EXT_TARGET to the
# Python's arch (e.g. x86_64-apple-darwin for x86_64 conda on an arm64 mac).
if [ -n "${COV_EXT_TARGET:-}" ]; then
  export CARGO_BUILD_TARGET="$COV_EXT_TARGET"
fi

echo "==> Preparing instrumented build environment"
cargo llvm-cov clean --workspace
# Export RUSTFLAGS / LLVM_PROFILE_FILE / etc. into this shell so every subsequent
# cargo build and the extension build below share one coverage profile set.
source <(cargo llvm-cov show-env --export-prefix)

echo "==> Rust unit + integration tests (pure-Rust crates)"
# e2e_integration self-gates on SPARK_REMOTE; golden/unit tests always run.
# --include-ignored runs the server-gated integration_test.rs cases too (they are
# #[ignore] so a serverless `cargo test` skips them) to cover the live-RPC paths.
# --lib --tests (no doctests): rustdoc cannot be profiled with -Cinstrument-coverage
# on stable, so doctests error under coverage; the lines their examples show are
# covered by real lib/integration tests instead.
# --test-threads=1: the e2e_integration tests share one Connect session/server; running
# them serially avoids the multiple-threads-on-the-shared-runtime races that otherwise
# return empty results under an instrumented build.
cargo test -p apache-spark-connect-core -p apache-spark-connect --lib --tests \
    -- --include-ignored --test-threads=1

echo "==> Building instrumented pyspark-rs extension into the skin"
# The extension must match RUST_PY's architecture or Python cannot load it (and then
# no coverage is recorded for pyspark-rs). In CI the native build already matches
# (x86_64 Linux + x86_64 Python). Locally, set COV_EXT_TARGET to cross-build, e.g.
# COV_EXT_TARGET=x86_64-apple-darwin for an x86_64 conda Python on an arm64 mac.
if [ -n "${COV_EXT_TARGET:-}" ]; then
  cargo build -p pyspark-rs --target "$COV_EXT_TARGET"
  ext_dir="target/${COV_EXT_TARGET}/debug"
else
  cargo build -p pyspark-rs
  ext_dir="target/debug"
fi
# Linux produces lib_pyspark.so; macOS produces lib_pyspark.dylib.
if [ -f "${ext_dir}/lib_pyspark.so" ]; then
  cp "${ext_dir}/lib_pyspark.so" python/pyspark/_pyspark.so
else
  cp "${ext_dir}/lib_pyspark.dylib" python/pyspark/_pyspark.so
fi

echo "==> Python suite against the instrumented extension (exercises pyspark-rs)"
export RUST_PYSPARK_SO="$REPO/python/pyspark/_pyspark.so"
if [ -n "${SPARK_REMOTE:-}" ]; then
  export SPARK_CONNECT_TESTING_REMOTE="${SPARK_CONNECT_TESTING_REMOTE:-$SPARK_REMOTE}"
  # Our own end-to-end script drives the drop-in wrapper (session/df/functions/streaming).
  # NOT `|| true`: e2e_wrapper exits 0 on per-op gaps but non-zero if it cannot run at
  # all (import/connect failure). Swallowing that would silently yield a report with
  # artificially low coverage - the coverage run must fail loudly instead.
  if [ -f scripts/e2e_wrapper.py ]; then
    PYTHONPATH="$REPO/python" "$RUST_PY" scripts/e2e_wrapper.py
  fi
  # The official connect suite through our transport drives the low-level bindings.
  # It exits 1 on *parity* failures (which must NOT abort a coverage measurement) but
  # >=2 when the harness itself could not run (no server, no test files); distinguish
  # the two so a broken run fails rather than reporting silently-low coverage.
  if [ -n "${SPARK_SOURCE:-}" ]; then
    off_rc=0
    SPARK_CONNECT_TESTING_REMOTE="$SPARK_CONNECT_TESTING_REMOTE" \
      "$RUST_PY" scripts/run_official_tests.py --spark "$SPARK_SOURCE" --jobs 1 || off_rc=$?
    if [ "$off_rc" -ge 2 ]; then
      echo "!! official suite could not run (exit $off_rc); coverage would be understated" >&2
      exit "$off_rc"
    fi
  fi
else
  echo "   (SPARK_REMOTE unset - skipping server-driven Python coverage)"
fi

echo "==> Merged coverage report (fail under ${FAIL_UNDER}% lines)"
[ "${COV_HTML:-0}" = "1" ] && cargo llvm-cov report "${CRATES[@]}" "$IGNORE" --html
cargo llvm-cov report "${CRATES[@]}" "$IGNORE" --summary-only --fail-under-lines "$FAIL_UNDER"
