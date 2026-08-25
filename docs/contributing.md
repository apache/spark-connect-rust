# Contributing

Contributions are welcome. This is an Apache Software Foundation project, so a few
conventions apply.

## Issues and discussion

Bugs and feature requests are tracked in **ASF JIRA** under the
[SPARK](https://issues.apache.org/jira/browse/SPARK) project - GitHub Issues are
disabled. Reference the JIRA key in your pull request title, e.g.
`[SPARK-XXXXX] Short summary`.

## Building and testing

Requires a Rust toolchain and `protobuf-compiler`.

```bash
cargo build                 # build the library crates
cargo test                  # unit + golden-proto tests
cargo build -p examples     # the Rust-native examples
```

Builder correctness is covered by **golden-proto tests**: plans, expressions, and
all SQL functions are asserted byte-for-byte against captured reference protos.
Transport and Arrow paths are exercised by the official Apache Spark Connect test
suite in CI - see the CI notes under `dev/design/`.

The `wasm-udf` example crates need the `wasm32` target and are built by manifest
path (they're excluded from the workspace):

```bash
rustup target add wasm32-unknown-unknown
cargo build --manifest-path examples/wasm-udf-inline/Cargo.toml
```

## Style

```bash
cargo fmt --all             # format
cargo clippy --workspace    # lint
```

Match the surrounding code, keep changes focused, and add or update golden tests
when you change plan or expression building.

## Pull requests

Open PRs against
[`apache/spark-connect-rust`](https://github.com/apache/spark-connect-rust).
Keep the PR description tied to its JIRA, and make sure `cargo test`, `rustfmt
--check`, and `clippy` pass.
