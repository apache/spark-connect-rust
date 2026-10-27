<!-- SPDX-License-Identifier: Apache-2.0 -->

# Shared Python and Arrow conversion

`spark-python-arrow` is a Rust library for Python-facing Spark clients and workers.
It depends on Arrow and PyO3, not Spark Connect transport, generated protobufs,
an async runtime, or a worker executable. Consumers use the same versioned source;
they should not maintain copied converter modules.

## Interfaces

- `bridge`: import and export Arrow schemas, types, arrays, and record batches
  through PyArrow's C Data Interface. Table export preserves batch boundaries and
  shares decoded Arrow data buffers. Field names or timezone strings containing
  NUL use a documented IPC fallback at the table boundary; low-level C exports
  reject them. Length-prefixed schema metadata can contain NUL.
- `scalar`: borrowed scalar materialization and Python extraction. Exact-built-in
  extraction is separate from coercive extraction so callers cannot accidentally
  apply UDF coercion rules to client inputs.
- `udf`: the `SQL_ARROW_BATCHED_UDF` conversion policy. `input_spec` and
  `result_spec` select supported native paths. `InputColumns` and `ResultColumn`
  preserve the Python worker's conversion order, mutable-result handling, and
  ArrowInvalid retry behavior. Python classes and converter callbacks come from
  the caller; the library does not import a particular PySpark distribution.
  `result_spec` resolves the caller's Variant class lazily, only after accepting
  a native Variant conversion.

Native UDF conversion covers signed integers, floats, booleans, strings, binary,
and tagged Variant structs, plus dates on input. Nested, decimal, timestamp, and
date-result columns retain the caller's Python fallback. Unsupported conversion
is `Ok(Err(reason))`; an error inspecting a supported conversion is `Err(PyErr)`.
Callers must not turn execution errors into unsupported-type fallback.

## Ownership and caller responsibilities

The library does not initialize Python, acquire a process-global interpreter,
create threads, or retain session state. The consumer chooses embedding versus
extension-module linking and owns all Python callbacks and their interpreter.
Array imports accept actual PyArrow objects and use their native exporters.
Bare extension types/arrays are rejected because Arrow's bare type cannot retain
their extension metadata. Schemas and record batches retain it on their fields;
UDF extension results select the caller's fallback.
As with the Arrow C Data Interface itself, producers are responsible for valid
buffer contents. Imports check structural invariants without scanning payloads.

Zero-copy describes compatible Arrow **data buffers**, not schema descriptors,
Python scalar/row objects, pandas materialization, IPC decoding, or fallback
conversion. Import/export release callbacks retain the buffers after either
language's temporary wrapper has been dropped.

The client's owned `Value` and `Row` types remain in the Python-free Rust client.
The Python binding adapts their scalar fields to `scalar::Scalar`; its row class,
schema inference, and temporal/nested compatibility rules remain caller-owned.
Worker execution, cancellation, transport, and session settings likewise stay
outside this crate.

## Integration

Cargo consumers use the crate as a normal Rust dependency. The library does not
enable PyO3's `extension-module` feature or select a minimum Python ABI: a Python
extension chooses those features in the host package, while an embedded worker
links its chosen Python runtime.
PyO3 must resolve to one compatible version in each consuming dependency graph.

Bazel consumers should pin the OSS revision or release using their existing
third-party Rust dependency mechanism and depend on this library target. There
must be no internal copy of `src/bridge.rs`, `src/scalar.rs`, or `src/udf`.
An unpublished local checkout may be used for development, but is not a release
pin and must not become a checked-in machine-specific path.

## Tests

With PyArrow installed in the Python environment used by PyO3:

```sh
CARGO_NET_OFFLINE=true CARGO_REGISTRIES_CRATES_IO_PROTOCOL=git \
  cargo test -p spark-python-arrow --locked --features pyo3/abi3-py39
```

Run this package separately from the extension's test/build command: Cargo
feature unification with an extension-module consumer can disable libpython
linking required by the embedded test executable. For a virtual environment,
set `PYO3_PYTHON` to its interpreter and expose its site-packages to the embedded
interpreter with `PYTHONPATH` when necessary.

The standalone tests cover buffer identity and lifetime, schema/type round trips,
empty shapes, native scalar behavior, and UDF fallback/error ordering. The
Python client's `test_arrow_handoff.py` exercises the public `toArrow()` and
`toPandas()` paths. These are not substitutes for downstream worker differential
tests or Spark's complete Connect parity suite.
