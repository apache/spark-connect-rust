<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

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

### UDF converter contract

The result policy accepts the stock, pure PySpark converters for **nullable
top-level UDF results**, not arbitrary callbacks or converters for non-nullable
fields. It can bypass those converters for `None` and exact strings/bytes whose
conversion is an identity. The Spark type check does not prove a callback has
these semantics. Callers must supply the matching PySpark converter; no probe is
performed because invoking a callback during policy selection can change error
timing and cannot prove its behavior for every value.

Run the UDF for every row before calling `ResultColumn::finish`. Mutable results
are retained until then, and conversion errors are deferred so a later UDF error
still takes precedence. Once a result has been converted, a capacity fallback
retains that converted value instead of invoking its converter again.

## Ownership and caller responsibilities

The library does not initialize Python, acquire a process-global interpreter,
create threads, or retain session state. The consumer chooses embedding versus
extension-module linking and owns all Python callbacks and their interpreter.
Array imports accept actual PyArrow objects and use their native exporters.
Bare extension types/arrays and bare ordered dictionaries are rejected because
Arrow's bare type cannot retain their field metadata or ordering flag. Schemas
and record batches retain these properties on their fields. Extension types in
dictionary-value positions are rejected, even within schemas, because that
position has no metadata-owning Rust field. UDF extension results select the
caller's fallback.
As with the Arrow C Data Interface itself, producers are responsible for valid
buffer contents and all offsets. Imports perform limited structural checks, not
full Arrow validation or a scan of every interior offset. A real PyArrow class
does not by itself prove complete buffer validity. CPython's strict UTF-8 decoder
checks the selected string bytes, not the Arrow offsets used to select them.

Imports call native `pyarrow.lib` class descriptors to avoid dispatching an
arbitrary object's exporter. Exports and Python fallback operations use the
public `pyarrow` aliases, allowing normal Python instrumentation, including the
ownership/failure-injection tests. This is not isolation from arbitrary Python
code running in the same interpreter.

Zero-copy describes compatible Arrow **value and offset buffers**, not schema
descriptors, Python scalar/row objects, pandas materialization, IPC decoding, or
fallback conversion. Sliced arrays whose value and null offsets differ can
require copying the validity bitmap during C Data export. Import/export release
callbacks retain shared buffers after either language's temporary wrapper has
been dropped.

UDF result builders allocate new Arrow buffers. A mixed native/fallback result
uses `pyarrow.concat_arrays` to join the prefix and suffix into the single array
the worker expects; that concatenation copies the combined column. Only the
handoff of a completed, compatible native array shares its buffers.

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
The manifest permits PyO3 0.28 and 0.29 so consumers on either line can use the
same source. The workspace lockfile selects 0.28.3 and does not validate 0.29.
The isolated converter compatibility CI job tests both versions independently;
0.29 support must not be considered verified until that job passes.

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

With official `pyspark-client==4.2.0` also installed, set
`SPARK_PYTHON_ARROW_TEST_PYSPARK=1` to enable comparisons against its nullable
String, Binary, and Variant result converters. The compatibility CI matrix sets
this flag; the default local command skips that optional comparison.

The standalone tests cover buffer identity and lifetime, schema/type round trips,
empty shapes, native scalar behavior, and UDF fallback/error ordering. The
Python client's `test_arrow_handoff.py` exercises the public `toArrow()` and
`toPandas()` paths. These are not substitutes for downstream worker differential
tests or Spark's complete Connect parity suite.
