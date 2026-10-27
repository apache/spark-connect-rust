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

`spark-python-arrow` contains the Arrow bridge and scalar conversion routines
used by `pyspark-rs`. It depends on Arrow and PyO3, not Spark Connect transport,
generated protobufs, or an async runtime. Worker-side UDF conversion is outside
this crate's scope.

## Interfaces

- `bridge`: import and export Arrow schemas, types, arrays, and record batches
  through PyArrow's C Data Interface. Table export preserves batch boundaries and
  shares decoded Arrow data buffers. Field names or timezone strings containing
  NUL use a documented IPC fallback at the table boundary; low-level C exports
  reject them. Length-prefixed schema metadata can contain NUL.
- `scalar`: borrowed scalar materialization for collected rows and Python
  extraction for client inputs. Extraction retains PyO3's coercion behavior;
  the client chooses the order of type checks and handles nested values.

`pyspark-rs` uses `bridge::record_batches_to_pyarrow` in `DataFrame.toArrow()`,
`scalar::Scalar` in row materialization, and `scalar::extract_*` in session input
conversion. These are the crate's current production consumers.

Table export raises `ValueError` if batches have inconsistent schemas, including
field metadata or dictionary ordering. This is stricter than the previous IPC
export path, which wrote batches under the first batch's schema. An empty batch
vector still produces an empty table with an empty schema; preserving schemas
from schema-only IPC streams requires a separate change to the decoding path.

## Ownership and caller responsibilities

The library does not initialize Python, acquire a process-global interpreter,
create threads, or retain session state. The consumer chooses embedding versus
extension-module linking and owns all Python callbacks and their interpreter.
Array imports accept actual PyArrow objects and use their native exporters.
Bare extension types/arrays and bare ordered dictionaries are rejected because
Arrow's bare type cannot retain their field metadata or ordering flag. Schemas
and record batches retain these properties on their fields. Extension types in
dictionary-value positions are rejected, even within schemas, because that
position has no metadata-owning Rust field.
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

The client's owned `Value` and `Row` types remain in the Python-free Rust client.
The Python binding adapts their scalar fields to `scalar::Scalar`; its row class,
schema inference, and temporal/nested compatibility rules remain caller-owned.
Cancellation, transport, and session settings likewise stay outside this crate.

## Integration

This is an unpublished workspace implementation crate (`publish = false`), not
a separately released package. `pyspark-rs` uses a path dependency. The current
package name and `0.1.0` version do not identify a crates.io release; independent
publication would require Apache package naming and release-flow integration.
Source-based consumers can pin a repository revision and validate it in their
own dependency graph. The repository's root `LICENSE` covers this crate.

The manifest keeps its package metadata self-contained so the compatibility
tests can build it outside the workspace lockfile. The library does not
enable PyO3's `extension-module` feature or select a minimum Python ABI: a Python
extension chooses those features in the host package, while an embedded worker
links its chosen Python runtime.
PyO3 must resolve to one compatible version in each consuming dependency graph.
The manifest permits PyO3 0.28, 0.29, and 0.30 so consumers on these lines can use
the same source. The OSS Python extension selects 0.30.1, which requires Rust
1.85 or newer; `pyspark-rs` declares that minimum independently of the workspace's
existing 1.82 default. The isolated converter compatibility CI job tests each minor
version independently of the workspace lockfile. A version is not considered
verified until its corresponding compatibility job passes.

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
empty shapes, schema mismatch errors, and scalar extraction/materialization. The
Python client's `test_arrow_handoff.py` exercises the public `toArrow()` and
`toPandas()` paths. These are not substitutes for Spark's complete Connect
parity suite.
