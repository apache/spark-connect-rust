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

# Arrow table export to PyArrow

`spark-python-arrow` exports decoded Rust Arrow record batches to PyArrow for
`pyspark-rs`. It has one public entry point,
`bridge::record_batches_to_pyarrow`, used by `DataFrame.toArrow()`. It does not
depend on Spark Connect transport, generated protobufs, or an async runtime.

Python-to-Rust Arrow imports, standalone array/type exports, and worker-side UDF
conversion are outside this change. Scalar input conversion and row
materialization remain in `pyspark-rs` until they have another concrete consumer.

## Export contract

The Arrow C Stream Interface path preserves batch boundaries, schema and field
metadata, and zero-column row counts. It shares compatible Arrow value and offset
buffers with PyArrow. The caller supplies valid Rust record batches and owns its
Python interpreter; the library does not initialize Python or retain session state.

`toArrow()` first collects all decoded batches into a Rust `Vec<RecordBatch>`.
The bridge exports a native Arrow reader over those batches, imports it once as
a PyArrow `RecordBatchReader`, and calls `read_all()`. PyArrow pulls each batch
through native callbacks, without a Python import call per batch or a temporary
Python list of batches. The callbacks do not call Python. C Stream is an
in-process ownership interface, not an IPC byte stream; it does not make result
collection network-streaming or bounded-memory.

PyArrow consumes the exported stream and retains each batch's release callback,
so the table's buffers outlive the reader and Rust's temporary wrappers. If an
importer raises before consuming the stream, Rust releases it and its remaining
batches. PyArrow and Python code in the same interpreter are trusted, including
the public aliases used by the export path.

Field names or timezone strings containing NUL use an IPC compatibility fallback
over the already collected batches, without another RPC. Length-prefixed schema
metadata can contain NUL without this fallback. The stream's `get_schema` callback
uses a schema descriptor that retains sorted-map and dictionary-order flags when
present; the other callbacks forward to Arrow's native stream implementation.

Table export raises `ValueError` if batches have inconsistent schemas, including
field metadata or dictionary ordering. This is stricter than the previous IPC
export path, which wrote batches under the first batch's schema. An empty batch
vector still produces an empty table with an empty schema; preserving schemas
from schema-only IPC streams requires a separate change to the decoding path.

Zero-copy describes compatible Arrow **value and offset buffers**, not schema
descriptors, Python scalar/row objects, pandas materialization, IPC decoding, or
the IPC fallback. The C Stream callbacks export batches using C Data descriptors;
sliced arrays whose value and null offsets differ can require copying the validity
bitmap during that export.

## Integration

This is an unpublished workspace implementation crate (`publish = false`), not
a separately released package. `pyspark-rs` uses a path dependency. The current
package name and `0.1.0` version do not identify a crates.io release; independent
publication would require Apache package naming and release-flow integration.
Source-based consumers can pin a repository revision and validate it in their
own dependency graph. The repository's root `LICENSE` covers this crate.

The library does not enable PyO3's `extension-module` feature or select a minimum Python ABI;
`pyspark-rs` selects these features for the Python extension.

PyO3 must resolve to one compatible version in each consuming dependency graph.
The manifest uses the same PyO3 0.28 line as the Python extension, with 0.28.3
pinned in the workspace lockfile. This change does not upgrade PyO3.
This crate declares Rust 1.85, matching Arrow 58.3.0's declared minimum.
The existing workspace default of 1.82 is unchanged; it is not a claim that
these Python/Arrow packages build on 1.82. The minimum compiler has not been
independently tested locally.

The Python Arrow export CI job tests this crate with the workspace lockfile.
These tests run separately from the extension build and are not repeated in coverage.

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

The standalone tests exercise the table export entry point: buffer identity and
lifetime, schema metadata, slices, chunks, empty shapes, schema mismatch errors,
and NUL fallback. The Python client's `test_arrow_handoff.py` exercises the public
`toArrow()` and `toPandas()` paths, including importer failures. Scalar regressions
live with the client tests. These are not substitutes for Spark's complete
Connect parity suite.
