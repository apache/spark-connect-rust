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

The bridge uses `arrow::pyarrow::Table` to export standard Spark result batches.
It preserves batch boundaries, schema and field metadata, and zero-column row
counts, sharing compatible Arrow value and offset buffers with PyArrow. The
caller supplies valid Rust record batches and owns its Python interpreter; the
library does not initialize Python or retain session state.

`toArrow()` collects all decoded batches into a Rust vector before the handoff.
In Arrow 58.3.0, the table helper imports each batch through its own native
C Stream reader and calls `read_next_batch()`, then builds a Python list and
calls `pyarrow.Table.from_batches()`. It does not export the whole result through
one reader or stream network results into Python. C Stream transfers buffer
descriptors and ownership, not serialized IPC bytes.

The contract covers Spark's ordinary map representation (`keys_sorted=false`).
Sorted-map flags and dictionary-encoded arrays are outside this Spark result
contract; the bridge does not add compatibility fixes for those generic Arrow
inputs.

PyArrow consumes the exported descriptors and retains their release callbacks,
so the data survives after Rust's temporary wrappers have been dropped. If an
importer raises before consuming a descriptor, Rust releases it. PyArrow and
Python code in the same interpreter are trusted, including the public aliases
used by the export path.

Field names or timezone strings containing NUL use an IPC compatibility fallback
over the already collected batches, without another RPC. Length-prefixed schema
metadata can contain NUL without this fallback. The bridge preflights
C-compatible schemas through PyArrow's standalone schema importer
before the table helper imports any batches. This lets older PyArrow versions
reject unsupported schemas before entering the C Stream path.

`Table::try_new` validates batch schema consistency, including field metadata,
before either export path. A mismatch raises `ValueError`. This is stricter than
the previous IPC export path, which wrote batches under the first batch's schema.
An empty batch vector still produces an empty table with an empty schema;
preserving schemas from schema-only IPC streams requires a separate change to
the decoding path.

Zero-copy describes compatible Arrow **value and offset buffers**, not schema
descriptors, Python scalar/row objects, pandas materialization, IPC decoding, or
the IPC fallback. Sliced arrays whose value and null offsets differ can require
copying the validity bitmap during C Data export.

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
cargo test -p spark-python-arrow --locked --features pyo3/abi3-py39
```

For an offline build with cached dependencies, prefix the command with
`CARGO_NET_OFFLINE=true CARGO_REGISTRIES_CRATES_IO_PROTOCOL=git`.

Run this package separately from the extension's test/build command: Cargo
feature unification with an extension-module consumer can disable libpython
linking required by the embedded test executable. For a virtual environment,
set `PYO3_PYTHON` to its interpreter and expose its site-packages to the embedded
interpreter with `PYTHONPATH` when necessary.

The standalone tests exercise the table export entry point: buffer identity and
lifetime, schema metadata, slices, chunks, empty shapes, schema mismatch errors,
and NUL fallback. The Python client's `test_arrow_handoff.py` exercises the public
`toArrow()` and `toPandas()` paths, including malformed IPC responses. Scalar regressions
live with the client tests. These are not substitutes for Spark's complete
Connect parity suite.
