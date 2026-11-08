// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! The runners the packer ships to the executors, driven with pyarrow arrays the
//! way Spark's Python worker drives them. Needs a Python with `pyspark`, `pyarrow`
//! and `wasmtime` (`$SPARK_CONNECT_PYTHON`, as for the packer). Run with
//! `cargo test --manifest-path tests/wasm-udf/Cargo.toml -- --ignored`.

use std::io::Write;
use std::process::{Command, Stdio};

use spark_connect::udf::eval_type;
use spark_connect::wasm_udf::{self, AbiType, PythonPacker, UserDefinedFunction};
use wasm_udf_tests::udf;

const PRELUDE: &str = r#"
import copy
import pickle
import sys

import pyarrow as pa
import wasmtime


def raises(exc, needle, f, *args):
    try:
        f(*args)
    except exc as e:
        assert needle in str(e), (needle, str(e))
    else:
        raise AssertionError(f"expected {exc.__name__}")
"#;

/// Runs `check` in Python with `udfs[name]` bound to each UDF's unpickled runner.
fn run_python(udfs: &[(&str, UserDefinedFunction)], check: &str) {
    let mut script = format!("{PRELUDE}\nudfs = {{}}\n");
    for (name, f) in udfs {
        let command = f.to_payload().expect("packer should succeed").command;
        let hex: String = command.iter().map(|b| format!("{b:02x}")).collect();
        script += &format!("udfs[{name:?}] = pickle.loads(bytes.fromhex({hex:?}))[0]\n");
    }
    script += check;

    let mut child = Command::new(PythonPacker::default().python_exe)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn python");
    child
        .stdin
        .take()
        .unwrap()
        .write_all(script.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(!stderr.contains("Exception ignored"), "{stderr}");
}

fn per_row(f: UserDefinedFunction) -> UserDefinedFunction {
    f.with_eval_type(eval_type::SQL_ARROW_BATCHED_UDF)
}

#[test]
#[ignore]
fn only_batch_capable_signatures_get_the_batch_runner() {
    run_python(
        &[
            ("scalars", udf::sub_i64_udf()),
            ("array", udf::sum_udf()),
            ("forced_per_row", per_row(udf::sub_i64_udf())),
        ],
        r#"
assert type(udfs["scalars"]).__name__ == "WasmArrowUDF"
assert type(udfs["array"]).__name__ == "WasmScalarUDF"
assert type(udfs["forced_per_row"]).__name__ == "WasmScalarUDF"
"#,
    );
}

#[test]
#[ignore]
fn batch_runner_returns_what_the_per_row_runner_returns() {
    let fixtures = [
        ("sub_i32", udf::sub_i32_udf()),
        ("sub_i64", udf::sub_i64_udf()),
        ("sub_f32", udf::sub_f32_udf()),
        ("sub_f64", udf::sub_f64_udf()),
        ("and_not", udf::and_not_udf()),
        ("concat", udf::concat_udf()),
        ("concat_bytes", udf::concat_bytes_udf()),
        ("label", udf::label_udf()),
    ];
    let row_names: Vec<String> = fixtures.iter().map(|(n, _)| format!("{n}_row")).collect();
    let mut udfs = Vec::new();
    for ((name, f), row_name) in fixtures.into_iter().zip(&row_names) {
        udfs.push((row_name.as_str(), per_row(f.clone())));
        udfs.push((name, f));
    }
    run_python(
        &udfs,
        r##"
cases = [
    ("sub_i32", pa.int32(), [pa.array([5, -7, 2**31 - 1, 0], pa.int32()), pa.array([2, None, -1, 0], pa.int32())], [3, None, -2**31, 0]),
    ("sub_i64", pa.int64(), [pa.array([5, -7, 0], pa.int64()), pa.array([2, None, 0], pa.int64())], [3, None, 0]),
    ("sub_f32", pa.float32(), [pa.array([1.5, 2.0], pa.float32()), pa.array([0.25, None], pa.float32())], [1.25, None]),
    ("sub_f64", pa.float64(), [pa.array([1.5, -2.0]), pa.array([None, 0.5])], [None, -2.5]),
    ("and_not", pa.bool_(),
     [pa.array([True, True, False, True, False, True, True, False, True]),
      pa.array([False, True, False, None, True, False, False, False, None])],
     [True, False, False, None, False, True, True, False, None]),
    ("concat", pa.string(), [pa.array(["a", "", "hé"]), pa.array(["b", None, "!"])], ["ab", None, "hé!"]),
    ("concat", pa.string(), [pa.array(["", ""]), pa.array(["", ""])], ["", ""]),
    ("concat_bytes", pa.binary(), [pa.array([b"\x00", b"", b"xy"]), pa.array([b"\xff", None, b""])], [b"\x00\xff", None, b"xy"]),
    ("label", pa.string(), [pa.array(["a", None]), pa.array([1, -2], pa.int64())], ["a#1", "#-2"]),
    ("sub_i64", pa.int64(), [pa.array([], pa.int64()), pa.array([], pa.int64())], []),
    ("concat", pa.string(), [pa.array([], pa.string()), pa.array([], pa.string())], []),
]
for name, out_type, args, expected in cases:
    got = udfs[name](*args)
    assert got.type == out_type, (name, got.type)
    assert got.to_pylist() == expected, (name, got.to_pylist(), expected)
    rows = zip(*(a.to_pylist() for a in args))
    assert [udfs[name + "_row"](*row) for row in rows] == expected, name
assert len(sys._spark_connect_wasm_udf_modules) == 1
"##,
    );
}

#[test]
#[ignore]
fn batch_runner_accepts_sliced_chunked_large_and_wider_inputs() {
    run_python(
        &[
            ("sub_i32", udf::sub_i32_udf()),
            ("sub_i64", udf::sub_i64_udf()),
            ("and_not", udf::and_not_udf()),
            ("concat", udf::concat_udf()),
            ("concat_bytes", udf::concat_bytes_udf()),
        ],
        r#"
x = pa.array([100, 1, 2, 3, 4, 5, 6, 7, 8, 9], pa.int64()).slice(3)
y = pa.array([None, 0, 0, 1, None, 1, 1, 1, 1, None], pa.int64()).slice(3)
assert udfs["sub_i64"](x, y).to_pylist() == [2, None, 4, 5, 6, 7, None]
x = pa.array([False] * 5 + [True, True, False, True, True, False]).slice(5)
y = pa.array([True] * 5 + [False, None, True, True, False, False]).slice(5)
assert udfs["and_not"](x, y).to_pylist() == [True, None, False, False, True, False]
x, y = pa.array(["skip", "a", "bb"]).slice(1), pa.array(["z", "1", None]).slice(1)
assert udfs["concat"](x, y).to_pylist() == ["a1", None]

for n in (2, 50_000, 3):
    x = ["ab" * (i % 17) for i in range(n)]
    y = [None if i % 7 == 3 else str(i) for i in range(n)]
    expected = [None if b is None else a + b for a, b in zip(x, y)]
    assert udfs["concat"](pa.array(x), pa.array(y)).to_pylist() == expected, n

x = pa.chunked_array([[5], [7, 9]], pa.int64())
y = pa.chunked_array([[1, None], [2]], pa.int64())
assert udfs["sub_i64"](x, y).to_pylist() == [4, None, 7]

x, y = pa.array(["a", "b"], pa.large_string()), pa.array(["1", None], pa.large_string())
assert udfs["concat"](x, y).to_pylist() == ["a1", None]
x, y = pa.array([b"a"], pa.large_binary()), pa.array([b"1"], pa.large_binary())
assert udfs["concat_bytes"](x, y).to_pylist() == [b"a1"]

assert udfs["sub_i32"](pa.array([5, 7], pa.int64()), pa.array([2, 3], pa.int64())).to_pylist() == [3, 4]
raises(pa.ArrowInvalid, "not in range", udfs["sub_i32"], pa.array([2**40]), pa.array([1]))
"#,
    );
}

#[test]
#[ignore]
fn batch_runner_errors_name_the_cause_and_the_next_batch_succeeds() {
    const MODULE: &[u8] = include_bytes!(env!("WASM_UDFS_MODULE"));
    let one_column = wasm_udf::udf(
        "sub_i64",
        MODULE,
        "spark_udf_sub_i64",
        vec![AbiType::I64],
        AbiType::Nullable(Box::new(AbiType::I64)),
    )
    .with_batch_entrypoint("spark_batch_udf_sub_i64");
    run_python(
        &[
            ("sub_i64", udf::sub_i64_udf()),
            ("one_column", one_column),
            (
                "missing",
                udf::sub_i64_udf().with_batch_entrypoint("spark_batch_udf_missing"),
            ),
            ("keep_first", udf::keep_first_udf()),
            ("non_negative", udf::non_negative_udf()),
        ],
        r#"
f = udfs["sub_i64"]
ok = [pa.array([3], pa.int64()), pa.array([1], pa.int64())]

raises(ValueError, "argument `x` is null in row 1", f, pa.array([1, None]), pa.array([1, 1]))
assert f(*ok).to_pylist() == [2]
raises(ValueError, "argument `#1` is null in row 0", udfs["keep_first"], ok[0], pa.array([None], pa.int64()))
raises(ValueError, "got arguments of different lengths: [3, 2]", f, pa.array([1, 2, 3]), pa.array([1, 2]))
raises(ValueError, "got arguments of different lengths: [0, 1]", f, pa.array([], pa.int64()), ok[1])
assert f(*ok).to_pylist() == [2]
raises(ValueError, "does not export 'spark_batch_udf_missing'", udfs["missing"], *ok)
raises(ValueError, "expected 2 argument(s), got 1", f, ok[0])
assert f(*ok).to_pylist() == [2]
raises(ValueError, "the batch has 1 columns, but the function takes 2", udfs["one_column"], ok[0])

big = pa.Array.from_buffers(pa.int64(), 2**27, [None, pa.allocate_buffer(2**30)])
raises(ValueError, "maxRecordsPerBatch", f, big, big)
assert f(*ok).to_pylist() == [2]

saved, sys.modules["wasmtime"] = sys.modules["wasmtime"], None
try:
    raises(ImportError, "The 'wasmtime' package is required", copy.copy(f), *ok)
finally:
    sys.modules["wasmtime"] = saved

g = udfs["non_negative"]
raises(wasmtime.Trap, "unreachable", g, pa.array([1, -1], pa.int64()))
assert g(pa.array([4], pa.int64())).to_pylist() == [4]
"#,
    );
}
