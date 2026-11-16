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

//! The runners the packer ships to the executors, driven the way Spark's Python
//! worker drives them, with a WASI module. Needs a Python with `pyspark`, `pyarrow`
//! and `wasmtime` (`$SPARK_CONNECT_PYTHON`, as for the packer). Run with
//! `cargo test --manifest-path tests/wasm-udf-wasi/Cargo.toml -- --ignored`.

use std::io::Write;
use std::process::{Command, Stdio};

use spark_connect::udf::eval_type;
use spark_connect::wasm_udf::{PythonPacker, UserDefinedFunction};
use wasm_udf_tests::udf as unknown;
use wasm_udf_wasi_tests::udf as wasi;

const PRELUDE: &str = r#"
import os
import pickle

import pyarrow as pa


def load(name):
    return pickle.loads(commands[name])[0]
"#;

/// Runs `check` in Python with `commands[name]` bound to each UDF's pickled command,
/// returning what the script wrote to standard output and error.
fn run_python(udfs: &[(&str, UserDefinedFunction)], check: &str) -> String {
    let mut script = format!("{PRELUDE}\ncommands = {{}}\n");
    for (name, f) in udfs {
        let command = f.to_payload().expect("packer should succeed").command;
        let hex: String = command.iter().map(|b| format!("{b:02x}")).collect();
        script += &format!("commands[{name:?}] = bytes.fromhex({hex:?})\n");
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
    format!("{}{stderr}", String::from_utf8_lossy(&output.stdout))
}

fn per_row(f: UserDefinedFunction) -> UserDefinedFunction {
    f.with_eval_type(eval_type::SQL_ARROW_BATCHED_UDF)
}

#[test]
#[ignore]
fn a_wasi_module_reads_the_environment_of_the_worker() {
    run_python(
        &[
            ("wasi", per_row(wasi::env_var_udf())),
            ("wasi_default", wasi::env_var_udf()),
            ("unknown", per_row(unknown::env_var_udf())),
            ("can_list", per_row(wasi::can_list_udf())),
        ],
        r#"
os.environ["WASI_PROBE"] = "from-worker"
os.environ.pop("WASI_UNSET", None)

f = load("wasi")
assert f("WASI_PROBE") == "from-worker", f("WASI_PROBE")
assert f("WASI_UNSET") is None

# The batch runner the macro picks by default reads the same environment.
g = load("wasi_default")
assert type(g).__name__ == "WasmArrowUDF"
out = g(pa.array(["WASI_PROBE", "WASI_UNSET"])).to_pylist()
assert out == ["from-worker", None], out

# The same function compiled for wasm32-unknown-unknown sees no environment at all.
assert load("unknown")("WASI_PROBE") is None

# An instance keeps the environment it started with; a new one, as a new task or
# worker makes, takes the current one.
os.environ["WASI_PROBE"] = "changed"
assert f("WASI_PROBE") == "from-worker"
assert load("wasi")("WASI_PROBE") == "changed"

# The module gets no directory.
assert load("can_list")("/") is False
assert load("can_list")(".") is False
"#,
    );
}

#[test]
#[ignore]
fn a_wasi_module_gets_neither_spark_variables_nor_ones_wasi_cannot_carry() {
    run_python(
        &[("wasi", per_row(wasi::env_var_udf()))],
        r#"
SPARK_OWNED = ("SPARK_PROBE", "PYSPARK_PROBE", "PYTHON_WORKER_FACTORY_SECRET",
               "OMP_NUM_THREADS", "PYTHON_UNIX_DOMAIN_ENABLED")
for name in SPARK_OWNED:
    os.environ[name] = "spark-owned"
os.environ["PYTHONWARNINGS"] = "ignore"
# Not valid UTF-8: WASI cannot carry it, and it must not stop the module starting.
os.environb[b"WASI_NOT_UTF8"] = b"\xff"
os.environ["WASI_OK"] = "ok"

f = load("wasi")
for name in SPARK_OWNED + ("WASI_NOT_UTF8",):
    assert f(name) is None, name
# A name that only resembles Spark's is the session's to set, so it gets through.
assert f("PYTHONWARNINGS") == "ignore"
assert f("WASI_OK") == "ok"
"#,
    );
}

#[test]
#[ignore]
fn a_wasi_module_writes_nowhere_and_its_exit_is_an_error() {
    let output = run_python(
        &[
            ("echo", per_row(wasi::echo_udf())),
            ("exit_row", per_row(wasi::exit_with_udf())),
            ("exit_batch", wasi::exit_with_udf()),
        ],
        r#"
import wasmtime

assert load("echo")("WASI_STDIO_MARKER") == "WASI_STDIO_MARKER"


def assert_exits(call, code):
    try:
        call()
    except wasmtime.ExitTrap as e:
        assert e.code == code, e.code
    else:
        raise AssertionError("expected ExitTrap")


assert_exits(lambda: load("exit_row")(3), 3)

batch = load("exit_batch")
assert type(batch).__name__ == "WasmArrowUDF"
assert_exits(lambda: batch(pa.array([3], pa.int32())), 3)
# The batch runner drops the instance an exit leaves behind.
assert batch._rt is None
"#,
    );
    assert!(!output.contains("WASI_STDIO_MARKER"), "{output}");
}

#[test]
#[ignore]
fn instantiate_initializes_a_reactor_and_names_an_import_it_cannot_provide() {
    run_python(
        &[("wasi", per_row(wasi::env_var_udf()))],
        r#"
import wasmtime

instantiate = type(load("wasi"))._ensure.__globals__["_instantiate"]
engine = wasmtime.Engine()


def instance(wat):
    store = wasmtime.Store(engine)
    module = wasmtime.Module(engine, wasmtime.wat2wasm(wat))
    return store, instantiate(engine, store, module)


ENVIRON = '(import "wasi_snapshot_preview1" "environ_sizes_get" ' \
          '(func (param i32 i32) (result i32)))'

# A reactor's `_initialize` runs before anything else is called.
store, exports = instance(f"""
(module {ENVIRON}
  (memory (export "memory") 1)
  (global $ready (export "ready") (mut i32) (i32.const 0))
  (func (export "_initialize") (global.set $ready (i32.const 1))))
""")
assert exports["ready"].value(store) == 1

# A module without one is instantiated all the same.
store, exports = instance(f"(module {ENVIRON} (memory (export \"memory\") 1))")
assert "memory" in exports

# An import that is neither WASI nor provided fails, naming it.
try:
    instance(f'(module {ENVIRON} (import "host" "lookup" (func)))')
except wasmtime.WasmtimeError as e:
    assert "lookup" in str(e), str(e)
else:
    raise AssertionError("expected WasmtimeError")
"#,
    );
}
