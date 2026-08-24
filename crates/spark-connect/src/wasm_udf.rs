//! Rust user-defined functions executed on Spark via WebAssembly.
//!
//! # Overview
//!
//! Spark Connect has no native "run this Rust code" UDF type. What it *does*
//! have is the [`PythonUDF`](crate::udf) path: a cloudpickled `(callable,
//! return_type)` tuple shipped inside a `CommonInlineUserDefinedFunction` and
//! executed row-by-row on the Spark executors by a Python worker.
//!
//! This module leverages that path to run **Rust** UDFs, distributed on the
//! cluster, without touching the Spark server:
//!
//! 1. The user compiles their Rust function to a WebAssembly module
//!    (`cargo build --target wasm32-unknown-unknown`, or `wasm32-wasip1`).
//! 2. [`udf`] embeds the `.wasm` bytes and the function signature and invokes
//!    the bundled Python packer (`python -m pyspark_wasm_udf.pack`), which uses
//!    `cloudpickle` to serialize a `WasmScalarUDF` runner **by value** into the
//!    `command`. Because it is serialized by value, the executors do **not**
//!    need the `pyspark_wasm_udf` package -- only the `wasmtime` package.
//! 3. On each executor the runner instantiates the module with `wasmtime` and
//!    invokes the exported entrypoint once per input row.
//!
//! # API
//!
//! The API mirrors `pyspark.sql.functions.udf`: [`udf`] is the factory and
//! returns a [`UserDefinedFunction`] you call on columns.
//!
//! ```no_run
//! use spark_connect::functions::col;
//! use spark_connect::types::DataType;
//! use spark_connect::wasm_udf::{udf, WasmValType};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let wasm = std::fs::read("add_one.wasm")?;
//! let add_one = udf("add_one", wasm, "run",
//!     vec![WasmValType::I64], WasmValType::I64, DataType::Long);
//! let c = add_one.call(vec![col("id")])?;   // -> Column
//! # Ok(()) }
//! ```
//!
//! # Requirements
//!
//! Building the command requires, on the client machine only:
//!   * a Python interpreter with `cloudpickle` and `pyspark` installed, and
//!   * the `pyspark_wasm_udf` package importable (ship the repo's `python/`
//!     directory, or point [`PythonPacker::pythonpath`] at it).
//!
//! The Spark executors need only the `wasmtime` Python package.
//!
//! # Scope
//!
//! This is a prototype. Only numeric scalar signatures are supported: the WASM
//! value types [`WasmValType`] map to the Python scalars `wasmtime` exchanges
//! directly. String/binary arguments require a memory-passing ABI and are left
//! as follow-up work. Supported Spark output types are the atomic types plus
//! `StringType`; other types return [`SparkError`].
//!
//! # Follow-up: Arrow-based ABI
//!
//! To widen type support beyond numeric scalars (strings, binary, nested
//! types), a natural next step is to adopt the [`arrow-udf`] WASM ABI, where
//! the module exchanges Arrow `RecordBatch`es (via the C Data Interface / IPC)
//! instead of individual scalars. On this server-side path the executor-side
//! runner would hand the arrow-optimized batch to the module and read an Arrow
//! array back, letting Arrow handle the type mapping for every Spark type.
//!
//! [`arrow-udf`]: https://crates.io/crates/arrow-udf

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use base64::Engine as _;

use crate::column::Column;
use crate::expression::Expression;
use crate::types::DataType;
use crate::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};
use spark_connect_core::error::{Result, SparkError};

/// A WebAssembly value type, describing one slot of the exported function's
/// signature as seen across the host/WASM boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WasmValType {
    I32,
    I64,
    F32,
    F64,
}

impl WasmValType {
    /// The tag the Python runner uses to coerce values (`"i32"`, `"i64"`, ...).
    fn tag(self) -> &'static str {
        match self {
            WasmValType::I32 => "i32",
            WasmValType::I64 => "i64",
            WasmValType::F32 => "f32",
            WasmValType::F64 => "f64",
        }
    }
}

/// How to invoke the Python packer that turns a WASM UDF into the cloudpickled
/// `command` bytes.
#[derive(Debug, Clone)]
pub struct PythonPacker {
    /// The Python executable to run (default: `$SPARK_CONNECT_PYTHON`, else
    /// `$PYSPARK_PYTHON`, else `python3`).
    pub python_exe: String,
    /// Directories prepended to `PYTHONPATH` so `pyspark_wasm_udf` is
    /// importable (default: `$SPARK_CONNECT_WASM_PACKER_PATH` if set).
    pub pythonpath: Vec<PathBuf>,
}

impl Default for PythonPacker {
    fn default() -> Self {
        let python_exe = std::env::var("SPARK_CONNECT_PYTHON")
            .or_else(|_| std::env::var("PYSPARK_PYTHON"))
            .unwrap_or_else(|_| "python3".to_string());
        let pythonpath = std::env::var("SPARK_CONNECT_WASM_PACKER_PATH")
            .ok()
            .map(|p| vec![PathBuf::from(p)])
            .unwrap_or_default();
        PythonPacker {
            python_exe,
            pythonpath,
        }
    }
}

impl PythonPacker {
    /// Run `python -m pyspark_wasm_udf.pack`, feeding `spec_json` on stdin and
    /// returning the raw `command` bytes from stdout.
    fn run(&self, spec_json: &str) -> Result<Vec<u8>> {
        let mut cmd = Command::new(&self.python_exe);
        cmd.arg("-m")
            .arg("pyspark_wasm_udf.pack")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if !self.pythonpath.is_empty() {
            let mut joined = self
                .pythonpath
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join(pathsep());
            if let Ok(existing) = std::env::var("PYTHONPATH") {
                joined.push_str(pathsep());
                joined.push_str(&existing);
            }
            cmd.env("PYTHONPATH", joined);
        }

        let mut child = cmd.spawn().map_err(|e| {
            SparkError::connect_msg(format!(
                "failed to spawn WASM UDF packer '{}': {e}",
                self.python_exe
            ))
        })?;

        child
            .stdin
            .take()
            .expect("stdin was piped")
            .write_all(spec_json.as_bytes())
            .map_err(|e| SparkError::connect_msg(format!("failed to write UDF spec: {e}")))?;

        let output = child
            .wait_with_output()
            .map_err(|e| SparkError::connect_msg(format!("WASM UDF packer failed: {e}")))?;

        if !output.status.success() {
            return Err(SparkError::connect_msg(format!(
                "WASM UDF packer exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            )));
        }
        if output.stdout.is_empty() {
            return Err(SparkError::connect_msg(
                "WASM UDF packer produced an empty command".to_string(),
            ));
        }
        Ok(output.stdout)
    }
}

#[cfg(windows)]
fn pathsep() -> &'static str {
    ";"
}
#[cfg(not(windows))]
fn pathsep() -> &'static str {
    ":"
}

/// Create a Rust/WASM user-defined function, mirroring
/// `pyspark.sql.functions.udf`.
///
/// Where PySpark's `udf(f, returnType)` takes a Python callable, the WASM
/// variant takes the compiled module plus its exported signature:
///
/// * `name` — the UDF name reported to Spark.
/// * `wasm_module` — the raw bytes of the compiled `.wasm` module.
/// * `entrypoint` — the exported function to invoke per row.
/// * `arg_types` — the WASM value type of each argument, in order.
/// * `ret_type` — the WASM value type returned by `entrypoint`.
/// * `return_type` — the Spark [`DataType`] of the produced column.
///
/// The returned [`UserDefinedFunction`] is callable on columns via
/// [`UserDefinedFunction::call`].
pub fn udf(
    name: impl Into<String>,
    wasm_module: impl Into<Vec<u8>>,
    entrypoint: impl Into<String>,
    arg_types: Vec<WasmValType>,
    ret_type: WasmValType,
    return_type: DataType,
) -> UserDefinedFunction {
    UserDefinedFunction {
        name: name.into(),
        wasm_module: wasm_module.into(),
        entrypoint: entrypoint.into(),
        arg_types,
        ret_type,
        return_type,
        deterministic: true,
        // Arrow-optimized Python UDF: func is still called once per row
        // (see pyspark worker `wrap_arrow_batch_udf`), but IO is Arrow.
        eval_type: eval_type::SQL_ARROW_BATCHED_UDF,
        python_ver: default_python_ver(),
        packer: PythonPacker::default(),
    }
}

/// A Rust/WASM user-defined function, mirroring
/// `pyspark.sql.udf.UserDefinedFunction`.
///
/// Construct one with [`udf`], then apply it to columns with
/// [`UserDefinedFunction::call`], or register it on a session by name via
/// [`SparkSession::register_function`](crate::session::SparkSession::register_function)
/// after [`UserDefinedFunction::to_expression`].
#[derive(Debug, Clone)]
pub struct UserDefinedFunction {
    name: String,
    wasm_module: Vec<u8>,
    entrypoint: String,
    arg_types: Vec<WasmValType>,
    ret_type: WasmValType,
    return_type: DataType,
    deterministic: bool,
    eval_type: i32,
    python_ver: String,
    packer: PythonPacker,
}

impl UserDefinedFunction {
    /// Mark the UDF non-deterministic, mirroring
    /// `pyspark.sql.udf.UserDefinedFunction.asNondeterministic`.
    pub fn as_nondeterministic(mut self) -> Self {
        self.deterministic = false;
        self
    }

    /// Override the Python evaluation type. Defaults to
    /// [`SQL_ARROW_BATCHED_UDF`](eval_type::SQL_ARROW_BATCHED_UDF); the
    /// row-at-a-time [`SQL_BATCHED_UDF`](eval_type::SQL_BATCHED_UDF) is also
    /// compatible with the runner's per-row calling convention.
    pub fn with_eval_type(mut self, value: i32) -> Self {
        self.eval_type = value;
        self
    }

    /// Override the Python version tag sent to the server (`"major.minor"`).
    pub fn with_python_ver(mut self, value: impl Into<String>) -> Self {
        self.python_ver = value.into();
        self
    }

    /// Override how the Python packer is invoked (executable / `PYTHONPATH`).
    pub fn with_packer(mut self, packer: PythonPacker) -> Self {
        self.packer = packer;
        self
    }

    /// Build the JSON spec handed to the Python packer. Validates that the
    /// Spark output type is supported before spawning any process.
    fn build_spec(&self) -> Result<String> {
        let output_type = output_type_token(&self.return_type)?;
        let spec = serde_json::json!({
            "wasm_b64": base64::engine::general_purpose::STANDARD.encode(&self.wasm_module),
            "entrypoint": self.entrypoint,
            "arg_types": self.arg_types.iter().map(|t| t.tag()).collect::<Vec<_>>(),
            "ret_type": self.ret_type.tag(),
            "output_type": output_type,
        });
        Ok(spec.to_string())
    }

    /// Build the cloudpickle-compatible `command` bytes by invoking the packer:
    /// a pickled `(WasmScalarUDF(...), return_type)` tuple.
    fn build_command(&self) -> Result<Vec<u8>> {
        let spec = self.build_spec()?;
        self.packer.run(&spec)
    }

    /// Build the [`PythonUDFPayload`] carrying the pickled command.
    pub fn to_payload(&self) -> Result<PythonUDFPayload> {
        Ok(PythonUDFPayload::new(
            self.return_type.clone(),
            self.eval_type,
            self.build_command()?,
            self.python_ver.clone(),
        ))
    }

    /// Build the [`CommonInlineUserDefinedFunctionExpression`] for the given
    /// argument columns.
    pub fn to_expression(
        &self,
        args: Vec<Column>,
    ) -> Result<CommonInlineUserDefinedFunctionExpression> {
        Ok(CommonInlineUserDefinedFunctionExpression::new(
            self.name.clone(),
            self.deterministic,
            args.iter().map(|c| c.expression().clone()).collect(),
            self.to_payload()?,
        ))
    }

    /// Apply the UDF to `args`, producing a [`Column`] — mirroring calling a
    /// PySpark `UserDefinedFunction` on columns (`my_udf(col("x"))`).
    pub fn call(&self, args: Vec<Column>) -> Result<Column> {
        let expr = self.to_expression(args)?;
        Ok(Column::new(Expression::CommonInlineUserDefinedFunction(
            Box::new(expr),
        )))
    }
}

/// Default Python version tag, matching how pyspark reports it
/// (`"%d.%d" % sys.version_info[:2]`). We cannot see the executors' interpreter,
/// so we send a reasonable default; callers can override with
/// [`UserDefinedFunction::with_python_ver`].
fn default_python_ver() -> String {
    "3.11".to_string()
}

/// Map a Spark [`DataType`] to the token the Python packer understands.
///
/// Only atomic types plus `StringType` are supported in this prototype; other
/// types return an error before any process is spawned.
fn output_type_token(dt: &DataType) -> Result<&'static str> {
    let token = match dt {
        DataType::Null => "null",
        DataType::Boolean => "boolean",
        DataType::Byte => "byte",
        DataType::Short => "short",
        DataType::Integer => "integer",
        DataType::Long => "long",
        DataType::Float => "float",
        DataType::Double => "double",
        DataType::Binary => "binary",
        DataType::Date => "date",
        DataType::Timestamp => "timestamp",
        DataType::TimestampNtz => "timestamp_ntz",
        DataType::String { .. } => "string",
        other => {
            return Err(SparkError::connect_msg(format!(
                "WASM UDF output type {other:?} is not supported yet; \
                 use an atomic type or StringType"
            )));
        }
    };
    Ok(token)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> UserDefinedFunction {
        udf(
            "add_one",
            vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00],
            "run",
            vec![WasmValType::I64],
            WasmValType::I64,
            DataType::Long,
        )
    }

    #[test]
    fn defaults_to_arrow_batched_udf() {
        let f = sample();
        assert_eq!(f.eval_type, eval_type::SQL_ARROW_BATCHED_UDF);
        assert!(f.deterministic);
    }

    #[test]
    fn as_nondeterministic_flips_flag() {
        assert!(!sample().as_nondeterministic().deterministic);
    }

    #[test]
    fn build_spec_encodes_signature_and_wasm() {
        let f = sample();
        let spec: serde_json::Value = serde_json::from_str(&f.build_spec().unwrap()).unwrap();
        assert_eq!(spec["entrypoint"], "run");
        assert_eq!(spec["arg_types"], serde_json::json!(["i64"]));
        assert_eq!(spec["ret_type"], "i64");
        assert_eq!(spec["output_type"], "long");
        // Base64 of the sample module bytes.
        assert_eq!(spec["wasm_b64"], "AGFzbQEA");
    }

    #[test]
    fn output_type_tokens() {
        assert_eq!(output_type_token(&DataType::Long).unwrap(), "long");
        assert_eq!(output_type_token(&DataType::Double).unwrap(), "double");
        assert_eq!(
            output_type_token(&DataType::String {
                collation: "UTF8_BINARY".to_string()
            })
            .unwrap(),
            "string"
        );
    }

    #[test]
    fn unsupported_output_type_errors() {
        let err = output_type_token(&DataType::Array {
            element_type: Box::new(DataType::Integer),
            contains_null: true,
        });
        assert!(err.is_err());
    }

    /// End-to-end packer test. Requires a Python interpreter with `cloudpickle`
    /// + `pyspark` and the `pyspark_wasm_udf` package importable (set
    /// `SPARK_CONNECT_WASM_PACKER_PATH` to the repo's `python/` dir). Run with
    /// `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn packer_produces_nonempty_command() {
        let f = sample();
        let cmd = f.build_command().expect("packer should succeed");
        assert!(!cmd.is_empty());
        // cloudpickle output starts with the PROTO opcode (0x80).
        assert_eq!(cmd[0], 0x80);
    }
}
