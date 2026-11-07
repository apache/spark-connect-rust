//! Rust user-defined functions executed on Spark via WebAssembly.
//!
//! # Overview
//!
//! Spark Connect has no native "run this Rust code" UDF type. What it *does*
//! have is the [`PythonUDF`](crate::udf) path: a cloudpickled `(callable,
//! return_type)` tuple shipped inside a `CommonInlineUserDefinedFunction` and
//! executed on the Spark executors by a Python worker.
//!
//! This module leverages that path to run **Rust** UDFs, distributed on the
//! cluster, without touching the Spark server:
//!
//! 1. The user compiles their Rust function to a WebAssembly module.
//! 2. [`udf`] embeds the `.wasm` bytes and the signature and runs a packer script
//!    **embedded in this crate** as `python -c` (so it executes as `__main__`),
//!    which uses `cloudpickle` to serialize a runner **by value**
//!    into the `command`. Running it as `__main__` is what makes cloudpickle embed
//!    the runner by value, so there is no separate `pyspark_wasm_udf` pip package to
//!    install on the client and the executors need only the `wasmtime` package.
//! 3. On each executor the runner instantiates the module with `wasmtime` and
//!    invokes the exported entrypoint once per Arrow batch (with a batch
//!    entrypoint, see [`UserDefinedFunction::with_batch_entrypoint`]) or once per
//!    input row.
//!
//! # ABI
//!
//! For a per-row entrypoint, arguments and the result cross the WASM boundary
//! as a little-endian, length-prefixed byte buffer (see [`AbiType`]): the runner
//! encodes the row's arguments into linear memory (via the module's exported
//! `spark_udf_alloc`), calls the entrypoint `fn(ptr, len) -> packed_ptr_len`, and
//! decodes the result. This supports scalars, `bool`, strings, binary, arrays,
//! and nullability — not just numbers.
//!
//! A batch entrypoint, which the [`spark_wasm_udf`](https://docs.rs/apache-spark-connect-macros)
//! macro generates for the signatures its docs list, instead takes the Arrow
//! buffers of a whole batch and returns the result's Arrow buffers.
//!
//! # API
//!
//! Prefer the [`spark_wasm_udf`](https://docs.rs/apache-spark-connect-macros)
//! macro, which writes the [`AbiType`]s and the constructor for you. This
//! lower-level [`udf`] factory (mirroring `pyspark.sql.functions.udf`) is for
//! loading a prebuilt module by hand:
//!
//! ```no_run
//! use spark_connect::functions::col;
//! use spark_connect::wasm_udf::{udf, AbiType};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let wasm = std::fs::read("shout.wasm")?;
//! // fn shout(s: String) -> String
//! let shout = udf("shout", wasm, "shout", vec![AbiType::Str], AbiType::Str);
//! let c = shout.call(vec![col("name")])?;   // -> Column
//! # Ok(()) }
//! ```
//!
//! # Requirements
//!
//! Building the command requires, on the client machine only, a Python
//! interpreter with `pyspark` importable (for its vendored `cloudpickle` and the
//! pure-Python type parser). The packer script itself is embedded in this crate, so
//! nothing extra needs installing — and the Spark executors need only the
//! `wasmtime` Python package.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use base64::Engine as _;

use crate::column::Column;
use crate::expression::ExpressionNode;
use crate::types::DataType;
use crate::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};
use spark_connect_core::error::{Result, SparkError};

/// The type of one WASM UDF argument or its result, describing how the value is
/// encoded across the WASM boundary and which Spark SQL type it maps to.
///
/// Built for you by the `#[spark_wasm_udf]` macro from the Rust signature:
///
/// | Rust                | `AbiType`                     | Spark SQL type |
/// |---------------------|-------------------------------|----------------|
/// | `i32`               | `I32`                         | `IntegerType`  |
/// | `i64`               | `I64`                         | `LongType`     |
/// | `f32`               | `F32`                         | `FloatType`    |
/// | `f64`               | `F64`                         | `DoubleType`   |
/// | `bool`              | `Bool`                        | `BooleanType`  |
/// | `String` / `&str`   | `Str`                         | `StringType`   |
/// | `Vec<u8>`           | `Binary`                      | `BinaryType`   |
/// | `Vec<T>`            | `Array(T)`                    | `ArrayType`    |
/// | `Option<T>`         | `Nullable(T)`                 | (nullable `T`) |
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbiType {
    I32,
    I64,
    F32,
    F64,
    Bool,
    /// UTF-8 string (`String` / `&str`).
    Str,
    /// Byte string (`Vec<u8>`).
    Binary,
    /// Homogeneous list (`Vec<T>`).
    Array(Box<AbiType>),
    /// Nullable value (`Option<T>`).
    Nullable(Box<AbiType>),
}

impl AbiType {
    /// The descriptor string the Python runner uses to drive its codec, e.g.
    /// `"i64"`, `"string"`, `"array:i64"`, `"option:string"`.
    pub fn descriptor(&self) -> String {
        match self {
            AbiType::I32 => "i32".to_string(),
            AbiType::I64 => "i64".to_string(),
            AbiType::F32 => "f32".to_string(),
            AbiType::F64 => "f64".to_string(),
            AbiType::Bool => "bool".to_string(),
            AbiType::Str => "string".to_string(),
            AbiType::Binary => "binary".to_string(),
            AbiType::Array(inner) => format!("array:{}", inner.descriptor()),
            AbiType::Nullable(inner) => format!("option:{}", inner.descriptor()),
        }
    }

    /// The Spark [`DataType`] this maps to (used for the proto `output_type`).
    pub fn to_data_type(&self) -> DataType {
        match self {
            AbiType::I32 => DataType::Integer,
            AbiType::I64 => DataType::Long,
            AbiType::F32 => DataType::Float,
            AbiType::F64 => DataType::Double,
            AbiType::Bool => DataType::Boolean,
            AbiType::Str => DataType::String {
                collation: "UTF8_BINARY".to_string(),
            },
            AbiType::Binary => DataType::Binary,
            AbiType::Array(inner) => DataType::Array {
                element_type: Box::new(inner.to_data_type()),
                contains_null: matches!(**inner, AbiType::Nullable(_)),
            },
            // Nullability is tracked at the column/field level; the element type
            // itself is what matters here.
            AbiType::Nullable(inner) => inner.to_data_type(),
        }
    }

    /// Spark's canonical type JSON (`DataType.jsonValue()` form), which the
    /// packer parses with `_parse_datatype_json_value` — so any type maps
    /// generically without a hand-maintained token table.
    fn to_spark_json(&self) -> serde_json::Value {
        use serde_json::json;
        match self {
            AbiType::I32 => json!("integer"),
            AbiType::I64 => json!("long"),
            AbiType::F32 => json!("float"),
            AbiType::F64 => json!("double"),
            AbiType::Bool => json!("boolean"),
            AbiType::Str => json!("string"),
            AbiType::Binary => json!("binary"),
            AbiType::Array(inner) => json!({
                "type": "array",
                "elementType": inner.to_spark_json(),
                "containsNull": matches!(**inner, AbiType::Nullable(_)),
            }),
            AbiType::Nullable(inner) => inner.to_spark_json(),
        }
    }

    fn batch_supported(&self) -> bool {
        match self {
            AbiType::I32
            | AbiType::I64
            | AbiType::F32
            | AbiType::F64
            | AbiType::Bool
            | AbiType::Str
            | AbiType::Binary => true,
            AbiType::Array(_) => false,
            AbiType::Nullable(inner) => {
                !matches!(**inner, AbiType::Array(_) | AbiType::Nullable(_))
            }
        }
    }
}

/// The packer script, embedded at compile time. Run as `python -c <this>` (so it
/// executes as `__main__`), which makes cloudpickle serialize the runner by value
/// automatically — no `pyspark_wasm_udf` pip package is needed on
/// the client (only a Python interpreter with `pyspark`) or on the executors (only
/// `wasmtime`).
const PACKER_SRC: &str = include_str!("wasm_packer.py");

/// How to invoke the Python packer that turns a WASM UDF into the cloudpickled
/// `command` bytes.
#[derive(Debug, Clone)]
pub struct PythonPacker {
    /// The Python executable to run (default: `$SPARK_CONNECT_PYTHON`, else
    /// `$PYSPARK_PYTHON`, else `python3`).
    pub python_exe: String,
    /// Extra directories prepended to `PYTHONPATH` (e.g. to locate `pyspark` if it
    /// is not already importable by `python_exe`); default `$SPARK_CONNECT_WASM_PACKER_PATH`
    /// if set. The packer itself is embedded in the crate, so this is only for deps.
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
    /// Run the embedded packer via `python -c <PACKER_SRC>`, feeding `spec_json` on
    /// stdin and returning the raw `command` bytes from stdout. Running it as `-c`
    /// (i.e. `__main__`) is what lets cloudpickle embed the runner by value.
    fn run(&self, spec_json: &str) -> Result<Vec<u8>> {
        let mut cmd = Command::new(&self.python_exe);
        cmd.arg("-c")
            .arg(PACKER_SRC)
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
/// * `name` — the UDF name reported to Spark.
/// * `wasm_module` — the raw bytes of the compiled `.wasm` module.
/// * `entrypoint` — the exported function to invoke per row.
/// * `arg_types` — the [`AbiType`] of each argument, in order.
/// * `ret_type` — the [`AbiType`] returned by `entrypoint`; also determines the
///   Spark output type.
///
/// The returned [`UserDefinedFunction`] is callable on columns via
/// [`UserDefinedFunction::call`], and calls `entrypoint` once per row unless
/// given a batch entrypoint with [`UserDefinedFunction::with_batch_entrypoint`].
pub fn udf(
    name: impl Into<String>,
    wasm_module: impl Into<Vec<u8>>,
    entrypoint: impl Into<String>,
    arg_types: Vec<AbiType>,
    ret_type: AbiType,
) -> UserDefinedFunction {
    UserDefinedFunction {
        name: name.into(),
        wasm_module: wasm_module.into(),
        entrypoint: entrypoint.into(),
        arg_types,
        ret_type,
        deterministic: true,
        // Arrow-optimized Python UDF: func is still called once per row
        // (see pyspark worker `wrap_arrow_batch_udf`), but IO is Arrow.
        eval_type: eval_type::SQL_ARROW_BATCHED_UDF,
        batch_entrypoint: None,
        python_ver: default_python_ver(),
        packer: PythonPacker::default(),
    }
}

/// A Rust/WASM user-defined function, mirroring
/// `pyspark.sql.udf.UserDefinedFunction`.
#[derive(Debug, Clone)]
pub struct UserDefinedFunction {
    name: String,
    wasm_module: Vec<u8>,
    entrypoint: String,
    arg_types: Vec<AbiType>,
    ret_type: AbiType,
    deterministic: bool,
    eval_type: i32,
    batch_entrypoint: Option<String>,
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

    /// Override the Python evaluation type (default
    /// [`SQL_ARROW_BATCHED_UDF`](eval_type::SQL_ARROW_BATCHED_UDF), or
    /// [`SQL_SCALAR_ARROW_UDF`](eval_type::SQL_SCALAR_ARROW_UDF) after
    /// [`with_batch_entrypoint`](Self::with_batch_entrypoint)). Only
    /// `SQL_SCALAR_ARROW_UDF` uses the batch entrypoint, so setting
    /// `SQL_ARROW_BATCHED_UDF` calls the UDF once per row.
    pub fn with_eval_type(mut self, value: i32) -> Self {
        self.eval_type = value;
        self
    }

    /// Call `entrypoint`, a batch export generated by `#[spark_wasm_udf]`, once per
    /// Arrow batch, by sending the UDF as a scalar Arrow UDF
    /// ([`SQL_SCALAR_ARROW_UDF`](eval_type::SQL_SCALAR_ARROW_UDF)).
    pub fn with_batch_entrypoint(mut self, entrypoint: impl Into<String>) -> Self {
        self.batch_entrypoint = Some(entrypoint.into());
        self.eval_type = eval_type::SQL_SCALAR_ARROW_UDF;
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

    /// The Spark output [`DataType`], derived from the return [`AbiType`].
    pub fn output_type(&self) -> DataType {
        self.ret_type.to_data_type()
    }

    /// Build the JSON spec handed to the Python packer.
    fn build_spec(&self) -> Result<String> {
        let batch_entrypoint = if self.eval_type == eval_type::SQL_SCALAR_ARROW_UDF {
            let Some(entrypoint) = &self.batch_entrypoint else {
                return Err(SparkError::value_msg(format!(
                    "WASM UDF '{}' has eval type SQL_SCALAR_ARROW_UDF, which needs a batch \
                     entrypoint; set one with with_batch_entrypoint",
                    self.name
                )));
            };
            if self.arg_types.is_empty() {
                return Err(SparkError::value_msg(format!(
                    "WASM UDF '{}' has no arguments, so it cannot be called once per batch",
                    self.name
                )));
            }
            let mut types = self.arg_types.iter().chain([&self.ret_type]);
            if let Some(ty) = types.find(|t| !t.batch_supported()) {
                return Err(SparkError::value_msg(format!(
                    "WASM UDF '{}' has type '{}', so it cannot be called once per batch; \
                     call it per row with with_eval_type(SQL_ARROW_BATCHED_UDF)",
                    self.name,
                    ty.descriptor()
                )));
            }
            Some(entrypoint)
        } else {
            None
        };
        let spec = serde_json::json!({
            "wasm_b64": base64::engine::general_purpose::STANDARD.encode(&self.wasm_module),
            "entrypoint": self.entrypoint,
            "batch_entrypoint": batch_entrypoint,
            "arg_types": self.arg_types.iter().map(|t| t.descriptor()).collect::<Vec<_>>(),
            "ret_type": self.ret_type.descriptor(),
            "output_type": self.ret_type.to_spark_json(),
        });
        Ok(spec.to_string())
    }

    /// Build the cloudpickle-compatible `command` bytes by invoking the packer.
    fn build_command(&self) -> Result<Vec<u8>> {
        self.packer.run(&self.build_spec()?)
    }

    /// Build the [`PythonUDFPayload`] carrying the pickled command.
    pub fn to_payload(&self) -> Result<PythonUDFPayload> {
        Ok(PythonUDFPayload::new(
            self.output_type(),
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
    pub fn call<C: Into<Column>>(&self, args: impl IntoIterator<Item = C>) -> Result<Column> {
        let args: Vec<Column> = args.into_iter().map(Into::into).collect();
        let expr = self.to_expression(args)?;
        Ok(Column::new(
            ExpressionNode::CommonInlineUserDefinedFunction(Box::new(expr)).into(),
        ))
    }

    /// Build the registration expression for this UDF under `name` (no bound
    /// arguments — arguments are supplied at each SQL/DataFrame call site).
    fn registration_expression(
        &self,
        name: &str,
    ) -> Result<CommonInlineUserDefinedFunctionExpression> {
        Ok(CommonInlineUserDefinedFunctionExpression::new(
            name.to_string(),
            self.deterministic,
            vec![],
            self.to_payload()?,
        ))
    }
}

/// UDF registration accessor, returned by [`crate::session::SparkSession::udf`] and
/// mirroring `pyspark.sql.SparkSession.udf`.
pub struct UdfRegistration<'a> {
    session: &'a crate::session::SparkSession,
}

impl<'a> UdfRegistration<'a> {
    /// Register `udf` under `name` so it can be called from SQL
    /// (`spark.sql("SELECT name(col) ...")`) and the DataFrame API.
    ///
    /// Mirrors `spark.udf.register(name, udf)`: the cloudpickled WASM runner is shipped
    /// by value inside a `RegisterFunction` command, so the Spark executors only need
    /// the `wasmtime` Python package (not `pyspark_wasm_udf`).
    pub fn register(&self, name: &str, udf: &UserDefinedFunction) -> Result<()> {
        self.session
            .register_function(udf.registration_expression(name)?)
    }
}

impl crate::session::SparkSession {
    /// UDF registration accessor. Mirrors `pyspark.sql.SparkSession.udf`, so a WASM
    /// UDF is registered with `spark.udf().register("name", &udf)`.
    pub fn udf(&self) -> UdfRegistration<'_> {
        UdfRegistration { session: self }
    }
}

/// Default Python version tag, matching how pyspark reports it
/// (`"%d.%d" % sys.version_info[:2]`). Callers can override with
/// [`UserDefinedFunction::with_python_ver`].
fn default_python_ver() -> String {
    // Try to detect the Python version from the interpreter used by PythonPacker
    let python_exe = std::env::var("SPARK_CONNECT_PYTHON")
        .or_else(|_| std::env::var("PYSPARK_PYTHON"))
        .unwrap_or_else(|_| "python3".to_string());

    match detect_python_version(&python_exe) {
        Some(version) => version,
        None => {
            // Fallback to a reasonable default if detection fails
            "3.11".to_string()
        }
    }
}

/// Detect the Python version by running the interpreter.
/// Returns MAJOR.MINOR format (e.g., "3.11"), or None if detection fails.
fn detect_python_version(python_exe: &str) -> Option<String> {
    use std::process::Command;

    let output = Command::new(python_exe)
        .arg("-c")
        .arg("import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    String::from_utf8(output.stdout)
        .ok()
        .and_then(|s| parse_python_version_output(&s))
}

/// Parse Python version output (MAJOR.MINOR format).
/// Handles whitespace and newlines.
fn parse_python_version_output(output: &str) -> Option<String> {
    let trimmed = output.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> UserDefinedFunction {
        // fn add_one(x: i64) -> i64
        udf(
            "add_one",
            vec![0x00, 0x61, 0x73, 0x6d, 0x01, 0x00],
            "add_one",
            vec![AbiType::I64],
            AbiType::I64,
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
    fn abitype_descriptors() {
        assert_eq!(AbiType::I64.descriptor(), "i64");
        assert_eq!(AbiType::Str.descriptor(), "string");
        assert_eq!(AbiType::Binary.descriptor(), "binary");
        assert_eq!(
            AbiType::Array(Box::new(AbiType::I64)).descriptor(),
            "array:i64"
        );
        assert_eq!(
            AbiType::Nullable(Box::new(AbiType::Str)).descriptor(),
            "option:string"
        );
        assert_eq!(
            AbiType::Array(Box::new(AbiType::Nullable(Box::new(AbiType::I32)))).descriptor(),
            "array:option:i32"
        );
    }

    #[test]
    fn abitype_to_data_type() {
        assert_eq!(AbiType::I64.to_data_type(), DataType::Long);
        assert_eq!(AbiType::Bool.to_data_type(), DataType::Boolean);
        assert_eq!(AbiType::Binary.to_data_type(), DataType::Binary);
        match AbiType::Array(Box::new(AbiType::Nullable(Box::new(AbiType::I64)))).to_data_type() {
            DataType::Array {
                element_type,
                contains_null,
            } => {
                assert_eq!(*element_type, DataType::Long);
                assert!(contains_null);
            }
            other => panic!("expected array, got {other:?}"),
        }
    }

    #[test]
    fn abitype_spark_json() {
        assert_eq!(AbiType::I64.to_spark_json(), serde_json::json!("long"));
        assert_eq!(
            AbiType::Array(Box::new(AbiType::Str)).to_spark_json(),
            serde_json::json!({"type": "array", "elementType": "string", "containsNull": false})
        );
    }

    fn spec_of(f: &UserDefinedFunction) -> serde_json::Value {
        serde_json::from_str(&f.build_spec().unwrap()).unwrap()
    }

    #[test]
    fn build_spec_encodes_signature_and_wasm() {
        let spec = spec_of(&sample());
        assert_eq!(spec["entrypoint"], "add_one");
        assert_eq!(spec["arg_types"], serde_json::json!(["i64"]));
        assert_eq!(spec["ret_type"], "i64");
        assert_eq!(spec["output_type"], "long");
        assert_eq!(spec["wasm_b64"], "AGFzbQEA");
    }

    #[test]
    fn eval_type_decides_whether_the_spec_carries_the_batch_entrypoint() {
        let batch = sample().with_batch_entrypoint("spark_batch_udf_add_one");
        let per_row = batch
            .clone()
            .with_eval_type(eval_type::SQL_ARROW_BATCHED_UDF);
        assert_eq!(batch.eval_type, eval_type::SQL_SCALAR_ARROW_UDF);
        assert_eq!(
            spec_of(&batch)["batch_entrypoint"],
            "spark_batch_udf_add_one"
        );
        assert_eq!(per_row.eval_type, eval_type::SQL_ARROW_BATCHED_UDF);
        assert_eq!(
            spec_of(&per_row)["batch_entrypoint"],
            serde_json::Value::Null
        );
    }

    #[test]
    fn scalar_arrow_eval_type_without_a_usable_batch_entrypoint_is_rejected() {
        let no_entrypoint = sample().with_eval_type(eval_type::SQL_SCALAR_ARROW_UDF);
        let no_args = udf("now", vec![0x00], "now", vec![], AbiType::I64)
            .with_batch_entrypoint("spark_batch_udf_now");
        let array_arg = udf(
            "sum",
            vec![0x00],
            "sum",
            vec![AbiType::Array(Box::new(AbiType::I64))],
            AbiType::I64,
        )
        .with_batch_entrypoint("spark_batch_udf_sum");
        let nested_option_ret = udf(
            "wrap",
            vec![0x00],
            "wrap",
            vec![AbiType::I64],
            AbiType::Nullable(Box::new(AbiType::Nullable(Box::new(AbiType::I64)))),
        )
        .with_batch_entrypoint("spark_batch_udf_wrap");
        let option_array_arg = udf(
            "first",
            vec![0x00],
            "first",
            vec![AbiType::Nullable(Box::new(AbiType::Array(Box::new(
                AbiType::I64,
            ))))],
            AbiType::I64,
        )
        .with_batch_entrypoint("spark_batch_udf_first");
        for (f, expected) in [
            (no_entrypoint, "needs a batch entrypoint"),
            (no_args, "has no arguments"),
            (array_arg, "has type 'array:i64'"),
            (nested_option_ret, "has type 'option:option:i64'"),
            (option_array_arg, "has type 'option:array:i64'"),
        ] {
            let err = f.to_payload().unwrap_err().message();
            assert!(err.contains(expected), "{err}");
        }
    }

    /// End-to-end packer test. Requires only a Python interpreter with `pyspark`
    /// importable (for its vendored `cloudpickle` + the pure-Python type parser);
    /// the packer script is embedded in the crate. Point `python_exe` at that
    /// interpreter via `$SPARK_CONNECT_PYTHON`. Run with
    /// `cargo test --features wasm-udf -- --ignored`.
    #[test]
    #[ignore]
    fn packer_produces_nonempty_command() {
        let f = sample();
        let cmd = f.build_command().expect("packer should succeed");
        assert!(!cmd.is_empty());
        assert_eq!(cmd[0], 0x80); // cloudpickle PROTO opcode
    }

    #[test]
    fn parse_python_version_output_trims_whitespace() {
        // Test that we correctly parse Python version strings from stdout
        // (with potential trailing whitespace/newline). Note: this test must not be
        // named `parse_python_version_output`, or it would shadow the function under
        // test (imported via `use super::*`) and fail to compile.
        assert_eq!(
            parse_python_version_output("3.11\n"),
            Some("3.11".to_string())
        );
        assert_eq!(
            parse_python_version_output("3.12"),
            Some("3.12".to_string())
        );
        assert_eq!(
            parse_python_version_output("3.9\n\n"),
            Some("3.9".to_string())
        );
        // Empty or invalid output
        assert_eq!(parse_python_version_output(""), None);
        assert_eq!(parse_python_version_output("   \n"), None);
    }
}
