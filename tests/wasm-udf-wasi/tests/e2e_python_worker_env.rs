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

//! The session's `spark.pythonWorkerEnv.*` variables, set from the Rust client, on a
//! live Spark Connect server that has them (SPARK-58752, Spark 4.4.0+), whose workers
//! need `pyarrow` and `wasmtime`. Run with `SPARK_REMOTE` set; skipped otherwise, and
//! on an older server.

use std::process::Command;

use spark_connect::column::{col, Column};
use spark_connect::expression::ExpressionNode;
use spark_connect::row::Value;
use spark_connect::session::SparkSession;
use spark_connect::types::DataType;
use spark_connect::udf::{eval_type, CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};
use spark_connect::wasm_udf::PythonPacker;
use wasm_udf_tests::udf as unknown;
use wasm_udf_wasi_tests::udf as wasi;

const PREFIX: &str = "spark.pythonWorkerEnv.";
const NONE: [Value; 4] = [Value::Null, Value::Null, Value::Null, Value::Null];

/// A Python UDF `name -> os.environ.get(name)`, pickled by the packer's Python.
fn python_env_var(arg: Column) -> Column {
    let script = r#"
import sys
from pyspark import cloudpickle
from pyspark.sql.types import StringType

def env_var(name):
    import os
    return os.environ.get(name)

print("%d.%d" % sys.version_info[:2])
print(cloudpickle.dumps((env_var, StringType())).hex())
"#;
    let output = Command::new(PythonPacker::default().python_exe)
        .args(["-c", script])
        .output()
        .expect("spawn python");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let (python_ver, hex) = stdout.trim().split_once('\n').unwrap();
    let command = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect();
    let payload = PythonUDFPayload::new(
        DataType::String {
            collation: "UTF8_BINARY".to_string(),
        },
        eval_type::SQL_BATCHED_UDF,
        command,
        python_ver.to_string(),
    );
    let udf = CommonInlineUserDefinedFunctionExpression::new(
        "env_var".to_string(),
        true,
        vec![arg.expression().clone()],
        payload,
    );
    Column::new(ExpressionNode::CommonInlineUserDefinedFunction(Box::new(udf)).into())
}

/// What a Python UDF, a wasm32-unknown-unknown UDF and a wasm32-wasip1 UDF, per row
/// and per batch, see as the variable `name`.
fn observe(spark: &SparkSession, name: &str) -> Vec<Value> {
    let rows = spark
        .sql(&format!("SELECT '{name}' AS name"))
        .unwrap()
        .select([
            python_env_var(col("name")),
            unknown::env_var(col("name")).unwrap(),
            wasi::env_var_udf()
                .with_eval_type(eval_type::SQL_ARROW_BATCHED_UDF)
                .call([col("name")])
                .unwrap(),
            wasi::env_var(col("name")).unwrap(),
        ])
        .collect()
        .unwrap();
    rows[0].values().to_vec()
}

/// Whether a server of the version `version`, such as `4.3.0` or `5.0.0-SNAPSHOT`,
/// delivers `spark.pythonWorkerEnv.*` to its workers.
fn has_python_worker_env(version: &str) -> bool {
    let mut parts = version
        .split(['.', '-'])
        .map(|p| p.parse::<u32>().unwrap_or(0));
    (parts.next().unwrap_or(0), parts.next().unwrap_or(0)) >= (4, 4)
}

fn s(v: &str) -> Value {
    Value::String(v.into())
}

#[test]
fn python_worker_env_reaches_python_and_wasi_udfs() {
    let Ok(url) = std::env::var("SPARK_REMOTE") else {
        return;
    };
    let spark = SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .expect("session");
    if !has_python_worker_env(&spark.version().unwrap()) {
        return;
    }
    let conf = spark.conf();
    let key = format!("{PREFIX}RUST_CLIENT_PROBE");

    assert_eq!(observe(&spark, "RUST_CLIENT_PROBE"), NONE);

    // A plain conf.set: no proto or client change is needed.
    assert_eq!(conf.set(&key, "hello").unwrap(), Vec::<String>::new());
    assert_eq!(conf.get(&key).unwrap().as_deref(), Some("hello"));
    assert_eq!(
        observe(&spark, "RUST_CLIENT_PROBE"),
        [s("hello"), Value::Null, s("hello"), s("hello")]
    );

    // A change takes effect on the next action.
    conf.set(&key, "world").unwrap();
    assert_eq!(
        observe(&spark, "RUST_CLIENT_PROBE"),
        [s("world"), Value::Null, s("world"), s("world")]
    );
    conf.unset(&key).unwrap();
    assert_eq!(observe(&spark, "RUST_CLIENT_PROBE"), NONE);

    // The server validates the write, and the Rust error carries the condition.
    for (name, condition) in [
        ("1BAD", "INVALID_PYTHON_WORKER_ENV_VAR_NAME"),
        ("SPARK_HOME", "RESERVED_PYTHON_WORKER_ENV_VAR_NAME"),
    ] {
        let err = conf.set(&format!("{PREFIX}{name}"), "x").unwrap_err();
        let text = format!("{} {}", err.error_class, err.message);
        assert!(text.contains(condition), "{name}: {err:?}");
    }

    // A silent batch, as PySpark's builder applies its options, sets the valid ones
    // and reports the rejected one as a warning that names no value.
    let warnings = conf
        .set_all(
            &[
                (format!("{PREFIX}1BAD"), "do-not-echo".to_string()),
                (format!("{PREFIX}RUST_OK"), "ok".to_string()),
            ],
            true,
        )
        .unwrap();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].contains(&format!("{PREFIX}1BAD")),
        "{warnings:?}"
    );
    assert!(!warnings[0].contains("do-not-echo"), "{warnings:?}");
    assert_eq!(observe(&spark, "RUST_OK")[3], s("ok"));
    conf.unset(&format!("{PREFIX}RUST_OK")).unwrap();

    // A name the redaction pattern matches is withheld on read, yet still delivered.
    let token = format!("{PREFIX}RUST_TOKEN");
    conf.set(&token, "s3cr3t").unwrap();
    // The server leaves the entry out of the response, so the read fails as for an
    // unknown key, as it does in PySpark.
    assert_eq!(
        conf.get(&token).unwrap_err().error_class,
        "SQL_CONF_NOT_FOUND"
    );
    assert!(!conf.get_all().unwrap().values().any(|v| v == "s3cr3t"));
    assert_eq!(
        observe(&spark, "RUST_TOKEN"),
        [s("s3cr3t"), Value::Null, s("s3cr3t"), s("s3cr3t")]
    );
    conf.unset(&token).unwrap();

    // The environment is the session's: a new session does not see this one's.
    conf.set(&key, "only-here").unwrap();
    let other = SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .expect("second session");
    assert_ne!(other.session_id(), spark.session_id());
    assert_eq!(observe(&other, "RUST_CLIENT_PROBE"), NONE);
    assert_eq!(observe(&spark, "RUST_CLIENT_PROBE")[0], s("only-here"));
    conf.unset(&key).unwrap();
}
