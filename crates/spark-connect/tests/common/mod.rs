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

//! Helpers shared by the integration tests of this crate. A test file takes them with
//! `mod common;`.

#![allow(dead_code)] // every test binary compiles this module and uses only a part of it

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::SparkSession;
use spark_connect_proto as proto;

/// Whether the tests of a live server run: only when `SPARK_REMOTE` is set.
pub fn should_run() -> bool {
    std::env::var("SPARK_REMOTE").is_ok()
}

/// A session of the server at `SPARK_REMOTE`, or of `sc://localhost:15002` without it. The
/// tests that use it return early when [`should_run`] is false.
pub fn session() -> SparkSession {
    let url = std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .expect("session")
}

/// A session of the server at `SPARK_REMOTE`, or `None` without it, for the tests that skip
/// themselves with `let Some(spark) = live_session() else { return }`.
pub fn live_session() -> Option<SparkSession> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    Some(
        SparkSession::builder()
            .remote(&remote)
            .get_or_create()
            .unwrap(),
    )
}

/// The cases of `tests/golden/<file>` at the repository root, by name. Each line of the file
/// is a JSON object with the `name` of the case and the `b64` of its message, which is decoded
/// as `M` and passed to `normalize` before it is kept.
pub fn load_goldens<M: Message + Default>(
    file: &str,
    mut normalize: impl FnMut(&mut M),
) -> HashMap<String, M> {
    // Goldens live at the repo root; tests run with CWD at the crate dir.
    let path = format!("{}/../../tests/golden/{file}", env!("CARGO_MANIFEST_DIR"));
    let file = File::open(path).unwrap_or_else(|_| panic!("golden file {file} missing"));
    let mut out = HashMap::new();
    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let obj: serde_json::Value = serde_json::from_str(&line).unwrap();
        let name = obj["name"].as_str().unwrap().to_string();
        let b64 = obj["b64"].as_str().unwrap();
        let bytes = STANDARD.decode(b64).unwrap();
        let mut message = M::decode(&bytes[..]).unwrap();
        normalize(&mut message);
        out.insert(name, message);
    }
    out
}

/// Clears what varies from run to run in the relation of an ML golden: `common` of the
/// relation itself and, for an `MlRelation` transform, the UID of its transformer. It does not
/// descend into the inputs of the relation; `plans_golden.rs` has the recursive form for plans.
pub fn normalize_relation(r: &mut proto::Relation) {
    r.common = None;
    if let Some(proto::relation::RelType::MlRelation(ml_rel)) = &mut r.rel_type {
        normalize_ml_relation(ml_rel);
    }
}

fn normalize_ml_relation(ml_rel: &mut proto::MlRelation) {
    if let Some(proto::ml_relation::MlType::Transform(transform)) = ml_rel.ml_type.as_mut() {
        if let Some(proto::ml_relation::transform::Operator::Transformer(op)) =
            transform.operator.as_mut()
        {
            // UIDs are instance identifiers, so they differ across runs.
            op.uid.clear();
        }
        // The values of `MlParams.params` are literals, with no plan ID or origin to clear.
    }
}
