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

/// A case of the ML goldens of `scripts/capture_ml_golden.py` (`ml.jsonl`, `ml_extra.jsonl`):
/// the root relation of the plan of its transformed DataFrame, normalized by
/// [`normalize_relation`], and the fit of its model, if it fits one: the ML commands of the fit
/// and their results, an entry of the replay of `common/ml_golden.rs`.
pub struct MlGolden {
    pub relation: proto::Relation,
    pub fit: Option<serde_json::Value>,
}

/// The cases of the ML golden `tests/golden/<file>` at the repository root, by name.
pub fn load_ml_goldens(file: &str) -> HashMap<String, MlGolden> {
    let path = format!("{}/../../tests/golden/{file}", env!("CARGO_MANIFEST_DIR"));
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|_| panic!("golden file {file} missing"));
    let mut goldens = HashMap::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let case: serde_json::Value = serde_json::from_str(line).unwrap();
        let name = case["name"].as_str().unwrap().to_string();
        let bytes = STANDARD.decode(case["b64"].as_str().unwrap()).unwrap();
        let Some(proto::plan::OpType::Root(mut relation)) =
            proto::Plan::decode(bytes.as_slice()).unwrap().op_type
        else {
            panic!("{name}: the golden has no root relation");
        };
        normalize_relation(&mut relation);
        let fit = case.get("fit").cloned();
        let golden = MlGolden { relation, fit };
        assert!(
            goldens.insert(name.clone(), golden).is_none(),
            "{name} is in {file} twice"
        );
    }
    goldens
}

/// The vector `[id, 1.0]` of a row of a range, as the ML goldens build their features:
/// `array_to_vector(array(col("id").cast("double"), lit(1.0)))`.
pub fn ml_features() -> spark_connect::Column {
    use spark_connect::column::{col, lit_double};
    use spark_connect::expression::Expression;
    let array =
        spark_connect::functions::array(vec![col("id").cast_str("double"), lit_double(1.0)]);
    spark_connect::Column::new(Expression::function(
        "array_to_vector",
        vec![array.expression().clone()],
    ))
}

/// The dataset of the ML goldens: a vector column `features` of [`ml_features`] for each id of
/// a range of `rows`. It is a plan: building it sends no request.
pub fn ml_vectors(session: &SparkSession, rows: i64) -> spark_connect::DataFrame {
    session
        .range(rows)
        .unwrap()
        .select(vec![ml_features().alias("features")])
}

/// Clears what varies from run to run in a relation of an ML golden, and in its inputs, as
/// `scripts/capture_ml_golden.py` normalizes it: the `common` (plan ID and origin) of each
/// relation and expression, the plan IDs of the attributes, and the uid of a transformer, an
/// instance identifier, which the golden holds as a placeholder. The ObjectRef of a model stays:
/// a replay of the golden's fit gives the model the golden's placeholder. A relation or an
/// expression of a kind that the ML goldens do not have fails, rather than being compared as it
/// was sent.
pub fn normalize_relation(r: &mut proto::Relation) {
    use proto::relation::RelType;
    r.common = None;
    match r.rel_type.as_mut() {
        Some(RelType::MlRelation(ml_rel)) => match ml_rel.ml_type.as_mut() {
            Some(proto::ml_relation::MlType::Transform(transform)) => {
                if let Some(proto::ml_relation::transform::Operator::Transformer(op)) =
                    transform.operator.as_mut()
                {
                    op.uid.clear();
                }
                if let Some(input) = transform.input.as_deref_mut() {
                    normalize_relation(input);
                }
                // The values of `MlParams.params` are literals, with no plan ID or origin to clear.
            }
            other => panic!("an MlRelation that the ML goldens do not have: {other:?}"),
        },
        Some(RelType::Project(project)) => {
            if let Some(input) = project.input.as_deref_mut() {
                normalize_relation(input);
            }
            for expression in &mut project.expressions {
                normalize_expression(expression);
            }
        }
        Some(RelType::Range(_)) => {}
        other => panic!("a relation that the ML goldens do not have: {other:?}"),
    }
}

/// [`normalize_relation`] of an expression.
fn normalize_expression(e: &mut proto::Expression) {
    use proto::expression::ExprType;
    e.common = None;
    match e.expr_type.as_mut() {
        Some(ExprType::Alias(alias)) => {
            if let Some(expr) = alias.expr.as_deref_mut() {
                normalize_expression(expr);
            }
        }
        Some(ExprType::Cast(cast)) => {
            if let Some(expr) = cast.expr.as_deref_mut() {
                normalize_expression(expr);
            }
        }
        Some(ExprType::UnresolvedFunction(function)) => {
            for argument in &mut function.arguments {
                normalize_expression(argument);
            }
        }
        Some(ExprType::UnresolvedAttribute(attribute)) => attribute.plan_id = None,
        Some(ExprType::Literal(_)) => {}
        other => panic!("an expression that the ML goldens do not have: {other:?}"),
    }
}
