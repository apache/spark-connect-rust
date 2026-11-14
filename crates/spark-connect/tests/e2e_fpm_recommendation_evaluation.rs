// SPDX-License-Identifier: Apache-2.0
//! FPGrowth, ALS and the evaluators against a live server, as official PySpark 4.3.0 gives them of
//! the configurations of tests/golden/frequent_patterns.json, als.json and evaluators.json:
//! - the members of their models and evaluators (tests/golden/fpm_recommendation_evaluation.json);
//! - the save and load of every estimator, model and evaluator, with the members of the loaded
//!   copies (tests/golden/fpm_recommendation_evaluation_persistence.json).
//!
//! scripts/capture_fpm_recommendation_evaluation_golden.py captures both. Set SPARK_REMOTE.
#[path = "common/als.rs"]
mod als;
mod common;
#[path = "common/evaluators.rs"]
mod evaluators;
#[path = "common/ml_golden.rs"]
mod ml_golden;

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use common::live_session;
use ml_golden::*;
use serde_json::{json, Value as Json};
use spark_connect::ml::*;
use spark_connect::{DataFrame, SparkSession};

/// A fitted model or a configured evaluator of the golden.
enum Stage {
    FPGrowth(FPGrowthModel),
    Als(ALSModel),
    Evaluator(Box<dyn Evaluator>),
}

/// The frames of a case: the dataset of its fit or evaluation, the held-out rows that its model
/// transforms, and the users and items of the ALS subsets.
struct Frames {
    data: DataFrame,
    held: Option<DataFrame>,
    users: Option<DataFrame>,
    items: Option<DataFrame>,
}

fn fpgrowth(values: &Json) -> FPGrowth {
    let mut estimator = FPGrowth::new();
    for (name, value) in values.as_object().unwrap() {
        estimator = match name.as_str() {
            "minSupport" => estimator.set_min_support(value.as_f64().unwrap()),
            "minConfidence" => estimator.set_min_confidence(value.as_f64().unwrap()),
            "itemsCol" => estimator.set_items_col(value.as_str().unwrap()),
            "predictionCol" => estimator.set_prediction_col(value.as_str().unwrap()),
            "numPartitions" => {
                estimator.set_num_partitions(value.as_i64().unwrap().try_into().unwrap())
            }
            other => panic!("unexpected FPGrowth parameter: {other}"),
        };
    }
    estimator
}

fn load_evaluator(class: &str, session: &SparkSession, path: &str) -> Box<dyn Evaluator> {
    match class {
        "BinaryClassificationEvaluator" => {
            Box::new(BinaryClassificationEvaluator::load(session, path).unwrap())
        }
        "RegressionEvaluator" => Box::new(RegressionEvaluator::load(session, path).unwrap()),
        "MulticlassClassificationEvaluator" => {
            Box::new(MulticlassClassificationEvaluator::load(session, path).unwrap())
        }
        "MultilabelClassificationEvaluator" => {
            Box::new(MultilabelClassificationEvaluator::load(session, path).unwrap())
        }
        "ClusteringEvaluator" => Box::new(ClusteringEvaluator::load(session, path).unwrap()),
        "RankingEvaluator" => Box::new(RankingEvaluator::load(session, path).unwrap()),
        other => panic!("uncovered evaluator {other}"),
    }
}

/// The member `label` of `stage`, as the golden labels it.
fn member(stage: &mut Stage, label: &str, frames: &Frames) -> Result<Json> {
    let held = || frames.held.as_ref().unwrap();
    Ok(match (stage, label) {
        (Stage::FPGrowth(m), "freqItemsets") => frame(&m.freq_itemsets()?)?,
        (Stage::FPGrowth(m), "associationRules") => frame(&m.association_rules()?)?,
        (Stage::FPGrowth(m), "transform") => frame(&m.transform(held())?)?,
        (Stage::Als(m), "rank") => json!(m.rank()),
        (Stage::Als(m), "userFactors") => frame(&m.user_factors()?)?,
        (Stage::Als(m), "itemFactors") => frame(&m.item_factors()?)?,
        (Stage::Als(m), "transform") => frame(&m.transform(held())?)?,
        (Stage::Als(m), "recommendForAllUsers-2") => frame(&m.recommend_for_all_users(2)?)?,
        (Stage::Als(m), "recommendForAllItems-2") => frame(&m.recommend_for_all_items(2)?)?,
        (Stage::Als(m), "recommendForUserSubset-2") => {
            frame(&m.recommend_for_user_subset(frames.users.as_ref().unwrap(), 2)?)?
        }
        (Stage::Als(m), "recommendForItemSubset-2") => {
            frame(&m.recommend_for_item_subset(frames.items.as_ref().unwrap(), 2)?)?
        }
        (Stage::Evaluator(e), "evaluate") => num(e.evaluate(&frames.data)?),
        (Stage::Evaluator(e), "isLargerBetter()") => json!(e.is_larger_better()),
        (_, label) => panic!("no member {label} in this test"),
    })
}

/// Each member of `members` (a golden `members` map) of `stage`: its value, or its error. The
/// repr() and str() of the Python objects are the Python classes' to compare.
fn assert_members(stage: &mut Stage, members: &Json, frames: &Frames, at: &str) {
    for (label, entry) in members.as_object().unwrap() {
        if label == "repr()" || label == "str()" {
            continue;
        }
        let at = format!("{at}.{label}");
        match (member(stage, label, frames), entry.get("value")) {
            (Ok(actual), Some(expected)) => {
                assert_matches(&normalized(actual), &normalized(expected.clone()), &at)
            }
            (Err(error), None) => assert_error(&error.to_string(), &entry["error"], &at),
            (actual, _) => panic!("{at}: {actual:?}, official {entry}"),
        }
    }
}

fn path_of(dir: &Path, key: &str) -> String {
    dir.join(key.replace(['.', '-'], "_"))
        .to_str()
        .unwrap()
        .to_string()
}

/// The uid and params of a loaded stage are those it was saved with, as official PySpark's record
/// has them.
fn assert_loaded(
    record: &Json,
    loaded: (&MlOperator, &Params),
    saved: (&MlOperator, &Params),
    key: &str,
) {
    assert_eq!(record["uid_same"], true, "{key}");
    assert_eq!(record["params_before"], record["params_after"], "{key}");
    assert_eq!(loaded.0.name, saved.0.name, "{key}");
    assert_eq!(loaded.0.uid, saved.0.uid, "{key}");
    assert_params(loaded.1, saved.1, key);
}

#[test]
fn fpm_recommendation_and_evaluation_match_official_pyspark() {
    let Some(session) = live_session() else {
        return;
    };
    let golden = fixture(include_str!(
        "../../../tests/golden/fpm_recommendation_evaluation.json"
    ));
    let records = fixture(include_str!(
        "../../../tests/golden/fpm_recommendation_evaluation_persistence.json"
    ))["persistence"]
        .clone();
    let datasets: HashMap<&str, DataFrame> = golden["datasets"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, spec)| (name.as_str(), dataset(&session, spec)))
        .collect();
    let frame_of = |name: &str| datasets[name].clone();
    let dir = std::env::temp_dir().join(format!("fpm-rec-eval-{}", std::process::id()));
    let mut seen = BTreeSet::new();

    let frequent = fixture(include_str!("../../../tests/golden/frequent_patterns.json"));
    for case in frequent["fpgrowth"].as_array().unwrap() {
        let key = format!("FPGrowth.{}", case["id"].as_str().unwrap());
        let name = format!("fpm_{}", case["id"].as_str().unwrap().replace('-', "_"));
        let frames = Frames {
            data: frame_of(&name),
            held: Some(frame_of(&format!("{name}_held"))),
            users: None,
            items: None,
        };
        assert_eq!(golden["fits"][&key]["dataset"], name.as_str());
        let mut estimator = fpgrowth(&case["values"]);
        let fitted = estimator.fit(&frames.data).unwrap();
        let mut stage = Stage::FPGrowth(fitted.clone());
        assert_members(
            &mut stage,
            &golden["models"][&key]["members"],
            &frames,
            &key,
        );

        let model_key = format!("{key}.model");
        let path = path_of(&dir, &model_key);
        fitted.write().unwrap().overwrite().save(&path).unwrap();
        let loaded = FPGrowthModel::load(&session, &path).unwrap();
        let record = &records[&model_key];
        assert_loaded(
            record,
            (Transformer::operator(&loaded), Transformer::params(&loaded)),
            (Transformer::operator(&fitted), Transformer::params(&fitted)),
            &model_key,
        );
        let mut loaded = Stage::FPGrowth(loaded);
        assert_members(
            &mut loaded,
            &record["members"]["members"],
            &frames,
            &model_key,
        );
        seen.insert(model_key);

        let estimator_key = format!("{key}.estimator");
        let path = path_of(&dir, &estimator_key);
        Estimator::write(&estimator)
            .session(&session)
            .overwrite()
            .save(&path)
            .unwrap();
        let loaded = FPGrowth::load(&session, &path).unwrap();
        assert_loaded(
            &records[&estimator_key],
            (Estimator::operator(&loaded), Estimator::params(&loaded)),
            (
                Estimator::operator(&estimator),
                Estimator::params(&estimator),
            ),
            &estimator_key,
        );
        seen.insert(estimator_key);
    }

    let fixture_als = als::fixture();
    let frames = Frames {
        data: frame_of("als_train"),
        held: Some(frame_of("als_held")),
        users: Some(frame_of("als_users")),
        items: Some(frame_of("als_items")),
    };
    for case in fixture_als["cases"].as_array().unwrap() {
        let key = format!("ALS.{}", case["name"].as_str().unwrap());
        assert_eq!(golden["fits"][&key]["dataset"], "als_train");
        let mut estimator = als::configured(&case["values"]);
        let fitted = estimator.fit(&frames.data).unwrap();
        let mut stage = Stage::Als(fitted.clone());
        assert_members(
            &mut stage,
            &golden["models"][&key]["members"],
            &frames,
            &key,
        );

        let model_key = format!("{key}.model");
        let path = path_of(&dir, &model_key);
        fitted.write().unwrap().overwrite().save(&path).unwrap();
        let loaded = ALSModel::load(&session, &path).unwrap();
        let record = &records[&model_key];
        assert_loaded(
            record,
            (Transformer::operator(&loaded), Transformer::params(&loaded)),
            (Transformer::operator(&fitted), Transformer::params(&fitted)),
            &model_key,
        );
        let mut loaded = Stage::Als(loaded);
        assert_members(
            &mut loaded,
            &record["members"]["members"],
            &frames,
            &model_key,
        );
        seen.insert(model_key);

        let estimator_key = format!("{key}.estimator");
        let path = path_of(&dir, &estimator_key);
        Estimator::write(&estimator)
            .session(&session)
            .overwrite()
            .save(&path)
            .unwrap();
        let loaded = ALS::load(&session, &path).unwrap();
        assert_loaded(
            &records[&estimator_key],
            (Estimator::operator(&loaded), Estimator::params(&loaded)),
            (
                Estimator::operator(&estimator),
                Estimator::params(&estimator),
            ),
            &estimator_key,
        );
        seen.insert(estimator_key);
    }

    let mut classes = BTreeSet::new();
    for case in evaluators::fixture()["cases"].as_array().unwrap() {
        let class = case["class"].as_str().unwrap();
        if !classes.insert(class) {
            continue;
        }
        let key = case["name"].as_str().unwrap();
        let frames = Frames {
            data: frame_of(&format!("evaluate_{}", case["dataset"].as_str().unwrap())),
            held: None,
            users: None,
            items: None,
        };
        let evaluator = evaluators::configured(class, &case["values"]);
        let path = path_of(&dir, &format!("{key}.evaluator"));
        evaluator
            .write()
            .session(&session)
            .overwrite()
            .save(&path)
            .unwrap();
        let loaded = load_evaluator(class, &session, &path);
        let evaluator_key = format!("{key}.evaluator");
        let record = &records[&evaluator_key];
        assert_loaded(
            record,
            (loaded.operator(), loaded.params()),
            (evaluator.operator(), evaluator.params()),
            &evaluator_key,
        );
        let mut stage = Stage::Evaluator(evaluator);
        assert_members(&mut stage, &golden["models"][key]["members"], &frames, key);
        let mut loaded = Stage::Evaluator(loaded);
        assert_members(
            &mut loaded,
            &record["members"]["members"],
            &frames,
            &evaluator_key,
        );
        seen.insert(evaluator_key);
    }

    let official: BTreeSet<String> = records.as_object().unwrap().keys().cloned().collect();
    assert_eq!(seen, official);
    let _ = std::fs::remove_dir_all(&dir);
    session.stop().unwrap();
}
