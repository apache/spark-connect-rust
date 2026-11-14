// SPDX-License-Identifier: Apache-2.0
//! The typed accessors of the clustering models, their summaries and their persistence, against
//! the entries that official PySpark 4.3.0 captured (`tests/golden/clustering_*.json`). No Spark
//! server is needed: the replay server of `common/ml_golden.rs` answers each ML command with the
//! golden result of that command, and the test checks that the accessor sent the golden commands
//! and returned the golden value. The commands are compared after the golden's normalization, per
//! entry (see `normalize_commands`).
//!
//! The replay answers ML commands only. The members whose golden is a DataFrame are skipped: the
//! `transform`, the `cluster`, `predictions` and `probability` of a summary, the `gaussiansDF` and
//! `describeTopics` of a model. `gaussians` is skipped as well: it collects `gaussiansDF`, so its
//! command is a relation. The `toLocal` of a distributed LDA model is one fetch, which the test
//! makes once and reads both as the member and as the local model.

#[path = "common/ml_golden.rs"]
mod ml_golden;

use base64::{engine::general_purpose::STANDARD, Engine};
use ml_golden::{
    assert_commands, fixture, golden_commands, matrix, num, nums, vector, ReplayServer,
};
use prost::Message;
use serde_json::{json, Value};
use spark_connect::ml::{
    fetch, BisectingKMeans, BisectingKMeansModel, BisectingKMeansSummary, ClusteringSummary,
    DistributedLDAModel, Estimator, GaussianMixture, GaussianMixtureModel, GaussianMixtureSummary,
    KMeans, KMeansModel, KMeansSummary, LDAModelApi, LocalLDAModel, MlMatrix, MlTarget, MlVector,
    Model, Params, PowerIterationClustering, LDA,
};
use spark_connect::{DataFrame, SparkSession};
use spark_connect_core::error::Result as CoreResult;
use spark_connect_proto as proto;

const MODELS: &str = include_str!("../../../tests/golden/clustering_models.json");
const SUMMARIES: &str = include_str!("../../../tests/golden/clustering_summaries.json");
const PERSISTENCE: &str = include_str!("../../../tests/golden/clustering_persistence.json");

/// The path that the persistence entries save to and load from: the golden normalizes it.
const PATH: &str = "/unused/path";

fn golden(text: &str) -> Value {
    fixture(text)
}

/// Assert that `actual` is the golden `expected`: numbers to 1e-9 relative, the rest exactly. The
/// golden renders NaN and the infinities as strings, as `ml_golden::num` does.
fn assert_close(actual: &Value, expected: &Value, path: &str) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!(
                (a - e).abs() <= 1e-9 * e.abs().max(1.0),
                "{path}: {a} != {e}"
            );
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{path}: length");
            for (index, (a, e)) in a.iter().zip(e).enumerate() {
                assert_close(a, e, &format!("{path}[{index}]"));
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            let mut keys: Vec<_> = a.keys().collect();
            let mut expected_keys: Vec<_> = e.keys().collect();
            keys.sort();
            expected_keys.sort();
            assert_eq!(keys, expected_keys, "{path}: keys");
            for (key, e) in e {
                assert_close(&a[key], e, &format!("{path}.{key}"));
            }
        }
        _ => assert_eq!(actual, expected, "{path}"),
    }
}

/// A member's value or its error, as the typed accessor returns it, rendered as the golden.
type Member = CoreResult<Value>;

fn plain<T: Into<Value>>(value: CoreResult<T>) -> Member {
    value.map(Into::into)
}

fn double(value: CoreResult<f64>) -> Member {
    value.map(num)
}

fn doubles(value: CoreResult<Vec<f64>>) -> Member {
    value.map(|values| nums(&values))
}

fn rows(value: CoreResult<Vec<Vec<f64>>>) -> Member {
    value.map(|rows| rows.iter().map(|row| nums(row)).collect())
}

fn vector_of(value: CoreResult<MlVector>) -> Member {
    value.map(|v| vector(&v))
}

fn matrix_of(value: CoreResult<MlMatrix>) -> Member {
    value.map(|m| matrix(&m))
}

/// The point that a golden `predict` member of a clustering model takes, by the suffix of its
/// name: `dense` is `Vectors.dense([6.1, 6.2])`, and `sparse` is `Vectors.sparse(2, [1], [1.3])`.
fn point(key: &str) -> MlVector {
    match key.rsplit('-').next().unwrap() {
        "dense" => MlVector::Dense(vec![6.1, 6.2]),
        "sparse" => MlVector::Sparse {
            size: 2,
            indices: vec![1],
            values: vec![1.3],
        },
        other => panic!("unknown point {other}"),
    }
}

/// Whether a golden member is a DataFrame, which the replay does not serve (see the module doc).
fn is_frame(entry: &Value) -> bool {
    entry["value"]["schema"].is_string()
}

/// Check one member of a golden entry against its actual value or error, and its commands.
fn check_member(server: &ReplayServer, entry: &Value, actual: Member, label: &str) {
    match entry.get("error") {
        Some(error) => {
            let message = error["message"].as_str().unwrap();
            let error = actual.expect_err(&format!("{label}: expected an error"));
            assert!(
                error.to_string().contains(message),
                "{label}: {error} does not carry {message}"
            );
        }
        None => assert_close(
            &actual.unwrap_or_else(|error| panic!("{label}: {error}")),
            &entry["value"],
            label,
        ),
    }
    assert_commands(server, entry, label);
}

/// The class name of a golden entry's value, as PySpark names it.
fn class_of(value: &Value) -> &str {
    value["type"].as_str().unwrap()
}

/// Check the golden members of an object (a model, a summary or a loaded copy) that `dispatch`
/// reads. The members are skipped that the replay does not serve: the Python `repr` and `str`,
/// the DataFrames, and the members of `skip`, which the caller checks itself or which are
/// relations. Returns the number of members checked.
fn check_members(
    server: &ReplayServer,
    members: &Value,
    dispatch: &dyn Fn(&str) -> Option<Member>,
    skip: &[&str],
    label: &str,
) -> usize {
    let mut checked = 0;
    for (key, entry) in members.as_object().unwrap() {
        if key.starts_with("repr")
            || key.starts_with("str")
            || skip.contains(&key.as_str())
            || is_frame(entry)
        {
            continue;
        }
        let Some(member) = dispatch(key) else {
            panic!("{label}: no accessor for {key}");
        };
        check_member(server, entry, member, &format!("{label}.{key}"));
        checked += 1;
    }
    checked
}

/// The params that a golden fit sent: the fit's params, which the estimator of the replay takes.
fn fitted_params(fit: &Value) -> Params {
    match golden_commands(&fit["fit"])[0].command.clone() {
        Some(proto::ml_command::Command::Fit(sent)) => Params::from_proto(&sent.params.unwrap()),
        other => panic!("the golden fit is not a Fit: {other:?}"),
    }
}

/// The uid that a golden load read: the golden normalizes it to a placeholder, so the loaded uid
/// is the placeholder too.
fn result_uid(load: &Value) -> String {
    let result = proto::MlCommandResult::decode(
        STANDARD
            .decode(load["results"][0].as_str().unwrap())
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    match result.result_type {
        Some(proto::ml_command_result::ResultType::OperatorInfo(info)) => info.uid.unwrap(),
        other => panic!("not an operator info: {other:?}"),
    }
}

/// The params that a golden save of an operator sent: the `MlParams` of its `Write`, which are the
/// params that PySpark read with the load, as it sends them again (see `read` of the ML object
/// module), so that the loaded params of the Rust client are compared with them.
fn saved_params(save: &Value) -> proto::MlParams {
    match golden_commands(save)[0].command.clone() {
        Some(proto::ml_command::Command::Write(write)) => write.params.unwrap_or_default(),
        other => panic!("the golden save is not a Write: {other:?}"),
    }
}

/// The estimator of a golden persistence entry, after its load: the load's commands, uid and
/// params, then the save's commands.
fn check_loaded_estimator<E: Estimator>(
    server: &ReplayServer,
    session: &SparkSession,
    entry: &Value,
    estimator: &E,
    label: &str,
) {
    assert_commands(server, &entry["load"], label);
    assert_eq!(
        estimator.operator().uid,
        result_uid(&entry["load"]),
        "{label}: uid"
    );
    assert_eq!(
        Estimator::params(estimator).to_proto(),
        saved_params(&entry["save"]),
        "{label}: params"
    );
    estimator
        .write()
        .session(session)
        .overwrite()
        .save(PATH)
        .unwrap();
    assert_commands(server, &entry["save"], label);
}

/// The model of a golden persistence entry, after its load: the load's commands and uid, its
/// members (the predictions of the copy), then the save's commands.
fn check_loaded_model<M: Model>(
    server: &ReplayServer,
    entry: &Value,
    model: &M,
    dispatch: &dyn Fn(&str) -> Option<Member>,
    label: &str,
) {
    assert_commands(server, &entry["load"], label);
    assert_eq!(
        model.operator().uid,
        result_uid(&entry["load"]),
        "{label}: uid"
    );
    check_members(server, &entry["members"]["members"], dispatch, &[], label);
    model.write().unwrap().overwrite().save(PATH).unwrap();
    assert_commands(server, &entry["save"], label);
}

// The members of the models. Each returns None for a member it has no accessor for.

fn kmeans_model_member(model: &KMeansModel, key: &str) -> Option<Member> {
    if key.starts_with("predict-") {
        return Some(plain(model.predict(&point(key))));
    }
    Some(match key {
        "clusterCenters()" => rows(model.cluster_centers()),
        "numFeatures" => plain(model.num_features()),
        "hasSummary" => Ok(json!(model.has_summary())),
        _ => return None,
    })
}

fn bisecting_model_member(
    model: &BisectingKMeansModel,
    key: &str,
    dataset: &DataFrame,
) -> Option<Member> {
    if key.starts_with("predict-") {
        return Some(plain(model.predict(&point(key))));
    }
    Some(match key {
        "clusterCenters()" => rows(model.cluster_centers()),
        "computeCost" => double(model.compute_cost(dataset)),
        "numFeatures" => plain(model.num_features()),
        "hasSummary" => Ok(json!(model.has_summary())),
        _ => return None,
    })
}

fn gaussian_mixture_model_member(model: &GaussianMixtureModel, key: &str) -> Option<Member> {
    if key.starts_with("predict-") {
        return Some(plain(model.predict(&point(key))));
    }
    if key.starts_with("predictProbability-") {
        return Some(vector_of(model.predict_probability(&point(key))));
    }
    Some(match key {
        "numFeatures" => plain(model.num_features()),
        "weights" => doubles(model.weights()),
        "hasSummary" => Ok(json!(model.has_summary())),
        _ => return None,
    })
}

/// The members of an LDA model that the local and the distributed model share.
fn lda_member<M: LDAModelApi>(model: &M, dataset: &DataFrame, key: &str) -> Option<Member> {
    Some(match key {
        "estimatedDocConcentration()" => vector_of(model.estimated_doc_concentration()),
        "isDistributed()" => plain(model.is_distributed()),
        "logLikelihood" => double(model.log_likelihood(dataset)),
        "logPerplexity" => double(model.log_perplexity(dataset)),
        "topicsMatrix()" => matrix_of(model.topics_matrix()),
        "vocabSize()" => plain(model.vocab_size()),
        _ => return None,
    })
}

/// The members of a distributed LDA model that the local one does not have.
fn distributed_member(model: &DistributedLDAModel, key: &str) -> Option<Member> {
    Some(match key {
        "getCheckpointFiles()" => plain(model.get_checkpoint_files()),
        "logPrior()" => double(model.log_prior()),
        "trainingLogLikelihood()" => double(model.training_log_likelihood()),
        _ => return None,
    })
}

// The members of the summaries.

/// The members of a clustering summary, which every summary of the family shares.
fn clustering_summary_member<S: ClusteringSummary>(summary: &S, key: &str) -> Option<Member> {
    Some(match key {
        "clusterSizes" => plain(summary.cluster_sizes()),
        "featuresCol" => plain(summary.features_col()),
        "k" => plain(summary.k()),
        "numIter" => plain(summary.num_iter()),
        "predictionCol" => plain(summary.prediction_col()),
        _ => return None,
    })
}

fn kmeans_summary_member(summary: &KMeansSummary, key: &str) -> Option<Member> {
    clustering_summary_member(summary, key).or_else(|| match key {
        "trainingCost" => Some(double(summary.training_cost())),
        _ => None,
    })
}

fn bisecting_summary_member(summary: &BisectingKMeansSummary, key: &str) -> Option<Member> {
    clustering_summary_member(summary, key).or_else(|| match key {
        "trainingCost" => Some(double(summary.training_cost())),
        _ => None,
    })
}

fn gaussian_mixture_summary_member(summary: &GaussianMixtureSummary, key: &str) -> Option<Member> {
    clustering_summary_member(summary, key).or_else(|| match key {
        "logLikelihood" => Some(double(summary.log_likelihood())),
        "probabilityCol" => Some(plain(summary.probability_col())),
        _ => None,
    })
}

/// The golden entries of a family's fit and model, and of its training summary.
fn family_entries<'a>(models: &'a Value, summaries: &'a Value, family: &str) -> Vec<&'a Value> {
    vec![
        &models["fits"][family]["fit"],
        &models["models"][family],
        &summaries["summaries"][format!("{family}.training").as_str()],
    ]
}

/// Check a fitted model's summary: its class, the summary's members, and the members of the model
/// that read it. The golden `summary` member has no command of its own.
fn check_training_summary<S>(
    server: &ReplayServer,
    entry: &Value,
    summary: &S,
    training: &Value,
    class: &str,
    dispatch: &dyn Fn(&S, &str) -> Option<Member>,
    family: &str,
) {
    assert_commands(server, &entry["members"]["summary"], family);
    assert_eq!(class_of(&entry["members"]["summary"]["value"]), class);
    check_members(
        server,
        &training["members"],
        &|key| dispatch(summary, key),
        &[],
        &format!("{family}.training"),
    );
}

/// Fit a KMeans of the golden's family, and check the fit, the model and its training summary.
fn check_kmeans_family(family: &str) {
    let models = golden(MODELS);
    let summaries = golden(SUMMARIES);
    let fit = &models["fits"][family];
    let entry = &models["models"][family];
    let training = &summaries["summaries"][format!("{family}.training").as_str()];
    let (server, session) = ReplayServer::start(&family_entries(&models, &summaries, family));
    let mut estimator = KMeans::new();
    *estimator.params_mut() = fitted_params(fit);
    let model = estimator.fit(&session.range(1).unwrap()).unwrap();
    assert_commands(&server, &fit["fit"], family);
    check_members(
        &server,
        &entry["members"],
        &|key| kmeans_model_member(&model, key),
        &["summary"],
        family,
    );
    let summary = model.summary().unwrap();
    check_training_summary(
        &server,
        entry,
        &summary,
        training,
        "KMeansSummary",
        &|s, key| kmeans_summary_member(s, key),
        family,
    );
}

/// Fit a BisectingKMeans of the golden's family, and check it as `check_kmeans_family`.
fn check_bisecting_family(family: &str) {
    let models = golden(MODELS);
    let summaries = golden(SUMMARIES);
    let fit = &models["fits"][family];
    let entry = &models["models"][family];
    let training = &summaries["summaries"][format!("{family}.training").as_str()];
    let (server, session) = ReplayServer::start(&family_entries(&models, &summaries, family));
    let mut estimator = BisectingKMeans::new();
    *estimator.params_mut() = fitted_params(fit);
    let model = estimator.fit(&session.range(1).unwrap()).unwrap();
    assert_commands(&server, &fit["fit"], family);
    let dataset = session.range(1).unwrap();
    check_members(
        &server,
        &entry["members"],
        &|key| bisecting_model_member(&model, key, &dataset),
        &["summary"],
        family,
    );
    let summary = model.summary().unwrap();
    check_training_summary(
        &server,
        entry,
        &summary,
        training,
        "BisectingKMeansSummary",
        &|s, key| bisecting_summary_member(s, key),
        family,
    );
}

/// Fit a GaussianMixture of the golden's family, and check it as `check_kmeans_family`. Its
/// `gaussians` is skipped: it collects the DataFrame `gaussiansDF`, a relation (see the module
/// doc).
fn check_gaussian_mixture_family(family: &str) {
    let models = golden(MODELS);
    let summaries = golden(SUMMARIES);
    let fit = &models["fits"][family];
    let entry = &models["models"][family];
    let training = &summaries["summaries"][format!("{family}.training").as_str()];
    let (server, session) = ReplayServer::start(&family_entries(&models, &summaries, family));
    let mut estimator = GaussianMixture::new();
    *estimator.params_mut() = fitted_params(fit);
    let model = estimator.fit(&session.range(1).unwrap()).unwrap();
    assert_commands(&server, &fit["fit"], family);
    check_members(
        &server,
        &entry["members"],
        &|key| gaussian_mixture_model_member(&model, key),
        &["summary", "gaussians"],
        family,
    );
    let summary = model.summary().unwrap();
    check_training_summary(
        &server,
        entry,
        &summary,
        training,
        "GaussianMixtureSummary",
        &|s, key| gaussian_mixture_summary_member(s, key),
        family,
    );
}

#[test]
fn kmeans_default_matches_official_pyspark() {
    check_kmeans_family("KMeans.default");
}

#[test]
fn kmeans_cosine_random_matches_official_pyspark() {
    check_kmeans_family("KMeans.cosine-random");
}

#[test]
fn kmeans_weighted_matches_official_pyspark() {
    check_kmeans_family("KMeans.weighted");
}

#[test]
fn bisecting_kmeans_default_matches_official_pyspark() {
    check_bisecting_family("BisectingKMeans.default");
}

#[test]
fn bisecting_kmeans_min_divisible_2_matches_official_pyspark() {
    check_bisecting_family("BisectingKMeans.min-divisible-2");
}

#[test]
fn bisecting_kmeans_cosine_matches_official_pyspark() {
    check_bisecting_family("BisectingKMeans.cosine");
}

#[test]
fn gaussian_mixture_default_matches_official_pyspark() {
    check_gaussian_mixture_family("GaussianMixture.default");
}

#[test]
fn gaussian_mixture_weighted_matches_official_pyspark() {
    check_gaussian_mixture_family("GaussianMixture.weighted");
}

/// The LDA model of the golden's family: a DistributedLDAModel of the em optimizer, whose
/// `toLocal` is a LocalLDAModel, or a LocalLDAModel of the online optimizer.
#[test]
fn lda_em_matches_official_pyspark() {
    let models = golden(MODELS);
    let family = "LDA.em";
    let fit = &models["fits"][family];
    let entry = &models["models"][family];
    let local = &models["models"][format!("{family}.toLocal").as_str()];
    let (server, session) = ReplayServer::start(&[&fit["fit"], entry, local]);
    let mut estimator = LDA::new();
    *estimator.params_mut() = fitted_params(fit);
    let model = estimator
        .fit(&session.range(1).unwrap())
        .unwrap()
        .into_distributed()
        .expect("the em optimizer fits a distributed model");
    assert_commands(&server, &fit["fit"], family);
    let dataset = session.range(1).unwrap();
    check_members(
        &server,
        &entry["members"],
        &|key| lda_member(&model, &dataset, key).or_else(|| distributed_member(&model, key)),
        &["toLocal"],
        family,
    );
    // The golden `toLocal` is one fetch: the local model is read from it, and its members are
    // checked on the local model.
    let to_local = model.to_local().unwrap();
    assert_commands(&server, &entry["members"]["toLocal"], family);
    assert_eq!(
        class_of(&entry["members"]["toLocal"]["value"]),
        "LocalLDAModel"
    );
    check_members(
        &server,
        &local["members"],
        &|key| lda_member(&to_local, &dataset, key),
        &[],
        &format!("{family}.toLocal"),
    );
}

#[test]
fn lda_online_matches_official_pyspark() {
    let models = golden(MODELS);
    let family = "LDA.online";
    let fit = &models["fits"][family];
    let entry = &models["models"][family];
    let (server, session) = ReplayServer::start(&[&fit["fit"], entry]);
    let mut estimator = LDA::new();
    *estimator.params_mut() = fitted_params(fit);
    let model = estimator
        .fit(&session.range(1).unwrap())
        .unwrap()
        .into_local()
        .expect("the online optimizer fits a local model");
    assert_commands(&server, &fit["fit"], family);
    let dataset = session.range(1).unwrap();
    check_members(
        &server,
        &entry["members"],
        &|key| lda_member(&model, &dataset, key),
        &[],
        family,
    );
}

/// The persistence of each family: its estimator and its model are saved and loaded back, and the
/// loaded model's predictions and commands are those of PySpark's persistence entries. Every
/// family runs even when an earlier one fails, and the failures are reported together.
#[test]
fn clustering_persistence_matches_official_pyspark() {
    let persistence = golden(PERSISTENCE)["persistence"].clone();
    let estimator = |family: &str| persistence[format!("{family}.estimator").as_str()].clone();
    let model = |family: &str| persistence[format!("{family}.model").as_str()].clone();
    let mut failures = Vec::new();
    let mut check = |family: &str, run: &dyn Fn()| {
        if let Err(panic) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
            failures.push(format!("{family}: {}", panic_message(panic)));
        }
    };

    for family in ["KMeans.default", "KMeans.cosine-random", "KMeans.weighted"] {
        check(family, &|| {
            let (estimator_entry, model_entry) = (estimator(family), model(family));
            let (server, session) = ReplayServer::start(&[&estimator_entry, &model_entry]);
            let loaded = KMeans::load(&session, PATH).unwrap();
            check_loaded_estimator(&server, &session, &estimator_entry, &loaded, family);
            let loaded = KMeansModel::load(&session, PATH).unwrap();
            check_loaded_model(
                &server,
                &model_entry,
                &loaded,
                &|key| kmeans_model_member(&loaded, key),
                family,
            );
        });
    }

    for family in [
        "BisectingKMeans.default",
        "BisectingKMeans.min-divisible-2",
        "BisectingKMeans.cosine",
    ] {
        check(family, &|| {
            let (estimator_entry, model_entry) = (estimator(family), model(family));
            let (server, session) = ReplayServer::start(&[&estimator_entry, &model_entry]);
            let loaded = BisectingKMeans::load(&session, PATH).unwrap();
            check_loaded_estimator(&server, &session, &estimator_entry, &loaded, family);
            let loaded = BisectingKMeansModel::load(&session, PATH).unwrap();
            let dataset = session.range(1).unwrap();
            check_loaded_model(
                &server,
                &model_entry,
                &loaded,
                &|key| bisecting_model_member(&loaded, key, &dataset),
                family,
            );
        });
    }

    for family in ["GaussianMixture.default", "GaussianMixture.weighted"] {
        check(family, &|| {
            let (estimator_entry, model_entry) = (estimator(family), model(family));
            let (server, session) = ReplayServer::start(&[&estimator_entry, &model_entry]);
            let loaded = GaussianMixture::load(&session, PATH).unwrap();
            check_loaded_estimator(&server, &session, &estimator_entry, &loaded, family);
            let loaded = GaussianMixtureModel::load(&session, PATH).unwrap();
            check_loaded_model(
                &server,
                &model_entry,
                &loaded,
                &|key| gaussian_mixture_model_member(&loaded, key),
                family,
            );
        });
    }

    check("LDA.em", &|| {
        let family = "LDA.em";
        let (estimator_entry, model_entry) = (estimator(family), model(family));
        let (server, session) = ReplayServer::start(&[&estimator_entry, &model_entry]);
        let loaded = LDA::load(&session, PATH).unwrap();
        check_loaded_estimator(&server, &session, &estimator_entry, &loaded, family);
        let loaded = DistributedLDAModel::load(&session, PATH).unwrap();
        let dataset = session.range(1).unwrap();
        check_loaded_model(
            &server,
            &model_entry,
            &loaded,
            &|key| lda_member(&loaded, &dataset, key).or_else(|| distributed_member(&loaded, key)),
            family,
        );
    });

    check("LDA.online", &|| {
        let family = "LDA.online";
        let (estimator_entry, model_entry) = (estimator(family), model(family));
        let (server, session) = ReplayServer::start(&[&estimator_entry, &model_entry]);
        let loaded = LDA::load(&session, PATH).unwrap();
        check_loaded_estimator(&server, &session, &estimator_entry, &loaded, family);
        let loaded = LocalLDAModel::load(&session, PATH).unwrap();
        let dataset = session.range(1).unwrap();
        check_loaded_model(
            &server,
            &model_entry,
            &loaded,
            &|key| lda_member(&loaded, &dataset, key),
            family,
        );
    });

    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The message of a panic, as the test reports it.
fn panic_message(panic: Box<dyn std::any::Any + Send>) -> String {
    match panic.downcast::<String>() {
        Ok(text) => *text,
        Err(panic) => panic
            .downcast::<&str>()
            .map_or_else(|_| "a panic".to_string(), |text| text.to_string()),
    }
}

/// PowerIterationClustering fits no model: its estimator is saved as the server's
/// `PowerIterationClusteringWrapper` transformer, of its uid, and loaded back. Its `assignClusters`
/// is captured by another golden script, and its members are not in this file.
#[test]
fn power_iteration_clustering_persistence_matches_official_pyspark() {
    let persistence = golden(PERSISTENCE)["persistence"].clone();
    let entry = &persistence["PowerIterationClustering.estimator"];
    let (server, session) = ReplayServer::start(&[entry]);
    let pic = PowerIterationClustering::load(&session, PATH).unwrap();
    assert_commands(&server, &entry["load"], "PowerIterationClustering");
    assert_eq!(pic.uid(), result_uid(&entry["load"]));
    assert_eq!(pic.params().to_proto(), saved_params(&entry["save"]));
    pic.write()
        .session(&session)
        .overwrite()
        .save(PATH)
        .unwrap();
    assert_commands(&server, &entry["save"], "PowerIterationClustering");
}

#[test]
fn a_request_that_differs_from_its_golden_fails_the_check() {
    let models = golden(MODELS);
    let centers = &models["models"]["KMeans.default"]["members"]["clusterCenters()"];
    let (server, session) = ReplayServer::start(&[centers]);

    // The golden has the fetch of the centers: a fetch of `numFeatures` is refused by the replay,
    // and the commands that were sent do not equal the golden's.
    let model = MlTarget::unowned(&session, "<id0>");
    assert!(fetch(&model, "numFeatures", vec![]).is_err());
    let check = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_commands(&server, centers, "clusterCenters")
    }));
    assert!(check.is_err(), "the differing request passed the check");
}
