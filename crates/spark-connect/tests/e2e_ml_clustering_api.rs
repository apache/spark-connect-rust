// SPDX-License-Identifier: Apache-2.0
//! The typed models of the clustering estimators: their members, their summaries and their
//! persistence, as PySpark 4.3.0 reads them over Spark Connect. Set SPARK_REMOTE to run against a
//! live Spark Connect server.
mod common;

use common::live_session;
use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::ml::{
    BisectingKMeans, BisectingKMeansModel, ClusteringSummary, DistributedLDAModel, Estimator,
    GaussianMixture, GaussianMixtureModel, KMeans, KMeansModel, LDAModelApi, LocalLDAModel,
    MlMatrix, MlVector, Model, PowerIterationClustering, LDA,
};
use spark_connect::{Column, DataFrame, SparkSession};

/// Two blobs of four points each, around (1, 1) and (8, 8).
fn points(s: &SparkSession) -> DataFrame {
    vectors(
        s,
        "SELECT array(CAST(x AS DOUBLE), CAST(y AS DOUBLE)) AS xs FROM VALUES \
         (1.0, 1.0), (1.4, 1.1), (0.8, 1.3), (1.2, 0.7), \
         (8.0, 8.0), (8.3, 7.6), (7.7, 8.4), (8.2, 8.1) AS t(x, y)",
    )
}

/// Two topics of three terms each.
fn documents(s: &SparkSession) -> DataFrame {
    vectors(
        s,
        "SELECT id, array(CAST(a AS DOUBLE), CAST(b AS DOUBLE), CAST(c AS DOUBLE), \
         CAST(d AS DOUBLE), CAST(e AS DOUBLE), CAST(f AS DOUBLE)) AS xs FROM VALUES \
         (0, 10, 8, 2, 0, 0, 0), (1, 8, 10, 1, 0, 0, 0), (2, 9, 7, 3, 0, 0, 0), \
         (3, 0, 0, 0, 10, 8, 2), (4, 0, 0, 0, 8, 10, 1), (5, 0, 0, 0, 9, 7, 3) \
         AS t(id, a, b, c, d, e, f)",
    )
}

fn vectors(s: &SparkSession, sql: &str) -> DataFrame {
    s.sql(sql).unwrap().with_column(
        "features",
        Column::new(Expression::function(
            "array_to_vector",
            vec![col("xs").expression().clone()],
        )),
    )
}

fn scratch_path(name: &str) -> String {
    std::env::temp_dir()
        .join(format!(
            "spark-connect-rust-ml-clustering-{}-{name}",
            std::process::id()
        ))
        .to_string_lossy()
        .into_owned()
}

fn values(vector: &MlVector) -> Vec<f64> {
    match vector {
        MlVector::Dense(values) => values.clone(),
        MlVector::Sparse { values, .. } => values.clone(),
    }
}

/// The checks of the members of a clustering summary of the eight points in two clusters.
fn check_clustering_summary(summary: &impl ClusteringSummary) {
    assert_eq!(summary.k().unwrap(), 2);
    let sizes = summary.cluster_sizes().unwrap();
    assert_eq!(sizes.iter().sum::<i64>(), 8);
    assert_eq!(sizes.len(), 2);
    assert!(summary.num_iter().unwrap() >= 1);
    assert_eq!(summary.features_col().unwrap(), "features");
    assert_eq!(summary.prediction_col().unwrap(), "prediction");
    assert_eq!(summary.predictions().unwrap().count().unwrap(), 8);
    assert_eq!(summary.cluster().unwrap().count().unwrap(), 8);
}

#[test]
fn kmeans_models_have_the_members_and_summaries_of_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let train = points(&s);
    let mut estimator = KMeans::new().set_k(2).set_seed(1);
    let model = estimator.fit(&train).unwrap();
    let mut centers = model.cluster_centers().unwrap();
    centers.sort_by(|a, b| a[0].total_cmp(&b[0]));
    assert_eq!(centers.len(), 2);
    assert!((centers[0][0] - 1.1).abs() < 1e-9 && (centers[1][1] - 8.025).abs() < 1e-9);
    assert_eq!(model.num_features().unwrap(), 2);
    let near_first = model.predict(&MlVector::Dense(vec![1.0, 1.0])).unwrap();
    let near_second = model.predict(&MlVector::Dense(vec![8.0, 8.0])).unwrap();
    assert_ne!(near_first, near_second);

    let summary = model.summary().unwrap();
    check_clustering_summary(&summary);
    assert!(summary.training_cost().unwrap() > 0.0);

    let path = scratch_path("kmeans-model");
    model.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = KMeansModel::load(&s, &path).unwrap();
    assert_eq!(
        loaded.cluster_centers().unwrap(),
        model.cluster_centers().unwrap()
    );
    assert!(!loaded.has_summary() && model.has_summary());

    let path = scratch_path("kmeans");
    Estimator::write(&estimator)
        .session(&s)
        .overwrite()
        .save(&path)
        .unwrap();
    assert_eq!(KMeans::load(&s, &path).unwrap().k(), 2);
    s.stop().unwrap();
}

#[test]
fn bisecting_kmeans_and_gaussian_mixture_models_have_the_members_of_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let train = points(&s);
    let bisecting = BisectingKMeans::new()
        .set_k(2)
        .set_seed(1)
        .fit(&train)
        .unwrap();
    assert_eq!(bisecting.cluster_centers().unwrap().len(), 2);
    assert_eq!(bisecting.num_features().unwrap(), 2);
    let summary = bisecting.summary().unwrap();
    check_clustering_summary(&summary);
    let cost = bisecting.compute_cost(&train).unwrap();
    assert!((cost - summary.training_cost().unwrap()).abs() < 1e-9);
    let path = scratch_path("bisecting-model");
    bisecting.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = BisectingKMeansModel::load(&s, &path).unwrap();
    assert!((loaded.compute_cost(&train).unwrap() - cost).abs() < 1e-9);

    let mixture = GaussianMixture::new()
        .set_k(2)
        .set_seed(1)
        .fit(&train)
        .unwrap();
    let weights = mixture.weights().unwrap();
    assert_eq!(weights.len(), 2);
    assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    // The components, of the collected rows of gaussiansDF.
    let gaussians = mixture.gaussians().unwrap();
    assert_eq!(gaussians.len(), 2);
    assert_eq!(mixture.gaussians_df().unwrap().count().unwrap(), 2);
    for gaussian in &gaussians {
        assert_eq!(values(&gaussian.mean).len(), 2);
        let MlMatrix::Dense {
            num_rows, num_cols, ..
        } = &gaussian.cov
        else {
            panic!("dense covariance")
        };
        assert_eq!((*num_rows, *num_cols), (2, 2));
    }
    let point = MlVector::Dense(vec![8.0, 8.0]);
    assert!([0, 1].contains(&mixture.predict(&point).unwrap()));
    let probability = values(&mixture.predict_probability(&point).unwrap());
    assert!((probability.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    let summary = mixture.summary().unwrap();
    check_clustering_summary(&summary);
    assert!(summary.log_likelihood().unwrap().is_finite());
    assert_eq!(summary.probability_col().unwrap(), "probability");
    assert_eq!(summary.probability().unwrap().count().unwrap(), 8);
    let path = scratch_path("mixture-model");
    mixture.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = GaussianMixtureModel::load(&s, &path).unwrap();
    assert_eq!(loaded.weights().unwrap(), weights);
    s.stop().unwrap();
}

#[test]
fn lda_models_are_local_or_distributed_by_their_optimizer_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let docs = documents(&s);
    let online = LDA::new()
        .set_k(2)
        .set_max_iter(5)
        .set_seed(1)
        .fit(&docs)
        .unwrap();
    assert!(!online.is_distributed().unwrap());
    assert_eq!(online.vocab_size().unwrap(), 6);
    let MlMatrix::Dense {
        num_rows, num_cols, ..
    } = online.topics_matrix().unwrap()
    else {
        panic!("dense topics")
    };
    assert_eq!((num_rows, num_cols), (6, 2));
    assert_eq!(online.describe_topics(3).unwrap().count().unwrap(), 2);
    assert!(online.log_likelihood(&docs).unwrap().is_finite());
    assert!(online.log_perplexity(&docs).unwrap().is_finite());
    assert_eq!(
        values(&online.estimated_doc_concentration().unwrap()).len(),
        2
    );
    assert!(online.clone().into_distributed().is_none());
    let local = online.into_local().expect("an online LDA model is local");
    let path = scratch_path("local-lda-model");
    local.write().unwrap().overwrite().save(&path).unwrap();
    let loaded = LocalLDAModel::load(&s, &path).unwrap();
    assert_eq!(loaded.vocab_size().unwrap(), 6);

    let em = LDA::new()
        .set_k(2)
        .set_max_iter(5)
        .set_seed(1)
        .set_optimizer("em")
        .fit(&docs)
        .unwrap();
    assert!(em.clone().into_local().is_none());
    let distributed = em
        .into_distributed()
        .expect("an em LDA model is distributed");
    assert!(distributed.is_distributed().unwrap());
    assert!(distributed.training_log_likelihood().unwrap().is_finite());
    assert!(distributed.log_prior().unwrap().is_finite());
    assert!(distributed.get_checkpoint_files().unwrap().is_empty());
    let local = distributed.to_local().unwrap();
    assert!(!local.is_distributed().unwrap());
    assert_eq!(local.describe_topics(2).unwrap().count().unwrap(), 2);
    let path = scratch_path("distributed-lda-model");
    distributed
        .write()
        .unwrap()
        .overwrite()
        .save(&path)
        .unwrap();
    let loaded = DistributedLDAModel::load(&s, &path).unwrap();
    assert!(loaded.is_distributed().unwrap());
    s.stop().unwrap();
}

#[test]
fn power_iteration_clustering_saves_and_loads_as_pyspark() {
    let Some(s) = live_session() else {
        return;
    };
    let algorithm = PowerIterationClustering::new().set_k(3).set_max_iter(7);
    assert!(algorithm.uid().starts_with("PowerIterationClustering_"));
    let path = scratch_path("pic");
    algorithm
        .write()
        .session(&s)
        .overwrite()
        .save(&path)
        .unwrap();
    let loaded = PowerIterationClustering::load(&s, &path).unwrap();
    assert_eq!(loaded.uid(), algorithm.uid());
    assert_eq!(loaded.k(), 3);
    assert_eq!(loaded.max_iter(), 7);
    s.stop().unwrap();
}
