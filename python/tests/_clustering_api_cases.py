# SPDX-License-Identifier: Apache-2.0
"""The cases of the clustering estimators that have no full API golden of their own:
BisectingKMeans, GaussianMixture, LDA and PowerIterationClustering, with their models and
summaries, over a live server.

`scripts/capture_clustering_api_cases_golden.py` runs them on official PySpark for
tests/golden/clustering_api_cases.json, and test_clustering_api_e2e.py runs them on the native
classes and compares. Official PySpark runs the tests too, as the reference (`_native_classes`).
The recording and the shared model and persistence cases are those of `_ml_api_cases`.
"""

from _ml_api_cases import Recorder, model_cases, persistence_cases
from _native_classes import native_class

# The classes of each case set: (module of the native class, name, module of PySpark's).
CLASSES = {
    "BisectingKMeans": [
        ("pyspark.ml.connect.clustering", "BisectingKMeans", "pyspark.ml.clustering"),
        ("pyspark.ml.connect.clustering", "BisectingKMeansModel", "pyspark.ml.clustering"),
    ],
    "GaussianMixture": [
        ("pyspark.ml.connect.clustering", "GaussianMixture", "pyspark.ml.clustering"),
        ("pyspark.ml.connect.clustering", "GaussianMixtureModel", "pyspark.ml.clustering"),
    ],
    "LDA": [
        ("pyspark.ml.connect.clustering", "LDA", "pyspark.ml.clustering"),
        ("pyspark.ml.connect.clustering", "LocalLDAModel", "pyspark.ml.clustering"),
        ("pyspark.ml.connect.clustering", "DistributedLDAModel", "pyspark.ml.clustering"),
    ],
    "PowerIterationClustering": [
        ("pyspark.ml.connect.clustering", "PowerIterationClustering", "pyspark.ml.clustering"),
    ],
}

POINTS = [
    [1.0, 1.0], [1.5, 1.2], [1.1, 1.6], [1.4, 1.4], [6.0, 6.0], [6.5, 5.8],
    [5.9, 6.6], [6.2, 6.3], [10.0, 1.5], [10.4, 1.1], [9.8, 1.9],
]
DOCUMENTS = [
    [1.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 3.0, 0.0, 0.0, 1.0],
    [2.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2.0, 2.0, 1.0],
]
EDGES = [(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0), (3, 4, 1.0), (4, 5, 1.0), (5, 3, 1.0)]


def classes(name):
    """The native classes of the case set `name`."""
    return [native_class(*spec) for spec in CLASSES[name]]


def _vectors(spark, rows, columns):
    from pyspark.ml.linalg import Vectors

    data = [tuple(Vectors.dense(v) if isinstance(v, list) else v for v in row) for row in rows]
    return spark.createDataFrame(data, columns).coalesce(1)


def _bisecting_kmeans_cases(rec, spark, base, found):
    estimator, model_class = found
    df = _vectors(spark, [(p,) for p in POINTS], ["features"])
    est = estimator(k=3, seed=5)
    model = est.fit(df)
    model_cases(rec, est, model, df)
    rec("clusterCenters", lambda: sorted([round(x, 9) for x in c] for c in model.clusterCenters()))
    rec("clusterCenters_type", lambda: type(model.clusterCenters()[0]).__name__)
    rec("computeCost", lambda: round(model.computeCost(df), 9))
    rec("numFeatures", lambda: model.numFeatures)
    rec("hasSummary", lambda: model.hasSummary)
    s = model.summary
    rec("summary_is_summary", lambda: model.summary is s)
    for a in ["k", "clusterSizes", "trainingCost", "numIter", "featuresCol", "predictionCol"]:
        rec("summary_" + a, lambda a=a: getattr(s, a))
    rec("summary_df_predictions", lambda: s.predictions.select("features"))
    rec("summary_df_cluster_count", lambda: s.cluster.count())
    persistence_cases(rec, est, model, model_class, df, base)


def _gaussian_mixture_cases(rec, spark, base, found):
    from pyspark.ml.linalg import Vectors

    estimator, model_class = found
    df = _vectors(spark, [(p,) for p in POINTS], ["features"])
    est = estimator(k=2, seed=7, maxIter=5)
    model = est.fit(df)
    model_cases(rec, est, model, df)
    rec("weights", lambda: [round(w, 9) for w in model.weights])
    rec("gaussians_type", lambda: [type(g).__name__ for g in model.gaussians])
    rec("gaussians_mean", lambda: [[round(x, 9) for x in g.mean] for g in model.gaussians])
    rec(
        "gaussians_cov",
        lambda: [[[round(x, 9) for x in r] for r in g.cov.toArray()] for g in model.gaussians],
    )
    rec("gaussiansDF_columns", lambda: model.gaussiansDF.columns)
    probe = Vectors.dense([6.1, 6.2])
    rec("predictProbability", lambda: [round(x, 9) for x in model.predictProbability(probe)])
    rec("numFeatures", lambda: model.numFeatures)
    rec("hasSummary", lambda: model.hasSummary)
    s = model.summary
    rec("summary_is_summary", lambda: model.summary is s)
    for a in ["k", "clusterSizes", "numIter", "featuresCol", "predictionCol", "probabilityCol"]:
        rec("summary_" + a, lambda a=a: getattr(s, a))
    rec("summary_logLikelihood", lambda: round(s.logLikelihood, 9))
    rec("summary_df_probability", lambda: s.probability.columns)
    rec("summary_df_cluster_count", lambda: s.cluster.count())
    persistence_cases(rec, est, model, model_class, df, base)


def _lda_cases(rec, spark, base, found):
    estimator, local_class, distributed_class = found
    docs = _vectors(spark, [(i, d) for i, d in enumerate(DOCUMENTS)], ["id", "features"])

    def lda(optimizer):
        return estimator(k=2, optimizer=optimizer, maxIter=3, seed=42, featuresCol="features")

    online = lda("online").fit(docs)
    rec("online_type", lambda: type(online).__name__)
    rec("online_isDistributed", lambda: online.isDistributed())
    rec("online_has_toLocal", lambda: hasattr(online, "toLocal"))
    rec("online_has_summary", lambda: hasattr(online, "summary"))
    rec("online_logLikelihood", lambda: type(online.logLikelihood(docs)).__name__)
    rec("online_logPerplexity", lambda: type(online.logPerplexity(docs)).__name__)
    rec("online_estimatedDocConcentration_len", lambda: len(online.estimatedDocConcentration()))
    rec("online_vocabSize", lambda: online.vocabSize())
    rec("online_describeTopics", lambda: online.describeTopics(3).select("termIndices"))
    rec("online_transform_columns", lambda: online.transform(docs).columns)
    persistence_cases(rec, lda("online"), online, local_class, docs, base)

    est = lda("em")
    em = est.fit(docs)
    rec("em_type", lambda: type(em).__name__)
    rec("em_isDistributed", lambda: em.isDistributed())
    rec("em_has_summary", lambda: hasattr(em, "summary"))
    rec("em_vocabSize", lambda: em.vocabSize())
    rec("em_topicsMatrix_shape", lambda: list(em.topicsMatrix().toArray().shape))
    rec("em_describeTopics_count", lambda: em.describeTopics(3).count())
    rec("em_logPrior_type", lambda: type(em.logPrior()).__name__)
    rec("em_trainingLogLikelihood_type", lambda: type(em.trainingLogLikelihood()).__name__)
    rec("em_getCheckpointFiles_type", lambda: type(em.getCheckpointFiles()).__name__)
    rec("em_toLocal_is_same", lambda: em.toLocal() is em.toLocal())
    local = em.toLocal()
    rec("toLocal_type", lambda: type(local).__name__)
    rec("toLocal_uid_is_fresh", lambda: local.uid != em.uid)
    rec("toLocal_isDistributed", lambda: local.isDistributed())
    rec("toLocal_isSet_k", lambda: local.isSet("k"))
    rec("toLocal_getK", lambda: local.getK())
    rec("toLocal_vocabSize", lambda: local.vocabSize())
    rec("em_transform_columns", lambda: em.transform(docs).columns)
    rec(
        "em_fitMultiple",
        lambda: sorted((i, m.getK()) for i, m in est.fitMultiple(docs, [{est.k: 2}, {est.k: 3}])),
    )
    persistence_cases(rec, est, em, distributed_class, docs, base)


def _power_iteration_cases(rec, spark, base, found):
    (estimator,) = found
    graph = spark.createDataFrame(EDGES, ["src", "dst", "weight"])
    pic = estimator(k=2, maxIter=5, initMode="degree")
    rec("assignClusters_columns", lambda: sorted(pic.assignClusters(graph).columns))
    rec("assignClusters_count", lambda: pic.assignClusters(graph).count())
    rec(
        "assignClusters_clusters",
        lambda: sorted(r.cluster for r in pic.assignClusters(graph).collect()),
    )
    # A dataset that is not a DataFrame: PySpark's helper call sends it as a literal, in a lazy
    # DataFrame, and the server rejects it when the DataFrame is collected; one that cannot be a
    # literal raises the error of its conversion before any request.
    for label, other in [("none", None), ("int", 5), ("str", "t"), ("list", [1, 2])]:
        rec("assignClusters_of_" + label, lambda other=other: pic.assignClusters(other).collect())
    rec("assignClusters_of_object", lambda: pic.assignClusters(object()))
    rec("params_getters", lambda: [pic.getK(), pic.getMaxIter(), pic.getInitMode()])
    rec("uid_prefix", lambda: pic.uid.split("_")[0])
    rec("save", lambda: pic.write().overwrite().save(base + "/pic"))
    loaded = type(pic).load(base + "/pic")
    rec("loaded_type", lambda: type(loaded).__name__)
    rec("loaded_uid_same", lambda: loaded.uid == pic.uid)
    rec("loaded_params", lambda: [loaded.getK(), loaded.getMaxIter(), loaded.getInitMode()])
    rec("loaded_assignClusters_count", lambda: loaded.assignClusters(graph).count())
    rec("loaded_isSet_initMode", lambda: loaded.isSet("initMode"))


def live_cases(name, spark, base):
    """The cases of the case set `name` against a server; `base` is a scratch directory of the
    server's file system."""
    rec = Recorder(base)
    found = classes(name)
    if name == "BisectingKMeans":
        _bisecting_kmeans_cases(rec, spark, base, found)
    elif name == "GaussianMixture":
        _gaussian_mixture_cases(rec, spark, base, found)
    elif name == "LDA":
        _lda_cases(rec, spark, base, found)
    else:
        _power_iteration_cases(rec, spark, base, found)
    return rec.out
