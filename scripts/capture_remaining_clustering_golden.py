#!/usr/bin/env python3
"""Capture LDA and PowerIterationClustering goldens with official PySpark 4.3.0."""

import argparse
import base64
import inspect
import json
from pathlib import Path

import pyspark
from pyspark.ml import clustering
from pyspark.ml.connect.serialize import serialize_ml_params
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

LDA_VALUES = dict(
    featuresCol="input",
    maxIter=10,
    seed=42,
    checkpointInterval=-1,
    k=2,
    optimizer="online",
    learningOffset=16.0,
    learningDecay=0.7,
    subsamplingRate=1.0,
    optimizeDocConcentration=False,
    docConcentration=[0.5, 0.5],
    topicConcentration=0.5,
    topicDistributionCol="topics",
    keepLastCheckpoint=False,
)
PIC_VALUES = dict(
    k=3, maxIter=40, initMode="degree", srcCol="source", dstCol="target", weightCol="affinity"
)
DOCUMENTS = [
    [10.0, 8.0, 2.0, 0.0, 0.0, 0.0],
    [8.0, 10.0, 1.0, 0.0, 0.0, 0.0],
    [9.0, 7.0, 3.0, 0.0, 0.0, 0.0],
    [12.0, 6.0, 2.0, 0.0, 0.0, 0.0],
    [7.0, 9.0, 3.0, 0.0, 0.0, 0.0],
    [11.0, 8.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 10.0, 8.0, 2.0],
    [0.0, 0.0, 0.0, 8.0, 10.0, 1.0],
    [0.0, 0.0, 0.0, 9.0, 7.0, 3.0],
    [0.0, 0.0, 0.0, 12.0, 6.0, 2.0],
    [0.0, 0.0, 0.0, 7.0, 9.0, 3.0],
    [0.0, 0.0, 0.0, 11.0, 8.0, 1.0],
]
HELD_OUT = [
    [8.0, 6.0, 1.0, 0.0, 0.0, 0.0],
    [10.0, 9.0, 2.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 8.0, 6.0, 1.0],
    [0.0, 0.0, 0.0, 10.0, 9.0, 2.0],
    [0.0] * 6,
]
EDGES = [(i, j, 0.5 + j * 0.2) for i in range(1, 6) for j in range(i)]


def encoded(message):
    return base64.b64encode(message.SerializeToString(deterministic=True)).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/remaining_clustering.json"))
    args = parser.parse_args()
    assert pyspark.__version__ == "4.3.0", pyspark.__version__
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    result = {
        "spark_version": pyspark.__version__,
        "defaults": {},
        "params": [],
        "pic_fetch": [],
        "documents": DOCUMENTS,
        "held_out": HELD_OUT,
        "edges": EDGES,
        "lda": [],
        "pic": [],
    }
    try:
        for name, values in [("LDA", LDA_VALUES), ("PowerIterationClustering", PIC_VALUES)]:
            cls = getattr(clustering, name)
            result["defaults"][name] = {
                k: v.default for k, v in inspect.signature(cls).parameters.items()
            }
            variants = [{}, values]
            if name == "LDA":
                variants += [
                    dict(values, seed=seed)
                    for seed in [2**31 - 1, 2**31, -(2**31) - 1, 2**63 - 1, -(2**63)]
                ]
            else:
                variants += [{"weightCol": None}, {"k": 2**31}, {"k": None}]
            for variant in variants:
                instance = cls(**variant)
                result["params"].append(
                    {
                        "class": name,
                        "values": variant,
                        "params_b64": encoded(serialize_ml_params(instance, spark.client)),
                    }
                )
                if name == "PowerIterationClustering":
                    fetch = (
                        instance.assignClusters(spark.range(1))
                        ._plan.plan(spark.client)
                        .ml_relation.fetch
                    )
                    fetch.methods[0].args[0].input.Clear()
                    result["pic_fetch"].append({"values": variant, "fetch_b64": encoded(fetch)})
        train = spark.createDataFrame(
            [(i, Vectors.dense(x)) for i, x in enumerate(DOCUMENTS)], ["id", "input"]
        ).coalesce(1)
        held = spark.createDataFrame(
            [(i, Vectors.dense(x)) for i, x in enumerate(HELD_OUT)], ["id", "input"]
        )
        for optimizer in ["online", "em"]:
            values = dict(LDA_VALUES, optimizer=optimizer)
            if optimizer == "em":
                values.update(docConcentration=[2.0, 2.0], topicConcentration=2.0)
            model = clustering.LDA(**values).fit(train)
            rows = (
                model.transform(held)
                .orderBy("id")
                .select(vector_to_array("topics").alias("topics"))
                .collect()
            )
            result["lda"].append({"values": values, "topics": [r.topics for r in rows]})
        graph = spark.createDataFrame(EDGES, ["source", "target", "affinity"]).repartition(1)
        for init_mode in ["random", "degree"]:
            for weighted in [False, True]:
                values = dict(PIC_VALUES, k=2, initMode=init_mode)
                if not weighted:
                    values.pop("weightCol")
                rows = (
                    clustering.PowerIterationClustering(**values)
                    .assignClusters(graph)
                    .orderBy("id")
                    .collect()
                )
                result["pic"].append(
                    {"values": values, "assignments": [[r.id, r.cluster] for r in rows]}
                )
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
