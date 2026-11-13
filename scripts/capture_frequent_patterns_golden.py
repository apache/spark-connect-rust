#!/usr/bin/env python3
"""Capture frequent-pattern mining fixtures with official PySpark 4.3.0."""

import argparse
import base64
import inspect
import json
from pathlib import Path

import pyspark
from pyspark.ml import fpm
from pyspark.ml.connect.serialize import serialize_ml_params
from pyspark.sql import SparkSession

TRANSACTIONS = "SELECT * FROM VALUES (0,array('a','b')), (1,array('b','c')), (2,array('a','b','c')), (3,array('a','c')), (4,array('a','b')), (5,array('b')) AS t(id,basket)"
HELD_OUT = "SELECT * FROM VALUES (10,array('a')), (11,array('b')), (12,array('a','b')), (13,array('z')), (14,CAST(array() AS ARRAY<STRING>)) AS t(id,basket)"
NUMERIC = "SELECT * FROM VALUES (0,array(1,2)), (1,array(2,3)), (2,array(1,2,3)), (3,array(1,3)), (4,array(1,2)), (5,array(2)) AS t(id,basket)"
SEQUENCES = "SELECT * FROM VALUES (0,array(array(1,2),array(3))), (1,array(array(1),array(3,2),array(1,2))), (2,array(array(1,2),array(5))), (3,array(array(6))) AS t(id,events)"


def rows(frame, kind):
    values = []
    for row in frame.collect():
        row = list(row)
        if kind == "predictions":
            row[-1] = sorted(row[-1])
        elif kind == "itemsets":
            row[0] = sorted(row[0])
        elif kind == "rules":
            row[0], row[1] = sorted(row[0]), sorted(row[1])
        else:
            row[0] = [sorted(v) for v in row[0]]
        values.append(row)
    return sorted(values, key=lambda v: json.dumps(v))


def encoded(message):
    return base64.b64encode(message.SerializeToString(deterministic=True)).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default="sc://localhost:15002")
    parser.add_argument("--out", type=Path, default=Path("tests/golden/frequent_patterns.json"))
    args = parser.parse_args()
    assert pyspark.__version__ == "4.3.0"
    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    result = {
        "spark_version": pyspark.__version__,
        "defaults": {},
        "params": [],
        "fpgrowth": [],
        "prefixspan": [],
        "sequences": SEQUENCES,
    }
    try:
        samples = {
            "FPGrowth": dict(
                minSupport=0.4,
                minConfidence=0.7,
                itemsCol="basket",
                predictionCol="suggested",
                numPartitions=2,
            ),
            "PrefixSpan": dict(
                minSupport=0.5, maxPatternLength=5, maxLocalProjDBSize=2**31, sequenceCol="events"
            ),
        }
        for name, values in samples.items():
            cls = getattr(fpm, name)
            result["defaults"][name] = {
                k: v.default for k, v in inspect.signature(cls).parameters.items()
            }
            variants = [{}, values, {k: None for k in values}]
            if name == "PrefixSpan":
                variants += [
                    dict(values, maxLocalProjDBSize=size) for size in [2**31 - 1, 2**63 - 1]
                ]
            for values in variants:
                instance = cls(**values)
                entry = {
                    "name": name,
                    "values": values,
                    "params_b64": encoded(serialize_ml_params(instance, spark.client)),
                }
                if name == "PrefixSpan":
                    fetch = (
                        instance.findFrequentSequentialPatterns(spark.range(1))
                        ._plan.plan(spark.client)
                        .ml_relation.fetch
                    )
                    entry["helper"] = fetch.methods[0].method
                    entry["args_b64"] = [encoded(arg.param) for arg in fetch.methods[0].args[1:]]
                result["params"].append(entry)
        for key, sql, held, support, confidence, partitions in [
            ("strings", TRANSACTIONS, HELD_OUT, 0.3, 0.5, 2),
            ("high-confidence", TRANSACTIONS, HELD_OUT, 0.5, 0.9, 1),
            ("integers", NUMERIC, NUMERIC, 0.3, 0.6, 2),
        ]:
            values = dict(
                minSupport=support,
                minConfidence=confidence,
                itemsCol="basket",
                predictionCol="suggested",
                numPartitions=partitions,
            )
            model = fpm.FPGrowth(**values).fit(spark.sql(sql))
            entry = dict(id=key, sql=sql, held=held, values=values)
            for name, frame, kind in [
                ("itemsets", model.freqItemsets, "itemsets"),
                ("rules", model.associationRules, "rules"),
                ("predictions", model.transform(spark.sql(held)), "predictions"),
            ]:
                entry[name] = {"rows": rows(frame, kind), "schema": frame.schema.jsonValue()}
            changed = (
                model.setMinConfidence(0.95).setPredictionCol("changed").transform(spark.sql(held))
            )
            entry["changed_predictions"] = {
                "rows": rows(changed, "predictions"),
                "schema": changed.schema.jsonValue(),
            }
            result["fpgrowth"].append(entry)
        for length, size in [(5, 32000000), (1, 1024)]:
            values = dict(
                minSupport=0.5,
                maxPatternLength=length,
                maxLocalProjDBSize=size,
                sequenceCol="events",
            )
            frame = fpm.PrefixSpan(**values).findFrequentSequentialPatterns(spark.sql(SEQUENCES))
            result["prefixspan"].append(
                dict(values=values, rows=rows(frame, "sequences"), schema=frame.schema.jsonValue())
            )
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print("Captured FPGrowth itemsets, rules and predictions, and PrefixSpan results.")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
