# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""SPARK-60659: exact interval inputs and live Spark round trips."""

from datetime import timedelta
import os
import unittest

from pyspark.sql import Row, SparkSession, types as T
from pyspark.errors import PySparkTypeError

from _mock_connect_server import mock_connect_session, parse_message

try:
    import grpc
except ImportError:
    grpc = None


# The bounds that PySpark takes: a timedelta of whole days, whose microseconds must fit in a
# signed 64-bit integer on their own, cannot go below -106751991 days.
LOWEST, HIGHEST = -106751991 * 86400000000, 2**63 - 1

INFERENCE_CONFS = {
    "spark.sql.pyspark.inferNestedDictAsStruct.enabled": "false",
    "spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled": "false",
    "spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled": "false",
    "spark.sql.timestampType": "TIMESTAMP_LTZ",
}


class IntervalInputConversionTests(unittest.TestCase):
    def test_overflow_is_checked_recursively_before_rpc(self):
        spark = SparkSession.builder.remote("sc://127.0.0.1:1").create()
        self.addCleanup(spark.stop)
        interval = T.DayTimeIntervalType()
        for micros in (-(2**63) - 1, 2**63, -(2**63) - 86400000000):
            value = timedelta(microseconds=micros)
            for data, data_type in (
                ([(value,)], interval),
                ([([value],)], T.ArrayType(interval)),
                ([({"k": value},)], T.MapType(T.StringType(), interval)),
                ([(Row(leaf=value),)], T.StructType([T.StructField("leaf", interval)])),
            ):
                with self.subTest(micros=micros, data=data):
                    schema = T.StructType([T.StructField("interval", data_type)])
                    with self.assertRaises(OverflowError):
                        spark.createDataFrame(data, schema)
            with self.assertRaises(OverflowError):
                spark.createDataFrame([(value,)])
            # A Python Row stores objects unchanged; native conversion starts here.
            with self.assertRaises(OverflowError):
                spark.createDataFrame([Row(interval=value)])
            with self.assertRaises(OverflowError):
                spark.createDataFrame([value], T.DayTimeIntervalType())
        for value in (timedelta.min, timedelta.max):
            with self.assertRaises(OverflowError):
                spark.createDataFrame([(value,)])

    def test_inference_rejects_mixed_interval_and_numeric_values(self):
        spark = SparkSession.builder.remote("sc://127.0.0.1:1").create()
        self.addCleanup(spark.stop)
        delta = timedelta(microseconds=-1)
        with self.assertRaises(PySparkTypeError) as error:
            spark.createDataFrame([(delta,), (1,)])
        self.assertEqual(error.exception.getCondition(), "CANNOT_MERGE_TYPE")
        self.assertEqual(error.exception.getMessageParameters(), {
            "data_type1": "DayTimeIntervalType", "data_type2": "LongType"})

    @unittest.skipIf(grpc is None, "grpcio is required")
    def test_nested_inference_reads_config_but_does_not_execute(self):
        delta = timedelta(microseconds=-1)
        with mock_connect_session() as (spark, service):
            service.confs = INFERENCE_CONFS
            for data in ([([delta, 1],)], [({"a": delta, "b": 1},)]):
                with self.subTest(data=data):
                    with self.assertRaises(PySparkTypeError) as error:
                        spark.createDataFrame(data)
                    self.assertEqual(error.exception.getCondition(), "CANNOT_INFER_TYPE_FOR_FIELD")
                    self.assertEqual(error.exception.getMessageParameters(), {"field_name": "_1"})
            spark.createDataFrame([([],), ([None, delta],)])
            spark.createDataFrame([({},), ({"k": delta},)])
            for data in ([([None, timedelta.max],)], [({"k": timedelta.min},)]):
                with self.assertRaises(OverflowError):
                    spark.createDataFrame(data)
            self.assertEqual(service.execute_calls, 0)
            self.assertEqual(set(service.read_confs), set(INFERENCE_CONFS))

    def test_bounds_and_null_first_inputs_build_without_rpc(self):
        spark = SparkSession.builder.remote("sc://127.0.0.1:1").create()
        self.addCleanup(spark.stop)
        for schema in (None, ["interval"], T.StructType([
                T.StructField("interval", T.DayTimeIntervalType())])):
            spark.createDataFrame([(None,), (timedelta(microseconds=LOWEST),),
                                   (timedelta(microseconds=HIGHEST),)], schema)
        interval = T.DayTimeIntervalType()
        arrays = T.StructType([T.StructField("a", T.ArrayType(interval))])
        maps = T.StructType([T.StructField("m", T.MapType(T.StringType(), interval))])
        spark.createDataFrame([([],), ([None, timedelta(microseconds=-1)],)], arrays)
        spark.createDataFrame([(Row(x=None),), (Row(x=timedelta(microseconds=1)),)])
        spark.createDataFrame([({},), ({"k": timedelta(microseconds=-1)},)], maps)
        # As PySpark, an int for an interval field is taken as microseconds.
        spark.createDataFrame([(1,)], T.StructType([T.StructField("interval", interval)]))

    @unittest.skipIf(grpc is None, "grpcio is required")
    def test_pandas_object_intervals_encode_exact_microseconds(self):
        try:
            import pandas as pd
            import pyarrow as pa
        except ImportError:
            self.skipTest("pandas and pyarrow are required")
        values = [None, LOWEST, LOWEST + 1, -1, 0, 1, HIGHEST]
        data = pd.DataFrame({"interval": pd.Series([
            None if value is None else timedelta(microseconds=value) for value in values
        ], dtype=object)})
        with mock_connect_session() as (spark, service):
            spark.createDataFrame(data).collect()
            # ExecutePlanRequest.plan=3, Plan.root=1, Relation.to_df=18,
            # ToDF.input=1, Relation.local_relation=11, LocalRelation.data=1.
            message = parse_message(service.execute_requests[0])
            for field in (3, 1, 18, 1, 11):
                message = parse_message(message[field][0])
            table = pa.ipc.open_stream(message[1][0]).read_all()
            self.assertEqual(table.column(0).type, pa.duration("us"))
            self.assertEqual(table.column(0).cast(pa.int64()).to_pylist(), values)
            for value in (timedelta.min, timedelta.max):
                overflow = pd.DataFrame({"interval": pd.Series([value], dtype=object)})
                # As PySpark's Arrow conversion: ArrowInvalid, a ValueError, of the duration.
                with self.assertRaises(ValueError):
                    spark.createDataFrame(overflow)
            self.assertEqual(service.execute_calls, 1)


@unittest.skipUnless(os.environ.get("SPARK_REMOTE"), "requires a real Spark Connect server")
class IntervalInputLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.remote(os.environ["SPARK_REMOTE"]).create()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_scalar_inferred_explicit_and_collected_inputs(self):
        values = [None, LOWEST, LOWEST + 1, -172800000001, -1000001, -1,
                  0, 1, 1000001, 172800000001, HIGHEST - 1, HIGHEST]
        data = [(None if v is None else timedelta(microseconds=v),) for v in values]
        for schema in (None, ["interval"], T.StructType([
                T.StructField("interval", T.DayTimeIntervalType())])):
            with self.subTest(schema=schema):
                frame = self.spark.createDataFrame(data, schema)
                rows = frame.collect()
                self.assertIsInstance(frame.schema.fields[0].dataType, T.DayTimeIntervalType)
                self.assertEqual([r[0] for r in rows], [r[0] for r in data])
                again = self.spark.createDataFrame(rows, frame.schema)
                self.assertEqual([r[0] for r in again.collect()], [r[0] for r in data])
        # Check an independently computed Spark SQL interval as well.
        frame = self.spark.sql("SELECT INTERVAL '-2 00:00:00.000001' DAY TO SECOND AS interval")
        rows = frame.collect()
        self.assertEqual(rows[0][0], timedelta(microseconds=-172800000001))
        self.assertEqual(self.spark.createDataFrame(rows, frame.schema).collect(), rows)

    def test_explicit_atomic_null_and_empty_inputs(self):
        interval = T.DayTimeIntervalType()
        data = [None, timedelta(microseconds=-1), timedelta(microseconds=1)]
        self.assertEqual([r[0] for r in self.spark.createDataFrame(data, interval).collect()], data)
        nulls = self.spark.createDataFrame([None, None], interval).collect()
        self.assertEqual([r[0] for r in nulls], [None, None])
        self.assertEqual(self.spark.createDataFrame([], interval).collect(), [])
        # As PySpark, an int for an interval field is taken as microseconds.
        schema = T.StructType([T.StructField("interval", interval)])
        self.assertEqual(self.spark.createDataFrame([(1,)], schema).collect(),
                         [Row(interval=timedelta(microseconds=1))])

    def test_nested_null_first_structs_lists_and_maps(self):
        delta = timedelta(microseconds=-172800000001)
        data = [(None, None, None), (Row(leaf=None), [], {}),
                (Row(leaf=delta), [None, delta], {"k": delta, "null": None})]
        interval = T.DayTimeIntervalType()
        explicit = T.StructType([
            T.StructField("s", T.StructType([T.StructField("leaf", interval)])),
            T.StructField("a", T.ArrayType(interval)),
            T.StructField("m", T.MapType(T.StringType(), interval)),
        ])
        for schema in (["s", "a", "m"], explicit):
            with self.subTest(schema=schema):
                frame = self.spark.createDataFrame(data, schema)
                rows = frame.collect()
                self.assertEqual(rows[0].asDict(), {"s": None, "a": None, "m": None})
                self.assertIsNone(rows[1][0][0])
                self.assertEqual(rows[1][1], [])
                self.assertEqual(rows[1][2], {})
                self.assertEqual(rows[2][0][0], delta)
                self.assertEqual(rows[2][1], [None, delta])
                self.assertEqual(rows[2][2], {"k": delta, "null": None})
                self.assertEqual(self.spark.createDataFrame(rows, frame.schema).collect(), rows)


if __name__ == "__main__":
    unittest.main()
