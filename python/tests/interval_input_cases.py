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


class IntervalInputConversionTests(unittest.TestCase):
    def test_overflow_is_checked_recursively_before_rpc(self):
        spark = SparkSession.builder.remote("sc://127.0.0.1:1").create()
        for micros in (-(2**63) - 1, 2**63, -(2**63) - 86400000000):
            value = timedelta(microseconds=micros)
            for data in ([(value,)], [([value],)], [({"k": value},)]):
                with self.subTest(micros=micros, data=data):
                    with self.assertRaisesRegex(OverflowError, "64-bit microseconds"):
                        spark.createDataFrame(data)
            with self.assertRaises(OverflowError):
                Row(interval=value)
            with self.assertRaises(OverflowError):
                spark.createDataFrame([value], T.DayTimeIntervalType())
        for value in (timedelta.min, timedelta.max):
            with self.assertRaises(OverflowError):
                spark.createDataFrame([(value,)])

    def test_inference_rejects_mixed_interval_and_numeric_values(self):
        spark = SparkSession.builder.remote("sc://127.0.0.1:1").create()
        delta = timedelta(microseconds=-1)
        for data in ([(delta,), (1,)], [([delta, 1],)], [({"a": delta, "b": 1},)]):
            with self.subTest(data=data):
                with self.assertRaisesRegex(TypeError, "cannot merge"):
                    spark.createDataFrame(data)

    def test_bounds_and_null_first_inputs_build_without_rpc(self):
        spark = SparkSession.builder.remote("sc://127.0.0.1:1").create()
        for schema in (None, ["interval"], T.StructType([
                T.StructField("interval", T.DayTimeIntervalType())])):
            spark.createDataFrame([(None,), (timedelta(microseconds=-(2**63)),),
                                   (timedelta(microseconds=2**63 - 1),)], schema)
        spark.createDataFrame([([],), ([None, timedelta(microseconds=-1)],)])
        spark.createDataFrame([(Row(x=None),), (Row(x=timedelta(microseconds=1)),)])
        spark.createDataFrame([({},), ({"k": timedelta(microseconds=-1)},)])


@unittest.skipUnless(os.environ.get("SPARK_REMOTE"), "requires a real Spark Connect server")
class IntervalInputLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.remote(os.environ["SPARK_REMOTE"]).create()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_scalar_inferred_explicit_and_collected_inputs(self):
        values = [None, -(2**63), -(2**63) + 1, -172800000001, -1000001, -1,
                  0, 1, 1000001, 172800000001, 2**63 - 2, 2**63 - 1]
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
        with self.assertRaisesRegex(RuntimeError, "Type mismatch"):
            schema = T.StructType([T.StructField("interval", interval)])
            self.spark.createDataFrame([(1,)], schema)

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
