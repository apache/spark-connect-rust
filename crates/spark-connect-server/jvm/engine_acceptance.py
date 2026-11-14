#!/usr/bin/env python3
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

"""Official PySpark and raw gRPC checks for the Rust frontend/JVM Engine backend."""

import argparse
import concurrent.futures
import datetime
from decimal import Decimal
import hashlib
import json
import logging
from pathlib import Path
import time
import tempfile
import traceback
import uuid


class RawSession:
    def __init__(self, remote, timeout):
        import grpc
        from pyspark.sql.connect.proto import base_pb2 as base
        from pyspark.sql.connect.proto import base_pb2_grpc as rpc

        self.base = base
        self.timeout = timeout
        self.session_id = str(uuid.uuid4())
        self.user = base.UserContext(user_id="jvm-engine-acceptance")
        self.server_id = None
        self.released = False
        address = remote.removeprefix("sc://").split("/", 1)[0]
        self.channel = grpc.insecure_channel(address)
        self.stub = rpc.SparkConnectServiceStub(self.channel)

    def identity(self):
        fields = {"session_id": self.session_id, "user_context": self.user}
        if self.server_id is not None:
            fields["client_observed_server_side_session_id"] = self.server_id
        return fields

    def observe(self, response):
        assert response.session_id == self.session_id, response
        assert response.server_side_session_id, response
        if self.server_id is not None:
            assert response.server_side_session_id == self.server_id, response
        self.server_id = response.server_side_session_id

    def config(self, name, payload):
        request = self.base.ConfigRequest(
            **self.identity(), operation=self.base.ConfigRequest.Operation(**{name: payload})
        )
        response = self.stub.Config(request, timeout=self.timeout)
        self.observe(response)
        return {pair.key: pair.value if pair.HasField("value") else None
                for pair in response.pairs}

    def execute(self, relation, operation_id=None):
        operation_id = operation_id or str(uuid.uuid4())
        request = self.base.ExecutePlanRequest(
            **self.identity(), operation_id=operation_id,
            plan=self.base.Plan(root=relation)
        )
        return operation_id, self.stub.ExecutePlan(request, timeout=self.timeout)

    def collect(self, relation, expected_arrow_schema=None):
        import pyarrow as pa

        operation_id, stream = self.execute(relation)
        rows = []
        complete = False
        arrow_batches = 0
        response_ids = set()
        try:
            for response in stream:
                self.observe(response)
                assert response.operation_id == operation_id, response
                assert response.response_id and response.response_id not in response_ids, response
                response_ids.add(response.response_id)
                if response.HasField("arrow_batch"):
                    table = pa.ipc.open_stream(response.arrow_batch.data).read_all()
                    assert table.num_rows == response.arrow_batch.row_count, response
                    if expected_arrow_schema is not None:
                        assert table.schema.equals(expected_arrow_schema, check_metadata=False), (
                            table.schema, expected_arrow_schema
                        )
                    arrow_batches += 1
                    rows.extend(table.to_pylist())
                if response.HasField("result_complete"):
                    complete = True
            assert complete, "query ended without result_complete"
            assert expected_arrow_schema is None or arrow_batches, "query returned no Arrow schema"
            return rows
        finally:
            stream.cancel()

    def release(self, timeout=None):
        response = self.stub.ReleaseSession(
            self.base.ReleaseSessionRequest(session_id=self.session_id, user_context=self.user),
            timeout=self.timeout if timeout is None else timeout,
        )
        self.released = True
        assert response.session_id == self.session_id, response
        if self.server_id is not None:
            assert response.server_side_session_id == self.server_id, response
        return response

    def close(self):
        try:
            if not self.released:
                self.release()
        finally:
            self.channel.close()


def expect_rpc_error(action, codes, contains=None):
    import grpc

    try:
        action()
    except grpc.RpcError as error:
        assert error.code() in codes, (error.code(), error.details())
        if contains is not None:
            assert contains in error.details(), error.details()
        return {"code": error.code().name, "details": error.details()}
    raise AssertionError("request unexpectedly succeeded")


def main():
    if not __debug__:
        raise RuntimeError("acceptance checks require Python assertions; disable optimized mode")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True, help="sc://host:port")
    parser.add_argument("--output", type=Path, help="write the checked JSON report here")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--cancel-timeout", type=float, default=15)
    args = parser.parse_args()
    if not args.remote.startswith("sc://"):
        parser.error("--remote must start with sc://")
    if args.timeout <= 0 or args.cancel_timeout <= 0:
        parser.error("timeouts must be positive")

    import grpc
    import pyspark
    from pyspark.sql import SparkSession, functions as F, types as T
    from pyspark.sql.connect.proto import base_pb2 as base
    from pyspark.sql.connect.proto import base_pb2_grpc as rpc
    from pyspark.sql.connect.proto import commands_pb2 as commands
    from pyspark.sql.connect.proto import relations_pb2 as relations

    logging.getLogger("SQLQueryContextLogger").setLevel(logging.CRITICAL)
    checks = []
    raw_sessions = []
    spark = None

    def check(name, action):
        started = time.monotonic()
        try:
            result = action()
            checks.append({"name": name, "passed": True,
                           "seconds": round(time.monotonic() - started, 3), "result": result})
        except Exception as error:
            checks.append({"name": name, "passed": False,
                           "seconds": round(time.monotonic() - started, 3),
                           "error": repr(error), "traceback": traceback.format_exc()})

    def raw_session():
        session = RawSession(args.remote, args.timeout)
        raw_sessions.append(session)
        return session

    def sql(query):
        return relations.Relation(sql=relations.SQL(query=query))

    try:
        spark = SparkSession.builder.remote(args.remote).create()

        def builder_configuration():
            key = "spark.acceptance.builder." + uuid.uuid4().hex
            shuffle = "spark.sql.shuffle.partitions"
            previous = spark.conf.get(shuffle)
            try:
                configured = (SparkSession.builder.remote(args.remote)
                              .config(key, "builder-value")
                              .config(shuffle, "3").getOrCreate())
                assert configured.session_id == spark.session_id
                assert configured.conf.get(key) == "builder-value"
                assert configured.conf.get(shuffle) == "3"
                return {"builder_silent_set": True, "unregistered_spark_key": True}
            finally:
                spark.conf.set(shuffle, previous)
                spark.conf.unset(key)

        check("official_builder_configuration", builder_configuration)

        def dataframe_join():
            left = spark.range(4).withColumnRenamed("id", "left_id")
            right = spark.range(2, 6).withColumnRenamed("id", "right_id")
            rows = left.join(right, F.col("left_id") == F.col("right_id")).orderBy(
                "left_id"
            ).collect()
            actual = [tuple(row) for row in rows]
            assert actual == [(2, 2), (3, 3)], actual
            return actual

        check("official_dataframe_join", dataframe_join)

        def unaliased_computations():
            frame = spark.range(4)
            assert [row[0] for row in frame.select(F.col("id") + 1).collect()] == [1, 2, 3, 4]
            assert frame.agg(F.sum("id")).collect()[0][0] == 6
            grouped = (frame.select((F.col("id") % 2).alias("k"), "id")
                       .groupBy("k").sum("id").orderBy("k").collect())
            assert [tuple(row) for row in grouped] == [(0, 2), (1, 4)]
            session = raw_session()
            rows = session.collect(sql("SELECT sum(id) FROM range(4)"))
            assert [list(row.values()) for row in rows] == [[6]], rows
            return {"sql_and_dataframe_unaliased_results": True}

        check("official_and_raw_unaliased_computations", unaliased_computations)

        def catalyst_types():
            frame = spark.sql(
                "SELECT transform(array(1, 2, 3), x -> x * x) AS squares, "
                "CAST('123.45' AS DECIMAL(10, 2)) AS amount, "
                "map('π', CAST(7 AS BIGINT)) AS labels, CAST(NULL AS INT) AS missing, "
                "1 AS required, array(1, CAST(NULL AS INT)) AS nullable_array, "
                "map('x', CAST(NULL AS INT)) AS nullable_map"
            )
            schema = frame.schema
            assert schema["squares"].dataType == T.ArrayType(T.IntegerType(), False), schema
            assert schema["amount"].dataType == T.DecimalType(10, 2), schema
            assert schema["labels"].dataType == T.MapType(T.StringType(), T.LongType(), False), schema
            assert schema["missing"].nullable and not schema["required"].nullable, schema
            assert schema["nullable_array"].dataType.containsNull, schema
            assert schema["nullable_map"].dataType.valueContainsNull, schema
            row = frame.collect()[0]
            assert row.squares == [1, 4, 9] and row.amount == Decimal("123.45"), row
            assert row.labels == {"π": 7} and row.missing is None and row.required == 1, row
            assert row.nullable_array == [1, None] and row.nullable_map == {"x": None}, row
            assert spark.range(0).collect() == [], "empty query returned rows"
            return {"schema": schema.jsonValue(), "amount": str(row.amount),
                    "squares": row.squares, "empty_rows": 0}

        check("official_catalyst_decimal_array_map_nullability", catalyst_types)

        def arrow_edge_types(large_vars):
            import pyarrow as pa

            query = (
                "SELECT NULL AS untyped_null, true AS true_value, false AS false_value, "
                "CAST(NULL AS BOOLEAN) AS null_boolean, '' AS empty_text, 'π😀' AS unicode_text, "
                "X'' AS empty_binary, X'00FFCF80F09F9880' AS binary_value, "
                "named_struct('empty', '', 'inner', named_struct('flag', true, 'missing', NULL)) "
                "AS nested, TIMESTAMP '2026-01-02 03:04:05.123456' AS instant, "
                "TIMESTAMP_NTZ '2026-01-02 03:04:05.123456' AS wall_time, "
                "DATE '2026-01-02' AS calendar_day"
            )
            text = pa.large_string() if large_vars else pa.string()
            binary = pa.large_binary() if large_vars else pa.binary()
            arrow_schema = pa.schema([
                pa.field("untyped_null", pa.null(), nullable=True),
                pa.field("true_value", pa.bool_(), nullable=False),
                pa.field("false_value", pa.bool_(), nullable=False),
                pa.field("null_boolean", pa.bool_(), nullable=True),
                pa.field("empty_text", text, nullable=False),
                pa.field("unicode_text", text, nullable=False),
                pa.field("empty_binary", binary, nullable=False),
                pa.field("binary_value", binary, nullable=False),
                pa.field("nested", pa.struct([
                    pa.field("empty", text, nullable=False),
                    pa.field("inner", pa.struct([
                        pa.field("flag", pa.bool_(), nullable=False),
                        pa.field("missing", pa.null(), nullable=True),
                    ]), nullable=False),
                ]), nullable=False),
                pa.field("instant", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("wall_time", pa.timestamp("us"), nullable=False),
                pa.field("calendar_day", pa.date32(), nullable=False),
            ])
            instant = datetime.datetime(2026, 1, 2, 3, 4, 5, 123456,
                                        tzinfo=datetime.timezone.utc)
            expected = {
                "untyped_null": None, "true_value": True, "false_value": False,
                "null_boolean": None, "empty_text": "", "unicode_text": "π😀",
                "empty_binary": b"", "binary_value": bytes.fromhex("00ffcf80f09f9880"),
                "nested": {"empty": "", "inner": {"flag": True, "missing": None}},
                "instant": instant, "wall_time": instant.replace(tzinfo=None),
                "calendar_day": datetime.date(2026, 1, 2),
            }
            session = raw_session()
            key = "spark.sql.execution.arrow.useLargeVarTypes"
            session.config("set", base.ConfigRequest.Set(pairs=[
                base.KeyValue(key=key, value=str(large_vars).lower()),
                base.KeyValue(key="spark.sql.session.timeZone", value="UTC"),
            ]))
            actual = session.collect(sql(query), expected_arrow_schema=arrow_schema)
            assert actual == [expected], actual

            spark_schema = T.StructType([
                T.StructField("untyped_null", T.NullType(), True),
                T.StructField("true_value", T.BooleanType(), False),
                T.StructField("false_value", T.BooleanType(), False),
                T.StructField("null_boolean", T.BooleanType(), True),
                T.StructField("empty_text", T.StringType(), False),
                T.StructField("unicode_text", T.StringType(), False),
                T.StructField("empty_binary", T.BinaryType(), False),
                T.StructField("binary_value", T.BinaryType(), False),
                T.StructField("nested", T.StructType([
                    T.StructField("empty", T.StringType(), False),
                    T.StructField("inner", T.StructType([
                        T.StructField("flag", T.BooleanType(), False),
                        T.StructField("missing", T.NullType(), True),
                    ]), False),
                ]), False),
                T.StructField("instant", T.TimestampType(), False),
                T.StructField("wall_time", T.TimestampNTZType(), False),
                T.StructField("calendar_day", T.DateType(), False),
            ])
            previous = {name: spark.conf.get(name) for name in
                        (key, "spark.sql.session.timeZone")}
            try:
                spark.conf.set(key, str(large_vars).lower())
                spark.conf.set("spark.sql.session.timeZone", "UTC")
                frame = spark.sql(query)
                assert frame.schema == spark_schema, (frame.schema, spark_schema)
                official = [row.asDict(recursive=True) for row in frame.collect()]
                official_expected = dict(expected)
                # Official PySpark collect converts TimestampType to the client's local timezone.
                official_expected["instant"] = instant.astimezone().replace(tzinfo=None)
                assert official == [official_expected], official
            finally:
                for name, value in previous.items():
                    spark.conf.set(name, value)
            return {"large_var_types": large_vars, "rows": 1,
                    "arrow_schema": str(arrow_schema), "spark_schema": spark_schema.jsonValue(),
                    "timestamp_utc": instant.isoformat(), "binary_hex": expected["binary_value"].hex(),
                    "exact_values_and_nullability": True}

        for large_vars in (False, True):
            check("raw_and_official_arrow_edge_types_large_vars_" + str(large_vars).lower(),
                  lambda large_vars=large_vars: arrow_edge_types(large_vars))

        def config_isolation_and_rollback():
            first, second = raw_session(), raw_session()
            key = "spark-connect.acceptance." + uuid.uuid4().hex
            created = key + ".new"
            first.config("set", base.ConfigRequest.Set(pairs=[base.KeyValue(key=key, value="before 😀")]))
            assert first.config("get", base.ConfigRequest.Get(keys=[key])) == {key: "before 😀"}
            assert second.config("get_option", base.ConfigRequest.GetOption(keys=[key])) == {key: None}
            assert first.server_id != second.server_id
            shuffle = "spark.sql.shuffle.partitions"
            previous = first.config("get", base.ConfigRequest.Get(keys=[shuffle]))[shuffle]
            failure = expect_rpc_error(
                lambda: first.config("set", base.ConfigRequest.Set(pairs=[
                    base.KeyValue(key=key, value="after"),
                    base.KeyValue(key=created, value="created"),
                    base.KeyValue(key=shuffle, value="invalid-integer"),
                ])), {grpc.StatusCode.INVALID_ARGUMENT}
            )
            actual = first.config("get_option", base.ConfigRequest.GetOption(keys=[key, created, shuffle]))
            assert actual == {key: "before 😀", created: None, shuffle: previous}, actual
            return {"isolated": True, "rollback_restored_existing_and_removed_new": True,
                    "failure": failure}

        check("raw_config_isolation_and_atomic_rollback", config_isolation_and_rollback)

        def silent_configuration():
            session = raw_session()
            before = "spark.acceptance.silent.before." + uuid.uuid4().hex
            after = "spark.acceptance.silent.after." + uuid.uuid4().hex
            shuffle = "spark.sql.shuffle.partitions"
            static = "spark.sql.warehouse.dir"
            previous = session.config("get", base.ConfigRequest.Get(keys=[shuffle]))[shuffle]
            request = base.ConfigRequest(
                **session.identity(), operation=base.ConfigRequest.Operation(
                    set=base.ConfigRequest.Set(silent=True, pairs=[
                        base.KeyValue(key=before, value="before"),
                        base.KeyValue(key=shuffle, value="invalid-integer"),
                        base.KeyValue(key=static, value="/ignored-static-setting"),
                        base.KeyValue(key=after, value="after"),
                    ])))
            response = session.stub.Config(request, timeout=args.timeout)
            session.observe(response)
            assert not response.pairs, response
            assert len(response.warnings) == 2, response
            assert any(shuffle in warning for warning in response.warnings), response
            assert any(static in warning for warning in response.warnings), response
            values = session.config("get", base.ConfigRequest.Get(keys=[before, after, shuffle]))
            assert values == {before: "before", after: "after", shuffle: previous}, values
            return {"valid_pairs_applied": True, "invalid_pairs_warned": 2}

        check("raw_silent_config_continues_after_invalid_and_static_keys", silent_configuration)

        def nullable_config_default():
            missing = "spark-connect.null-default." + uuid.uuid4().hex
            assert spark.conf.get(missing, None) is None
            session = raw_session()
            values = session.config("get_with_default", base.ConfigRequest.GetWithDefault(
                pairs=[base.KeyValue(key=missing)]))
            assert values == {missing: None}, values
            return {"official_null_default": True, "raw_absent_value_is_null": True}

        check("official_and_raw_nullable_config_default", nullable_config_default)

        def unsupported_plans():
            session = raw_session()
            name = "engine_forbidden_" + uuid.uuid4().hex
            queries = {
                "sql_write": sql(f"CREATE TEMP VIEW {name} AS SELECT 1 AS id"),
                "sql_script": sql(f"BEGIN CREATE TEMP VIEW {name} AS SELECT 1 AS id; END"),
                "with_relations": relations.Relation(
                    with_relations=relations.WithRelations(root=sql("SELECT 1 AS id"))
                ),
            }
            result = {}
            for label, relation in queries.items():
                result[label] = expect_rpc_error(
                    lambda relation=relation: session.collect(relation),
                    {grpc.StatusCode.UNIMPLEMENTED},
                )
            result["view_absent"] = expect_rpc_error(
                lambda: session.collect(sql(f"SELECT * FROM {name}")),
                {grpc.StatusCode.INVALID_ARGUMENT}, contains=name,
            )
            return result

        check("raw_unsupported_sql_writes_scripts_with_relations", unsupported_plans)

        def unsupported_spark_schemas():
            session = raw_session()
            cases = {
                "collated_string": sql("SELECT 'a' COLLATE UTF8_LCASE AS c"),
                "field_metadata": spark.range(1).select(
                    F.col("id").alias("id", metadata={"unit": "seconds"})
                )._plan.plan(spark.client),
            }
            result = {}
            for name, relation in cases.items():
                request = base.AnalyzePlanRequest(
                    **session.identity(), schema=base.AnalyzePlanRequest.Schema(
                        plan=base.Plan(root=relation)))
                result[name] = {
                    "schema": expect_rpc_error(
                        lambda: session.stub.AnalyzePlan(request, timeout=args.timeout),
                        {grpc.StatusCode.UNIMPLEMENTED}),
                    "execute": expect_rpc_error(
                        lambda: session.collect(relation), {grpc.StatusCode.UNIMPLEMENTED}),
                }
            return result

        check("raw_spark_only_schema_information_is_rejected", unsupported_spark_schemas)

        def shuffle_cleanup_guard():
            session = raw_session()
            key = "spark.sql.classic.shuffleDependency.fileCleanup.enabled"
            previous = session.config("get_all", base.ConfigRequest.GetAll())
            try:
                session.config("set", base.ConfigRequest.Set(pairs=[base.KeyValue(key=key, value="true")]))
                failure = expect_rpc_error(
                    lambda: session.collect(sql("SELECT SUM(id) AS total FROM range(100)")),
                    {grpc.StatusCode.UNIMPLEMENTED})
                session.config("set", base.ConfigRequest.Set(pairs=[base.KeyValue(key=key, value="false")]))
                assert session.collect(sql("SELECT 2 AS answer")) == [{"answer": 2}]
            finally:
                if key in previous:
                    session.config("set", base.ConfigRequest.Set(pairs=[base.KeyValue(key=key, value=previous[key])]))
                else:
                    session.config("unset", base.ConfigRequest.Unset(keys=[key]))
            restored = session.config("get_all", base.ConfigRequest.GetAll())
            assert (key in restored) == (key in previous), (previous, restored)
            assert restored.get(key) == previous.get(key), (previous, restored)
            return {"unsupported_setting": failure, "session_recovered": True}

        check("raw_per_fetch_execution_rejects_shuffle_file_cleanup", shuffle_cleanup_guard)

        def rejected_inserts_leave_data_unchanged():
            def sql_string(value):
                return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

            import pyarrow as pa
            import pyarrow.parquet as pq

            session = raw_session()
            with tempfile.TemporaryDirectory(prefix="engine-write-guard-quote'-") as directory:
                literal = "quote' and backslash\\value"
                assert session.collect(sql("SELECT " + sql_string(literal) + " AS value")) == [
                    {"value": literal}]
                path = Path(directory)
                pq.write_table(pa.table({"id": pa.array([7], type=pa.int64())}), path / "seed.parquet")
                target = "parquet.`" + directory.replace("`", "``") + "`"

                def snapshot():
                    return {str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
                            for file in path.rglob("*") if file.is_file()}

                before = snapshot()
                assert session.collect(sql(f"SELECT * FROM {target}")) == [{"id": 7}]
                results = {}
                for label, statement in {
                    "insert": f"INSERT INTO {target} VALUES (99)",
                    "overwrite": f"INSERT OVERWRITE TABLE {target} VALUES (99)",
                    "cte_insert": f"WITH source AS (SELECT 99 AS id) INSERT INTO {target} SELECT * FROM source",
                    "execute_immediate_insert": "EXECUTE IMMEDIATE "
                        + sql_string(f"INSERT INTO {target} VALUES (99)"),
                    "directory_overwrite": "INSERT OVERWRITE DIRECTORY "
                        + sql_string(directory) + " USING parquet SELECT 99 AS id",
                }.items():
                    relation = sql(statement)
                    results[label] = {
                        "execute_root": expect_rpc_error(
                            lambda: session.collect(relation), {grpc.StatusCode.UNIMPLEMENTED})
                    }
                    for field, message in (("schema", base.AnalyzePlanRequest.Schema),
                                           ("is_local", base.AnalyzePlanRequest.IsLocal),
                                           ("is_streaming", base.AnalyzePlanRequest.IsStreaming)):
                        request = base.AnalyzePlanRequest(
                            **session.identity(), **{field: message(plan=base.Plan(root=relation))})
                        results[label][field] = expect_rpc_error(
                            lambda: session.stub.AnalyzePlan(request, timeout=args.timeout),
                            {grpc.StatusCode.UNIMPLEMENTED})
                    # This is the command shape emitted by official PySpark SparkSession.sql().
                    command = commands.Command(sql_command=commands.SqlCommand(input=relation))
                    request = base.ExecutePlanRequest(
                        **session.identity(), operation_id=str(uuid.uuid4()),
                        plan=base.Plan(command=command))
                    results[label]["sql_command"] = expect_rpc_error(
                        lambda: list(session.stub.ExecutePlan(request, timeout=args.timeout)),
                        {grpc.StatusCode.UNIMPLEMENTED})
                    assert snapshot() == before, "rejected SQL changed data files: " + label
                    assert session.collect(sql(f"SELECT * FROM {target}")) == [{"id": 7}]
                return {"target_rows": 1, "target_value": 7, "files_unchanged": True,
                        "rejections": results}

        check("raw_insert_overwrite_rejected_without_data_mutation", rejected_inserts_leave_data_unchanged)

        def user_error_statuses():
            session = raw_session()
            key = "spark-connect.missing." + uuid.uuid4().hex
            return {
                "missing_config": expect_rpc_error(
                    lambda: session.config("get", base.ConfigRequest.Get(keys=[key])),
                    {grpc.StatusCode.NOT_FOUND}, contains=key),
                "invalid_sql": expect_rpc_error(
                    lambda: session.collect(sql("SELECT FROM")), {grpc.StatusCode.INVALID_ARGUMENT}),
                "missing_column": expect_rpc_error(
                    lambda: session.collect(sql("SELECT absent_column FROM range(1)")),
                    {grpc.StatusCode.INVALID_ARGUMENT}, contains="absent_column"),
            }

        check("raw_spark_user_error_statuses", user_error_statuses)

        def unicode_error():
            session = raw_session()
            marker = "missing_π_😀_" + uuid.uuid4().hex
            return expect_rpc_error(
                lambda: session.collect(sql(f"SELECT * FROM `{marker}`")),
                {grpc.StatusCode.INVALID_ARGUMENT}, contains=marker,
            )

        check("raw_unicode_jvm_error", unicode_error)

        def release_cancels_and_session_recovers():
            session = raw_session()
            operation_id, stream = session.execute(
                sql("SELECT SUM(id % 13) AS total FROM range(1000000000000)")
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as workers:
                future = None
                try:
                    header = next(stream)
                    session.observe(header)
                    assert header.HasField("schema") and header.operation_id == operation_id, header

                    def consume():
                        return expect_rpc_error(lambda: list(stream), {grpc.StatusCode.CANCELLED})

                    future = workers.submit(consume)
                    time.sleep(0.3)
                    assert not future.done(), "long query ended before cancellation"
                    started = time.monotonic()
                    released = session.stub.ReleaseExecute(
                        base.ReleaseExecuteRequest(
                            **session.identity(), operation_id=operation_id,
                            release_all=base.ReleaseExecuteRequest.ReleaseAll(),
                        ), timeout=args.cancel_timeout,
                    )
                    session.observe(released)
                    assert released.operation_id == operation_id, released
                    cancelled = future.result(timeout=args.cancel_timeout)
                    cancel_seconds = time.monotonic() - started
                finally:
                    stream.cancel()
                assert session.collect(sql("SELECT 7 AS value")) == [{"value": 7}]
                return {"cancel_seconds": round(cancel_seconds, 3), "rpc": cancelled,
                        "same_session_recovered": True}

        check("raw_release_all_cancellation_and_same_session_recovery", release_cancels_and_session_recovers)

        range_rows = 1_000_000

        def partially_consumed_range(session):
            import pyarrow as pa

            session.config("set", base.ConfigRequest.Set(pairs=[base.KeyValue(
                key="spark.sql.execution.arrow.maxRecordsPerBatch", value="256"
            )]))
            operation_id, stream = session.execute(relations.Relation(
                range=relations.Range(start=0, end=range_rows, step=1, num_partitions=64)
            ))
            try:
                for response in stream:
                    session.observe(response)
                    assert response.operation_id == operation_id, response
                    assert not response.HasField("result_complete"), "range completed before data"
                    if response.HasField("arrow_batch"):
                        table = pa.ipc.open_stream(response.arrow_batch.data).read_all()
                        assert 0 < table.num_rows <= 256, table.num_rows
                        assert table.num_rows == response.arrow_batch.row_count, response
                        assert table.schema.equals(pa.schema([
                            pa.field("id", pa.int64(), nullable=False)
                        ]), check_metadata=False), table.schema
                        assert table.to_pylist() == [
                            {"id": value} for value in range(table.num_rows)
                        ], table
                        return operation_id, stream, table.num_rows
                raise AssertionError("range ended before its first Arrow batch")
            except BaseException:
                stream.cancel()
                raise

        def drain_interrupted_stream(session, operation_id, stream, initial_rows):
            def consume():
                rows = initial_rows
                complete = False
                try:
                    for response in stream:
                        session.observe(response)
                        assert response.operation_id == operation_id, response
                        if response.HasField("arrow_batch"):
                            assert response.arrow_batch.row_count > 0, response
                            rows += response.arrow_batch.row_count
                            assert rows <= range_rows, rows
                        if response.HasField("result_complete"):
                            complete = True
                except grpc.RpcError as error:
                    assert error.code() == grpc.StatusCode.CANCELLED, (
                        error.code(), error.details()
                    )
                    return {"terminal": error.code().name, "rows_observed": rows}
                # The transport can already have buffered the complete result when interrupted.
                assert complete and rows == range_rows, (complete, rows)
                return {"terminal": "result_complete", "rows_observed": rows}

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as workers:
                future = workers.submit(consume)
                try:
                    return future.result(timeout=args.cancel_timeout)
                finally:
                    stream.cancel()

        def release_partially_consumed_session():
            session = raw_session()
            operation_id, stream, first_rows = partially_consumed_range(session)
            started = time.monotonic()
            try:
                session.release(timeout=args.cancel_timeout)
                terminal = drain_interrupted_stream(session, operation_id, stream, first_rows)
                release_seconds = time.monotonic() - started
            finally:
                stream.cancel()
            released = expect_rpc_error(
                lambda: session.collect(sql("SELECT 1 AS value")),
                {grpc.StatusCode.FAILED_PRECONDITION}, contains="released",
            )
            recovery = raw_session()
            assert recovery.collect(sql("SELECT 11 AS value")) == [{"value": 11}]
            return {"first_batch_rows": first_rows, "range_rows": range_rows,
                    "release_seconds": round(release_seconds, 3),
                    "stream": terminal, "released_session": released,
                    "fresh_session_recovered": True}

        check("raw_release_session_after_partial_arrow_stream", release_partially_consumed_session)

        def channel_close_cancels_partial_stream():
            session = raw_session()
            operation_id, stream, first_rows = partially_consumed_range(session)
            previous_channel = session.channel
            address = args.remote.removeprefix("sc://").split("/", 1)[0]
            # Keep session cleanup and the recovery probe usable after closing the active channel.
            session.channel = grpc.insecure_channel(address)
            session.stub = rpc.SparkConnectServiceStub(session.channel)
            started = time.monotonic()
            try:
                previous_channel.close()
                terminal = drain_interrupted_stream(session, operation_id, stream, first_rows)
                cancel_seconds = time.monotonic() - started
            finally:
                stream.cancel()
                previous_channel.close()
            assert session.collect(sql("SELECT 13 AS value")) == [{"value": 13}]
            return {"first_batch_rows": first_rows, "range_rows": range_rows,
                    "cancel_seconds": round(cancel_seconds, 3),
                    "stream": terminal, "same_session_recovered_on_new_channel": True}

        check("raw_channel_close_after_partial_arrow_stream", channel_close_cancels_partial_stream)

        def released_session_cannot_reappear():
            session = raw_session()
            payload = base.ConfigRequest.Get(keys=["spark.sql.session.timeZone"])
            session.config("get", payload)
            server_id = session.server_id
            session.release()
            config = expect_rpc_error(
                lambda: session.config("get", payload), {grpc.StatusCode.FAILED_PRECONDITION},
                contains="released",
            )
            execute = expect_rpc_error(
                lambda: session.collect(sql("SELECT 1")), {grpc.StatusCode.FAILED_PRECONDITION},
                contains="released",
            )
            return {"server_side_session_id": server_id, "config": config, "execute": execute}

        check("raw_release_session_tombstone", released_session_cannot_reappear)
    except Exception as error:
        checks.append({"name": "client_setup", "passed": False, "error": repr(error),
                       "traceback": traceback.format_exc()})
    finally:
        for index, session in enumerate(reversed(raw_sessions)):
            check("raw_session_cleanup_" + str(index), session.close)
        if spark is not None:
            check("official_session_cleanup", spark.stop)

    report = {"schema_version": 1, "profile": "jvm-engine", "remote": args.remote,
              "client_versions": {"pyspark": pyspark.__version__, "grpc": grpc.__version__},
              "checks": checks, "passed": bool(checks) and all(item["passed"] for item in checks)}
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    print(serialized, flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
