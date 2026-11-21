#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""The programs whose records of the logger of Spark Connect (`pyspark.sql.connect.logging`)
tests/golden/connect_logs.json holds of official PySpark 4.3.0 (pyspark-client), against the mock
server of _mock_connect_server.py, which scripts/capture_connect_logs_golden.py captures and
python/tests/test_connect_logs_offline.py compares the drop-in's with: each case sets up the mock
server and runs one call of the client, of a session of its own whose first plan reads the plan
compression settings, at the debug level or at the level that it names.

The records are compared of their logger, their level and their message, normalized of what is of a
run (`normalized`): the IDs that the client makes (of the session and the operations), the plan
IDs that it assigns, the user agent, the time of a plan compression and the IPC stream of the Arrow
data of a local relation."""

import contextlib
import logging
import os
import re
import sys

# The user of the requests, of both clients.
USER = "connect-logs"
# The confs of the plan compression of the server, which the first plan of a session reads, and
# those that `toPandas` reads, of a Spark 4.3.0 server.
COMPRESSION_CONFS = {
    "spark.connect.session.planCompression.threshold": "10485760",
    "spark.connect.session.planCompression.defaultAlgorithm": "ZSTD",
    "spark.sql.execution.pandas.structHandlingMode": "legacy",
    "spark.sql.execution.arrow.pyspark.selfDestruct.enabled": "false",
}


def _pb2():
    import pyspark.sql.connect.proto as pb2

    return pb2


def _ipc(table):
    # The Arrow IPC stream of `table`, of pyarrow, as the server streams a batch.
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _response(**fields):
    # The raw fields of an ExecutePlanResponse, which the mock server sends after its session ID,
    # operation ID and response ID.
    return _pb2().ExecutePlanResponse(**fields).SerializeToString()


def _schema_response():
    pb2 = _pb2()
    return _response(
        schema=pb2.DataType(
            struct=pb2.DataType.Struct(
                fields=[pb2.DataType.StructField(name="id", data_type=pb2.DataType(
                    long=pb2.DataType.Long()), nullable=False)]
            )
        )
    )


def _batch_response(values=(0, 1, 2), **arrow_fields):
    import pyarrow as pa

    pb2 = _pb2()
    data = _ipc(pa.table({"id": pa.array(list(values), type=pa.int64())}))
    return _response(
        arrow_batch=pb2.ExecutePlanResponse.ArrowBatch(
            row_count=len(values), data=data, **arrow_fields
        )
    )


def _chunked_batch_responses():
    # One Arrow batch in two chunks, as the server splits a batch over its chunk size.
    import pyarrow as pa

    pb2 = _pb2()
    data = _ipc(pa.table({"id": pa.array([0, 1, 2], type=pa.int64())}))
    half = len(data) // 2
    return [
        _response(arrow_batch=pb2.ExecutePlanResponse.ArrowBatch(
            row_count=3, data=data[:half], start_offset=0, chunk_index=0, num_chunks_in_batch=2)),
        _response(arrow_batch=pb2.ExecutePlanResponse.ArrowBatch(
            row_count=3, data=data[half:], start_offset=0, chunk_index=1, num_chunks_in_batch=2)),
    ]


def _metrics_response():
    pb2 = _pb2()
    metrics = pb2.ExecutePlanResponse.Metrics(
        metrics=[pb2.ExecutePlanResponse.Metrics.MetricObject(
            name="Range", plan_id=7, parent=7,
            execution_metrics={"numOutputRows": pb2.ExecutePlanResponse.Metrics.MetricValue(
                name="number of output rows", value=3, metric_type="sum")},
        )]
    )
    observed = pb2.ExecutePlanResponse.ObservedMetrics(
        name="obs", values=[pb2.Expression.Literal(long=3)], keys=["n"], plan_id=8
    )
    return _response(metrics=metrics, observed_metrics=[observed])


def _sql_command_response(query="select 1 as a"):
    pb2 = _pb2()
    relation = pb2.Relation(sql=pb2.SQL(query=query))
    return _response(
        sql_command_result=pb2.ExecutePlanResponse.SqlCommandResult(relation=relation)
    )


def _server_error(error_class="DIVIDE_BY_ZERO", message="[DIVIDE_BY_ZERO] Division by zero."):
    # An error that the server raised, of an ErrorInfo of its class, as `execute_error` takes it.
    return (
        message,
        "org.apache.spark.SparkArithmeticException",
        {
            "errorClass": error_class,
            "sqlState": "22012",
            "messageParameters": "{}",
            "classes": '["org.apache.spark.SparkArithmeticException"]',
        },
    )


def _range(spark):
    return spark.range(3)


def collect(spark, service):
    service.execute_payloads = [_schema_response(), _batch_response(), _metrics_response()]
    return _range(spark).collect()


def collect_again(spark, service):
    # The second plan of a session, which reads no settings.
    service.execute_payloads = [_schema_response(), _batch_response()]
    _range(spark).collect()
    return "again", _range(spark).collect()


def collect_of_a_failed_lookup(spark, service):
    # A server that does not have the confs of the plan compression: the lookup fails, which the
    # client logs, of the exception as the argument of a text of no placeholder for it.
    service.execute_payloads = [_schema_response(), _batch_response()]
    return _range(spark).collect()


def collect_of_a_lookup_that_a_policy_fails(spark, service):
    # The lookup fails of the exception of a retry policy, which the client logs as it caught it.
    from pyspark.sql.connect.client.retries import RetryPolicy

    class Failing(RetryPolicy):
        def can_retry(self, e):
            raise KeyError("of can_retry")

    spark.client.set_retry_policies([Failing()])
    service.execute_payloads = [_schema_response(), _batch_response()]
    return _range(spark).collect()


def collect_of_chunks(spark, service):
    service.execute_payloads = [_schema_response(), *_chunked_batch_responses()]
    return _range(spark).collect()


def to_pandas(spark, service):
    service.execute_payloads = [_schema_response(), _batch_response()]
    return _range(spark).toPandas().shape


def to_arrow(spark, service):
    service.execute_payloads = [_schema_response(), _batch_response()]
    return _range(spark).toArrow().num_rows


def to_local_iterator(spark, service):
    service.execute_payloads = [_schema_response(), _batch_response()]
    return list(_range(spark).toLocalIterator())


def schema(spark, service):
    return _range(spark).schema


def explain(spark, service):
    with contextlib.redirect_stdout(None):
        _range(spark).explain(mode="formatted")
    return "explained"


def sql(spark, service):
    service.execute_payloads = [_sql_command_response()]
    return type(spark.sql("select 1 as a")).__name__


def write(spark, service):
    service.execute_payloads = []
    _range(spark).write.mode("overwrite").parquet("/tmp/connect-logs-case")
    return "written"


def server_error(spark, service):
    service.execute_error = _server_error()
    try:
        _range(spark).collect()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__


def cannot_parse(spark, service):
    # The server cannot parse the plan, so the client disables its plan compression.
    service.execute_error = _server_error(
        "CONNECT_INVALID_PLAN.CANNOT_PARSE", "[CONNECT_INVALID_PLAN.CANNOT_PARSE] no parse."
    )
    try:
        _range(spark).collect()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__


def registration(spark, service):
    service.execute_payloads = []
    spark.udf.registerJavaFunction("connect_logs_java", "org.example.Fn", "int")
    return "registered"


def compressed(spark, service):
    # Of a threshold of 40 bytes: a command over it, which compresses, and one that compression does
    # not make smaller, of a text of no repetition. Of commands of no plan IDs, whose sizes are of
    # their text only, as the plan IDs that a client assigns grow with its plans.
    pb2 = _pb2()
    service.confs = {**COMPRESSION_CONFS, "spark.connect.session.planCompression.threshold": "40"}
    service.execute_payloads = [_sql_command_response()]
    for query in ("select " + " + ".join(["1"] * 40), "select 'abcdefghijklmnopqrstuvwxyz0123'"):
        command = pb2.Command()
        command.sql_command.input.sql.query = query
        spark.client.execute_command(command)
    return "compressed"


def client_api(spark, service):
    # The API of SparkSession.client, as pyspark.ml and other code that builds its own messages
    # uses it, of a plan of `to_proto`.
    pb2 = _pb2()
    client = spark.client
    plan = _range(spark)._plan.to_proto(client)
    service.execute_payloads = [_schema_response(), _batch_response()]
    client.to_table(plan, {})
    client.to_pandas(plan, {})
    list(client.to_table_as_iterator(plan, {}))
    client.schema(plan)
    client.explain_string(plan, "simple")
    command = pb2.Command()
    command.sql_command.input.sql.query = "select 1 as a"
    service.execute_payloads = [_sql_command_response()]
    _, properties, _ = client.execute_command(command)
    list(client.execute_command_as_iterator(command))
    return sorted(properties)


def resources(spark, service):
    # The resources of the cluster, and a resource profile that the server creates.
    pb2 = _pb2()
    service.execute_payloads = [_response(
        get_resources_command_result=pb2.GetResourcesCommandResult()
    )]
    found = spark.client._resources()
    service.execute_payloads = [_response(
        create_resource_profile_command_result=pb2.CreateResourceProfileCommandResult(
            profile_id=3
        )
    )]
    return found, spark.client._create_profile(pb2.ResourceProfile())


def call_stack(spark, service):
    # With SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK, the requests of the executions, the analyses and
    # the configurations carry the call stack of the user's code, of its frames of this module:
    # the stack trace of their last user context extension, of the file of each frame by its
    # name, and of the names of the frames of this module's functions.
    service.execute_payloads = [_schema_response(), _batch_response()]
    os.environ["SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK"] = "1"
    try:
        _range(spark).collect()
        spark.conf.get("spark.sql.session.timeZone")
        _range(spark).schema
        spark.range(1).toPandas()
    finally:
        del os.environ["SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK"]
    _range(spark).collect()
    requests = [("execute", r) for r in service.execute_requests]
    requests += [("analyze", r) for r in service.analyze_requests]
    requests += [("config", r) for r in service.config_requests]
    return [(kind, call_stack_of(request)) for kind, request in requests]


def call_stack_of_calls(spark, service):
    # With SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK: the stack of an attribute that is a
    # `functools.cached_property` of PySpark's (`isStreaming`), of the frame of its `__get__`; of
    # a method of the client API, of the name of its method; of a call of a hook of the executions
    # during the execution, of the stack of the execution, as PySpark's stack ends at the first
    # frame of the pyspark package; and of the Config of the stack trace of an error of the server.
    # (Of no progress handler, whose module binds the `sys.stdout` of its import.)
    from pyspark.sql.connect.session import SparkSession

    calls = []

    class Hook(SparkSession.Hook):
        def __init__(self, session):
            self.session = session

        def on_execute_plan(self, request):
            if not calls:
                calls.append(self.session.conf.get("spark.sql.session.timeZone"))
            return request

    service.execute_payloads = [_schema_response(), _batch_response()]
    hooked = SparkSession.builder.remote(service.remote)._registerHook(Hook).create()
    os.environ["SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK"] = "1"
    try:
        _range(spark).isStreaming
        spark.client.to_table(_range(spark)._plan.to_proto(spark.client), {})
        _range(hooked).collect()
        service.execute_error = _server_error()
        try:
            _range(spark).collect()
        except Exception:  # noqa: BLE001
            pass
    finally:
        del os.environ["SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK"]
        # Of no release, which the mock server would refuse of a second session.
        hooked.release_session_on_close = False
        hooked.stop()
    requests = [("execute", r) for r in service.execute_requests]
    requests += [("analyze", r) for r in service.analyze_requests]
    requests += [("config", r) for r in service.config_requests]
    return [(kind, call_stack_of(request, "call_stack_of_calls")) for kind, request in requests]


def _incompressible(spark):
    # A plan over the threshold of 200 bytes that compression does not make smaller: of a binary
    # literal of random bytes, more than the text of the origins of its expressions.
    import random

    from pyspark.sql import functions as F

    rnd = random.Random(11)
    blob = bytes(rnd.randrange(256) for _ in range(2000))
    return spark.range(1).select(F.lit(blob).alias("b"))


def incompressible(spark, service):
    # The compression of the plan of each action is tried once, as PySpark's `to_proto` tries it,
    # also of the plan of `toPandas` and of `toLocalIterator`, which read settings of their own once
    # they built it, and the plan of the client API is sent as it is, of none, as PySpark's
    # `to_table`, `to_pandas`, `to_table_as_iterator`, `schema` and `explain_string` send the plan
    # that they are given.
    service.confs = {**COMPRESSION_CONFS, "spark.connect.session.planCompression.threshold": "200"}
    service.execute_payloads = [_schema_response(), _batch_response()]
    df = _incompressible(spark)
    df.collect()
    df.toPandas()
    list(df.toLocalIterator())
    client = spark.client
    plan = df._plan.to_proto(client)
    client.to_table(plan, {})
    client.to_pandas(plan, {})
    list(client.to_table_as_iterator(plan, {}))
    client.schema(plan)
    client.explain_string(plan, "simple")
    return plan.WhichOneof("op_type")


def hook(spark, service):
    # The hook of the executions of a session (`Builder._registerHook`): the plan is compressed,
    # or tried, once, before the hook, and the request that the hook returns is sent as it is.
    from pyspark.sql.connect.session import SparkSession

    class Hook(SparkSession.Hook):
        def __init__(self, session):
            self.session = session

        def on_execute_plan(self, request):
            logging.getLogger("pyspark.sql.connect.logging").debug("hook")
            return request

    service.confs = {**COMPRESSION_CONFS, "spark.connect.session.planCompression.threshold": "200"}
    service.execute_payloads = [_schema_response(), _batch_response()]
    hooked = SparkSession.builder.remote(service.remote)._registerHook(Hook).create()
    try:
        _incompressible(hooked).collect()
    finally:
        # Of no release, which the mock server would refuse of a second session.
        hooked.release_session_on_close = False
        hooked.stop()
    return "hooked"


class _Raising(logging.Handler):
    # A handler of the logger that raises `exception`, once, at the first record whose text starts
    # with `prefix`.
    prefix = None
    exception = ValueError

    def emit(self, record):
        if self.prefix is not None and str(record.msg).startswith(self.prefix):
            prefix, self.prefix = self.prefix, None
            raise self.exception(f"raised at {prefix}")


def handler_raises(spark, service):
    # A handler of the logger that raises at a record: the call fails at the record, of the
    # exception of the handler, and before its request when the record precedes it, as the
    # exception of PySpark's log goes out of its call; but the log of the settings of the plan
    # compression is in the `try` of PySpark's lookup, whose `except Exception` catches it and
    # disables the compression, of the exception logged; and a KeyboardInterrupt at a response
    # interrupts the execution, as PySpark's `except KeyboardInterrupt` does.
    from pyspark.sql.connect.logging import logger

    raising = _Raising()
    logger.addHandler(raising)
    service.execute_payloads = [_schema_response(), _batch_response()]
    outcomes = []
    try:
        for prefix, exception, action in [
            ("Plan compression threshold", ValueError, lambda: _range(spark).collect()),
            ("ExecuteAndFetch", ValueError, lambda: _range(spark).collect()),
            ("ExecuteAndFetchAsIterator. Request", ValueError, lambda: _range(spark).collect()),
            ("Received the schema.", ValueError, lambda: _range(spark).collect()),
            ("Received arrow batch", ValueError, lambda: _range(spark).toPandas()),
            ("Schema for plan", ValueError, lambda: _range(spark).schema),
            ("Received arrow batch", KeyboardInterrupt, lambda: _range(spark).collect()),
        ]:
            raising.prefix, raising.exception = prefix, exception
            try:
                action()
                outcomes.append((prefix, "returned"))
            except BaseException as e:  # noqa: BLE001
                outcomes.append((prefix, type(e).__name__, str(e)))
    finally:
        logger.removeHandler(raising)
    return outcomes


def handler_interrupts(spark, service):
    # A handler of the logger that raises KeyboardInterrupt at a record: at a response, of collect
    # and of toLocalIterator, the execution is interrupted, and logged so, as by PySpark's `except
    # KeyboardInterrupt` around the iterator of the responses; at a record of the conversion of the
    # failure of the execution (`_handle_rpc_error`), in PySpark's `except Exception`, nothing is
    # interrupted, and the KeyboardInterrupt has grpcio's error of the call as its context.
    from pyspark.sql.connect.logging import logger

    raising = _Raising()
    logger.addHandler(raising)
    outcomes = []
    try:
        for prefix, error, action in [
            ("Received arrow batch", None, lambda: _range(spark).collect()),
            ("Received arrow batch", None, lambda: list(_range(spark).toLocalIterator())),
            ("GRPC Error received", _server_error(), lambda: _range(spark).collect()),
            ("Received ErrorInfo", _server_error(), lambda: _range(spark).collect()),
        ]:
            service.execute_payloads = [_schema_response(), _batch_response()]
            service.execute_error = error
            raising.prefix, raising.exception = prefix, KeyboardInterrupt
            interrupts = len(service.interrupt_requests)
            try:
                action()
                outcomes.append((prefix, "returned"))
            except BaseException as e:  # noqa: BLE001
                context = None if e.__context__ is None else type(e.__context__).__name__
                sent = len(service.interrupt_requests) - interrupts
                outcomes.append((prefix, type(e).__name__, str(e), context, sent))
    finally:
        logger.removeHandler(raising)
        service.execute_error = None
    return outcomes


def _raised_at_a_failed_lookup(spark, service, exception):
    # A handler of the logger that raises `exception` at the record of the failed lookup of the
    # plan compression, which PySpark's `except Exception as e` of the lookup logs: the exception
    # goes out of the call, of the error of the lookup as its context, before any request, and the
    # compression is disabled, so that the next plan looks nothing up; of a KeyboardInterrupt too,
    # which interrupts nothing.
    from pyspark.sql.connect.logging import logger

    raising = _Raising()
    raising.prefix, raising.exception = "Plan compression is disabled", exception
    logger.addHandler(raising)
    service.execute_payloads = [_schema_response(), _batch_response()]
    try:
        try:
            _range(spark).collect()
            outcome = "returned"
        except BaseException as e:  # noqa: BLE001
            context = None if e.__context__ is None else type(e.__context__).__name__
            sent = len(service.execute_requests), len(service.interrupt_requests)
            outcome = (type(e).__name__, str(e), context, sent)
    finally:
        logger.removeHandler(raising)
    return outcome, _range(spark).collect()


def raised_at_a_failed_lookup(spark, service):
    return _raised_at_a_failed_lookup(spark, service, ValueError)


def interrupted_at_a_failed_lookup(spark, service):
    return _raised_at_a_failed_lookup(spark, service, KeyboardInterrupt)


def closed_client(spark, service):
    # Calls of a closed client, as of a stopped session, which PySpark's client raises
    # NO_ACTIVE_SESSION of as it reads its `_stub` to send, after the records that come before: of
    # an execution, its plan, or its command, `ExecuteAndFetch` and its request, of whose operation
    # ID the error is, and of an analysis, its plan. Nothing is sent. (Of `_closed`, which `stop()`
    # sets, as the release of the session of `stop()` is of a response of the mock server of no
    # session ID, which PySpark's `release_session` warns of.)
    service.execute_payloads = [_schema_response(), _batch_response()]
    _range(spark).collect()
    plan = _range(spark)._plan.to_proto(spark.client)
    spark.client._closed = True
    sent = len(service.execute_requests), len(service.analyze_requests)
    outcomes = []
    try:
        for name, action in [
            ("collect", lambda: _range(spark).collect()),
            ("schema", lambda: _range(spark).filter("id > 0").schema),
            ("sql", lambda: spark.sql("select 1 as a")),
            ("to_table", lambda: spark.client.to_table(plan, {})),
        ]:
            try:
                action()
                outcomes.append((name, "returned"))
            except Exception as e:  # noqa: BLE001
                has_operation_id = getattr(e, "_operation_id", None) is not None
                outcomes.append((name, type(e).__name__, e.getCondition(), has_operation_id))
    finally:
        spark.client._closed = False
    sent = (len(service.execute_requests), len(service.analyze_requests)) == sent
    return outcomes, sent


def closed_client_before_its_first_plan(spark, service):
    # A client closed before its first plan: the lookup of the plan compression fails of the
    # closed client, which PySpark's `except Exception as e` logs, and the execution logs the
    # records that come before its NO_ACTIVE_SESSION.
    spark.client._closed = True
    try:
        _range(spark).collect()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__, e.getCondition()
    finally:
        spark.client._closed = False


def explain_not_allowed(spark, service):
    # A mode that `explain_string` does not allow: PySpark's logs the plan of the mode as it is
    # given, and then raises VALUE_NOT_ALLOWED, of a DataFrame and of the client API.
    outcomes = []
    for explain in (
        lambda: _range(spark).explain(mode="FORMATTED"),
        lambda: spark.client.explain_string(_range(spark)._plan.to_proto(spark.client), "bogus"),
    ):
        try:
            with contextlib.redirect_stdout(None):
                explain()
        except Exception as e:  # noqa: BLE001
            outcomes.append(type(e).__name__)
    return outcomes


def call_stack_of(request, case="call_stack"):
    """The call stack of the user context of `request`, the bytes of a request of the mock
    server, of the frames of `run_case` and of the case `case`: of each, its function, the name of
    its file and its line; or None of a request that carries none."""
    pb2 = _pb2()
    # The user_context of each request of a call stack: ExecutePlanRequest=2,
    # AnalyzePlanRequest=2, ConfigRequest=2.
    from _mock_connect_server import parse_message

    context = parse_message(request).get(2)
    if not context:
        return None
    extensions = pb2.UserContext.FromString(context[0]).extensions
    if not extensions:
        return None
    error = pb2.FetchErrorDetailsResponse.Error()
    if not extensions[-1].Unpack(error):
        return None
    # Of the line of a frame of this module, and of none of another file, as `functools` of the
    # `__get__` of a cached property, whose lines are of the Python that runs.
    this = os.path.basename(__file__)
    frames = [
        (
            element.method_name,
            os.path.basename(element.file_name),
            element.line_number if os.path.basename(element.file_name) == this else None,
        )
        for element in error.stack_trace
    ]
    # From the frame of `run_case`, which calls the case, of the same callers in both runs.
    names = [name for name, _, _ in frames]
    return frames[names.index(case):] if case in names else frames


def keyboard_interrupt(spark, service):
    # The SIGINT of Ctrl-C while an execution waits for the server, which has sent its first
    # responses: the client logs it as it interrupts the execution, and reads nothing more of it,
    # which the server then fails with its OPERATION_CANCELED.
    import signal
    import threading

    service.execute_payloads = [_schema_response(), _batch_response()]
    service.block_result_complete = True
    service.interrupted_error = (
        "[OPERATION_CANCELED] Operation has been cancelled.",
        "org.apache.spark.SparkException",
        {"errorClass": "OPERATION_CANCELED"},
    )
    timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGINT))
    timer.start()
    try:
        _range(spark).collect()
    except KeyboardInterrupt:
        return "KeyboardInterrupt"
    finally:
        timer.cancel()
        service.block_result_complete = False


def signal_while_a_failure_is_converted(spark, service):
    # The SIGINT of Ctrl-C while the client converts the failure of an execution, as it reads the
    # stack-trace settings of the error of the server, which the server holds: PySpark's
    # `_handle_error` converts the failure in the `except Exception` clause of the execution, so
    # the KeyboardInterrupt goes out of the call as it is, of grpcio's error of the failure as its
    # context and of the operation ID of the execution, and nothing is interrupted.
    import signal
    import threading

    service.execute_error = _server_error()
    service.block_config = True
    interrupts = len(service.interrupt_requests)
    timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGINT))
    timer.start()
    try:
        _range(spark).collect()
        outcome = "returned"
    except BaseException as e:  # noqa: BLE001
        context = None if e.__context__ is None else type(e.__context__).__name__
        has_operation_id = getattr(e, "_operation_id", None) is not None
        sent = len(service.interrupt_requests) - interrupts
        outcome = (type(e).__name__, context, has_operation_id, sent)
    finally:
        timer.cancel()
        service.block_config = False
        service.release_response()
        service.execute_error = None
    return outcome


def signal_in_a_call_of_a_handler_of_a_conversion(spark, service):
    # A handler of a log of the conversion of the failure of an execution (`Received ErrorInfo`)
    # that runs an execution of its own, which the SIGINT of Ctrl-C ends as it waits: the nested
    # call runs its own `except KeyboardInterrupt`, which logs the interrupt and interrupts its
    # execution, and the KeyboardInterrupt goes out of the handler, and so out of the conversion of
    # the outer call, of grpcio's error of its failure as its context and of its operation ID.
    import signal
    import threading

    from pyspark.sql.connect.logging import logger

    class Nested(logging.Handler):
        armed = True

        def emit(self, record):
            if self.armed and str(record.msg).startswith("Received ErrorInfo"):
                self.armed = False
                service.execute_error = None
                service.execute_payloads = [_schema_response(), _batch_response()]
                service.block_result_complete = True
                timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGINT))
                timer.start()
                try:
                    _range(spark).collect()
                finally:
                    timer.cancel()

    service.execute_error = _server_error()
    service.interrupted_error = (
        "[OPERATION_CANCELED] Operation has been cancelled.",
        "org.apache.spark.SparkException",
        {"errorClass": "OPERATION_CANCELED"},
    )
    interrupts = len(service.interrupt_requests)
    handler = Nested()
    logger.addHandler(handler)
    try:
        _range(spark).collect()
        outcome = "returned"
    except BaseException as e:  # noqa: BLE001
        context = None if e.__context__ is None else type(e.__context__).__name__
        has_operation_id = getattr(e, "_operation_id", None) is not None
        sent = len(service.interrupt_requests) - interrupts
        outcome = (type(e).__name__, context, has_operation_id, sent)
    finally:
        logger.removeHandler(handler)
        service.block_result_complete = False
        service.execute_error = None
    return outcome


def signal_in_a_config_read_of_a_handler_of_a_conversion(spark, service):
    # A handler of a log of the conversion of the failure of an execution (`Received ErrorInfo`)
    # that reads a config, which the server holds, as the SIGINT of Ctrl-C comes: the read has no
    # `except KeyboardInterrupt`, and the conversion runs in the `except Exception` clause of the
    # execution, so the KeyboardInterrupt goes out of the read, of the handler and of the
    # conversion as it is, of grpcio's error of the failure as its context and of its operation
    # ID, and nothing is interrupted.
    import signal
    import threading

    from pyspark.sql.connect.logging import logger

    class Reading(logging.Handler):
        armed = True

        def emit(self, record):
            if self.armed and str(record.msg).startswith("Received ErrorInfo"):
                self.armed = False
                service.block_config = True
                timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGINT))
                timer.start()
                try:
                    spark.conf.get("spark.sql.shuffle.partitions")
                finally:
                    timer.cancel()

    service.execute_error = _server_error()
    interrupts = len(service.interrupt_requests)
    handler = Reading()
    logger.addHandler(handler)
    try:
        _range(spark).collect()
        outcome = "returned"
    except BaseException as e:  # noqa: BLE001
        context = None if e.__context__ is None else type(e.__context__).__name__
        has_operation_id = getattr(e, "_operation_id", None) is not None
        sent = len(service.interrupt_requests) - interrupts
        outcome = (type(e).__name__, context, has_operation_id, sent)
    finally:
        logger.removeHandler(handler)
        service.block_config = False
        service.release_response()
        service.execute_error = None
    return outcome


def iterator_open_across_close(spark, service):
    # A toLocalIterator left open across the close of its client, which `stop()` makes: grpcio's
    # `close` of the channel cancels the stream of the execution, so the next read of the stream,
    # past the rows of the batch that the iterator holds, raises the CANCELLED of the cancel, which
    # PySpark converts, of no more of its responses.
    service.execute_payloads = [_schema_response(), _batch_response()]
    iterator = _range(spark).toLocalIterator()
    first = next(iterator)
    spark.client.close()
    try:
        rest = list(iterator)
        outcome = ("rest", len(rest))
    except Exception as e:  # noqa: BLE001
        outcome = (type(e).__name__, str(e))
    return first, outcome


def stopped_by_another_thread(spark, service):
    # A session that another thread stops while an execution of it waits to retry the failure of
    # its stream: PySpark's iterator retries through the stub that it was made with, of the channel
    # that `stop()` closed, which raises grpc's `ValueError`, and its release of the execution
    # fails of the closed channel too, so the server gets neither.
    import threading
    import time

    from pyspark.sql.connect.logging import logger

    retrying = threading.Event()

    class Retrying(logging.Handler):
        def emit(self, record):
            if str(record.msg).startswith("Got error"):
                retrying.set()

    def stop():
        if retrying.wait(10):
            spark.stop()

    handler = Retrying()
    logger.addHandler(handler)
    service.execute_payloads = [_schema_response()]
    service.stream_failure = ("UNAVAILABLE", 300)
    stopper = threading.Thread(target=stop)
    stopper.start()
    try:
        _range(spark).collect()
        outcome = "returned"
    except Exception as e:  # noqa: BLE001
        outcome = (type(e).__name__, str(e))
    finally:
        stopper.join()
        logger.removeHandler(handler)
        service.stream_failure = None
    # The release, which no caller waits for.
    time.sleep(0.5)
    return outcome, len(service.reattach_requests), len(service.release_execute_requests)


def info_level(spark, service):
    service.execute_payloads = [_schema_response(), _batch_response()]
    return _range(spark).collect()


def info_level_of_an_error(spark, service):
    service.execute_error = _server_error()
    try:
        _range(spark).collect()
    except Exception as e:  # noqa: BLE001
        return type(e).__name__


# The cases, by name, of the level of their logs.
CASES = {
    "collect": (collect, "DEBUG"),
    "collect_again": (collect_again, "DEBUG"),
    "collect_of_a_failed_lookup": (collect_of_a_failed_lookup, "DEBUG"),
    "collect_of_a_lookup_that_a_policy_fails": (collect_of_a_lookup_that_a_policy_fails, "DEBUG"),
    "collect_of_chunks": (collect_of_chunks, "DEBUG"),
    "to_pandas": (to_pandas, "DEBUG"),
    "to_arrow": (to_arrow, "DEBUG"),
    "to_local_iterator": (to_local_iterator, "DEBUG"),
    "schema": (schema, "DEBUG"),
    "explain": (explain, "DEBUG"),
    "sql": (sql, "DEBUG"),
    "write": (write, "DEBUG"),
    "server_error": (server_error, "DEBUG"),
    "cannot_parse": (cannot_parse, "DEBUG"),
    "registration": (registration, "DEBUG"),
    "compressed": (compressed, "DEBUG"),
    "client_api": (client_api, "DEBUG"),
    "resources": (resources, "DEBUG"),
    "call_stack": (call_stack, "CRITICAL"),
    "call_stack_of_calls": (call_stack_of_calls, "CRITICAL"),
    "incompressible": (incompressible, "DEBUG"),
    "hook": (hook, "DEBUG"),
    "handler_raises": (handler_raises, "DEBUG"),
    "explain_not_allowed": (explain_not_allowed, "DEBUG"),
    "handler_interrupts": (handler_interrupts, "DEBUG"),
    "raised_at_a_failed_lookup": (raised_at_a_failed_lookup, "DEBUG"),
    "interrupted_at_a_failed_lookup": (interrupted_at_a_failed_lookup, "DEBUG"),
    "closed_client": (closed_client, "DEBUG"),
    "closed_client_before_its_first_plan": (closed_client_before_its_first_plan, "DEBUG"),
    "keyboard_interrupt": (keyboard_interrupt, "DEBUG"),
    "signal_while_a_failure_is_converted": (signal_while_a_failure_is_converted, "DEBUG"),
    "signal_in_a_call_of_a_handler_of_a_conversion": (
        signal_in_a_call_of_a_handler_of_a_conversion,
        "DEBUG",
    ),
    "signal_in_a_config_read_of_a_handler_of_a_conversion": (
        signal_in_a_config_read_of_a_handler_of_a_conversion,
        "DEBUG",
    ),
    "iterator_open_across_close": (iterator_open_across_close, "DEBUG"),
    "stopped_by_another_thread": (stopped_by_another_thread, "DEBUG"),
    "info_level": (info_level, "INFO"),
    "info_level_of_an_error": (info_level_of_an_error, "INFO"),
}

# The cases of a server without the confs of the plan compression.
WITHOUT_COMPRESSION_CONFS = {
    "collect_of_a_failed_lookup",
    "collect_of_a_lookup_that_a_policy_fails",
    "raised_at_a_failed_lookup",
    "interrupted_at_a_failed_lookup",
}
# The cases whose records of a compression are of plans of relations, whose sizes hold the plan IDs
# that each client assigns: of the sizes and the ratio normalized.
OF_PLAN_IDS = {"incompressible", "hook"}

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_PLAN_SIZES = re.compile(r"original_size=\d+, compressed_size=\d+, saving_ratio=-?[0-9.]+")
_PLAN_SIZES_OF_IDS = "original_size=<n>, compressed_size=<n>, saving_ratio=<r>"


def normalized(text):
    """`text`, a message of a record, of what is of a run normalized: the IDs that the client
    makes, the plan IDs that it assigns, the user agent, the port of the server, the time of a
    plan compression, the Arrow data of a local relation and the directory of the call site of
    the origin of an expression."""
    text = _UUID.sub("<uuid>", text)
    text = re.sub(r"plan_id: \d+", "plan_id: <id>", text)
    text = re.sub(r'client_type: "[^"]*"', 'client_type: "<agent>"', text)
    text = re.sub(r"duration_s=[0-9.]+", "duration_s=<t>", text)
    text = re.sub(r"127\.0\.0\.1:\d+", "127.0.0.1:<port>", text)
    text = re.sub(r'(local_relation \{ data: )"(\\.|[^"\\])*"', r'\1"<arrow>"', text)
    # The call site of the origin of an expression, of the name of its file, of no directory.
    text = re.sub(r'(call_site: )"[^"]*/([^"/]*)"', r'\1"\2"', text)
    # The responses of the mock server carry no server-side session ID, which PySpark's requests
    # then observe as an empty one, where the drop-in's observe none; a Spark server sends one.
    text = text.replace(' client_observed_server_side_session_id: ""', "")
    return text


def record_of(record):
    """The (logger, level, message) of `record`, normalized, its message of `getMessage()`, or of
    the error of its formatting, as `logging` reports it: the text, the type of each argument and
    the error."""
    try:
        message = record.getMessage()
    except Exception as e:  # noqa: BLE001
        message = "<unformatted: {} % ({}) of {}>".format(
            record.msg, ", ".join(type(arg).__name__ for arg in record.args), type(e).__name__
        )
    message = normalized(message)
    if record.exc_info:
        message += f" <exc_info: {type(record.exc_info[1]).__name__}>"
    return [record.name, record.levelname, message]


@contextlib.contextmanager
def connect_records(level):
    """The records of the logger of Spark Connect at `level`, as the program runs, of a handler of
    its own in place of its handlers: those of the thread that runs the case, as each case logs on
    its thread, and not those of a request that a session of another test, of a server that is
    gone, sends in the background, such as the retries of the release of an execution."""
    import threading

    from pyspark.sql.connect.logging import logger

    records = []
    thread = threading.get_ident()

    class Recorder(logging.Handler):
        def emit(self, record):
            if record.thread == thread:
                records.append(record)

    state = (logger.disabled, logger.level, logger.handlers[:], logger.propagate)
    logger.disabled, logger.propagate = False, False
    logger.handlers = [Recorder()]
    logger.setLevel(level)
    try:
        yield records
    finally:
        logger.disabled, previous, logger.handlers, logger.propagate = state
        logger.setLevel(previous)


def run_case(name, remote, service):
    """The records of the case `name`, of a new session of the mock server `remote` whose service
    is `service`, and what the case returned, as its repr."""
    from pyspark.errors import utils as errors_utils
    from pyspark.sql import SparkSession

    case, level = CASES[name]
    service.reset()
    if name not in WITHOUT_COMPRESSION_CONFS:
        service.confs = dict(COMPRESSION_CONFS)
    os.environ["SPARK_USER"] = USER
    # Of the origins of the expressions of the plans, as PySpark's `is_debugging_enabled` keeps the
    # answer of its first call of the process (`_enable_debugging_cache`), which an earlier test of
    # the process may have left False.
    debugging = errors_utils._enable_debugging_cache
    errors_utils._enable_debugging_cache = True
    spark = SparkSession.builder.remote(remote).create()
    try:
        with connect_records(level) as records:
            returned = case(spark, service)
        records = [record_of(record) for record in records]
        if name in OF_PLAN_IDS:
            records = [
                [logger, level, _PLAN_SIZES.sub(_PLAN_SIZES_OF_IDS, message)]
                for logger, level, message in records
            ]
        return records, repr(returned)
    finally:
        with connect_records("CRITICAL"):
            spark.stop()
        errors_utils._enable_debugging_cache = debugging


def capture(remote, service):
    """The records of every case, by name."""
    return {name: run_case(name, remote, service) for name in CASES}


# The records of a plan truncated by PySpark's `_proto_to_string(p, True)`, whose `_truncate`
# reads `FieldDescriptor.label`, which protobuf 7 and the drop-in's runtime without protobuf do not
# have: of their text of the truncation error then, and of the truncated plan where the
# descriptors have it, as those of protobuf 6.
TRUNCATED = re.compile(
    r"^(Executing plan |Schema for plan: |Explain \(mode=\w+\) for plan "
    r"|Execute command (as iterator )?for command )"
)


def truncated_plans(records):
    """The messages of the records of truncated plans of `records`, in order."""
    return [message for _, _, message in records if TRUNCATED.match(message)]


if __name__ == "__main__":  # pragma: no cover
    sys.exit("run scripts/capture_connect_logs_golden.py")
