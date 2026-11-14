#!/usr/bin/env bash
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
# Verify that files/directories vendored verbatim from Apache Spark match the pinned
# upstream tag. These are pure upstream copies with no fork-specific edits, so any
# difference means either an accidental local edit or an upstream bump that needs
# re-vendoring. Fails CI on drift.
#
# Extend VENDORED_DIRS (whole byte-identical trees) or VENDORED_FILES (individual
# byte-identical files whose sibling files are fork-adapted) as more is vendored, and
# VENDORED_MAPPED_FILES for a byte-identical copy outside python/pyspark.
# Do NOT list Rust-backed shims here (anything importing `pyspark._pyspark`), nor
# fork-adapted files (e.g. pipelines/api.py) — those intentionally differ from upstream.
set -euo pipefail

SPARK_TAG="v4.3.0"
SPARK_REPO="https://github.com/apache/spark.git"

# Directories under python/pyspark vendored verbatim from upstream (whole tree byte-identical).
VENDORED_DIRS=(
  "cloudpickle"
  # The entire pandas-on-Spark package is vendored byte-for-byte from upstream (only its
  # tests/ subdir is intentionally not shipped; excluded in check_one). Do NOT hand-edit any
  # file under pyspark/pandas -- re-vendor from Apache instead.
  "pandas"
  "mllib/linalg"
  "sql/worker"
  "sql/plot"
  "testing/tests"
  # New in 4.3.0: the client-side message-receiver package (pyspark.messages).
  "messages"
)

# Individual files under python/pyspark vendored verbatim from upstream. Listed per-file
# (not per-dir) because their parent dirs also hold fork-adapted files that intentionally
# differ (e.g. pipelines/api.py's connect->functions import repointing, and the Rust-backed
# errors/exceptions/__init__.py; upstream's errors/exceptions/connect.py is omitted here
# as it needs grpc, and captured.py imports py4j only where the JVM is used). Keep these
# byte-identical to the tag.
VENDORED_FILES=(
  # Call sites of the JVM gateway, which pyspark.sql.pandas.conversion imports.
  "traceback_utils.py"

  # The SparkConf and the TaskContext of pyspark, pure Python, which it exports.
  "conf.py"
  "taskcontext.py"

  # The Arrow C stream of a DataFrame, of DataFrame.__arrow_c_stream__.
  "sql/interchange.py"

  # Logging
  "logger/__init__.py"
  "logger/logger.py"
  "logger/worker_io.py"

  # Errors
  "errors/__init__.py"
  "errors/error_classes.py"
  "errors/error-conditions.json"
  "errors/utils.py"
  "errors/exceptions/base.py"
  "errors/exceptions/captured.py"
  "errors/exceptions/tblib.py"

  # Pipelines
  "pipelines/__init__.py"
  "pipelines/flow.py"
  "pipelines/output.py"
  # The CLI of Spark Declarative Pipelines (spark-pipelines), and the official tests of it, which
  # run through this client against a server.
  "pipelines/cli.py"
  "pipelines/init_cli.py"
  "pipelines/add_pipeline_analysis_context.py"
  "pipelines/block_session_mutations.py"
  "pipelines/tests/test_cli.py"
  "pipelines/tests/test_init_cli.py"
  "pipelines/tests/test_add_pipeline_analysis_context.py"
  "pipelines/tests/test_block_session_mutations.py"
  "pipelines/tests/test_spark_connect.py"
  "pipelines/tests/test_auto_cdc_flow.py"
  "pipelines/source_code_location.py"
  "pipelines/graph_element_registry.py"
  "pipelines/type_error_utils.py"
  "pipelines/logging_utils.py"
  "pipelines/tests/__init__.py"
  "pipelines/tests/local_graph_element_registry.py"
  "pipelines/tests/test_decorators.py"
  "pipelines/tests/test_graph_element_registry.py"

  # The progress handlers and the progress bar of the executions of a session, whose
  # ExecutePlanResponse is that of the message class of pyspark/sql/connect/proto.py.
  "sql/connect/shell/progress.py"

  # The logger of Spark Connect, and the types of the connect import path.
  "sql/connect/logging.py"
  "sql/connect/_typing.py"
  # The conversions of the storage levels and the cached relations of their messages.
  "sql/connect/conversion.py"

  # The spatial reference systems of the spatial types, and the SQLStringFormatter, of
  # pyspark.sql.
  "sql/geo_utils.py"
  "sql/sql_formatter.py"

  # Top-level modules
  "accumulators.py"
  "memory_profiler_ext.py"
  "profiler.py"
  # The ConnectProfilerCollector of spark.profile, of the UDF profiles that the client takes
  # from the updates of the Python accumulators of its executions.
  "sql/connect/profiler.py"
  "worker_util.py"
  "instrumentation_utils.py"

  # ML: core utilities
  "ml/common.py"
  "ml/dl_util.py"
  "ml/util.py"
  "ml/functions.py"
  # ML: the API of estimators, models, pipelines, evaluators and tuning, which runs on the server
  # through the Connect ML protocol (pyspark.ml.util), over the client of SparkSession.client.
  "ml/__init__.py"
  "ml/_typing.pyi"
  "ml/base.py"
  "ml/classification.py"
  "ml/clustering.py"
  "ml/evaluation.py"
  "ml/feature.py"
  "ml/fpm.py"
  "ml/image.py"
  "ml/model_cache.py"
  "ml/pipeline.py"
  "ml/recommendation.py"
  "ml/regression.py"
  # ml/stat.py is a Rust-backed shim; it is not vendored.
  "ml/tree.py"
  "ml/tuning.py"
  "ml/wrapper.py"
  # ML: the internals of pyspark.ml on Connect: its plans, the serialization of its params, and
  # its readers and writers.
  "ml/connect/proto.py"
  "ml/connect/serialize.py"
  "ml/connect/readwrite.py"
  "ml/connect/io_utils.py"
  "ml/connect/summarizer.py"
  "ml/connect/util.py"
  # The generated Spark Connect protobuf messages, which pyspark.ml builds (not base_pb2_grpc:
  # the Rust transport speaks gRPC; sql/connect/proto/__init__.py is the client's own).
  "sql/connect/proto/base_pb2.py"
  "sql/connect/proto/base_pb2.pyi"
  "sql/connect/proto/types_pb2.py"
  "sql/connect/proto/types_pb2.pyi"
  "sql/connect/proto/commands_pb2.py"
  "sql/connect/proto/commands_pb2.pyi"
  "sql/connect/proto/expressions_pb2.py"
  "sql/connect/proto/expressions_pb2.pyi"
  "sql/connect/proto/relations_pb2.py"
  "sql/connect/proto/relations_pb2.pyi"
  "sql/connect/proto/catalog_pb2.py"
  "sql/connect/proto/catalog_pb2.pyi"
  "sql/connect/proto/common_pb2.py"
  "sql/connect/proto/common_pb2.pyi"
  "sql/connect/proto/ml_pb2.py"
  "sql/connect/proto/ml_pb2.pyi"
  "sql/connect/proto/ml_common_pb2.py"
  "sql/connect/proto/ml_common_pb2.pyi"
  "sql/connect/proto/pipelines_pb2.py"
  "sql/connect/proto/pipelines_pb2.pyi"
  "sql/connect/proto/example_plugins_pb2.py"
  "sql/connect/proto/example_plugins_pb2.pyi"
  # ML connect base ABCs (Estimator/Transformer/Evaluator/Model) - pure upstream.
  "ml/connect/base.py"
  # The scalers and ArrayAssembler of pyspark.ml.connect, which run their fit and transform in
  # Python on the server (summarize_dataframe and the closures of the models) - pure upstream.
  "ml/connect/feature.py"
  # vector_to_array/array_to_vector of pyspark.ml.functions build their columns here, through
  # pyspark.sql.connect.functions.builtin - pure upstream.
  "ml/connect/functions.py"

  # ML: param (param API is unmodified)
  "ml/param/__init__.py"
  "ml/param/_shared_params_code_gen.py"
  "ml/param/shared.py"

  # ML: torch (excluding fork-adapted spark_connect_* shims)
  "ml/torch/__init__.py"
  "ml/torch/data.py"
  "ml/torch/distributor.py"
  "ml/torch/log_communication.py"
  "ml/torch/torch_run_process_wrapper.py"

  # ML: deepspeed (excluding fork-adapted spark_connect_* shims)
  "ml/deepspeed/__init__.py"
  "ml/deepspeed/deepspeed_distributor.py"

  # MLlib (stat dir handled separately due to adapted __init__.py)
  "mllib/__init__.py"
  "mllib/classification.py"
  "mllib/clustering.py"
  "mllib/common.py"
  "mllib/evaluation.py"
  "mllib/feature.py"
  "mllib/fpm.py"
  "mllib/random.py"
  "mllib/recommendation.py"
  "mllib/regression.py"
  "mllib/stat/_statistics.py"
  "mllib/stat/distribution.py"
  "mllib/stat/KernelDensity.py"
  "mllib/stat/test.py"
  "mllib/tree.py"
  "mllib/util.py"

  # Pandas: the whole pyspark/pandas tree is vendored as a VENDORED_DIRS entry above (every
  # file, tests/ excluded), so no individual pandas/*.py entries are listed here.

  # SQL: pandas (excluding adapted group_ops, functions)
  "sql/pandas/__init__.py"
  "sql/pandas/conversion.py"
  "sql/pandas/map_ops.py"
  "sql/pandas/serializers.py"
  "sql/pandas/typehints.py"
  "sql/pandas/types.py"
  "sql/pandas/utils.py"
  # pandas_udf and arrow_udf, which create the UDF with pyspark.sql.connect.udf._create_udf.
  "sql/pandas/functions.py"

  # SQL: core utilities
  "sql/variant_utils.py"
  "sql/conversion.py"
  "sql/datasource_internal.py"
  "sql/internal.py"
  "sql/profiler.py"
  # ExecutionInfo / PlanMetrics / MetricValue (DataFrame.executionInfo).
  "sql/metrics.py"

  # SQL: streaming. TwsTester runs a StatefulProcessor locally - pure upstream.
  "sql/streaming/tws_tester.py"
  # The listener and its events and progress classes, which the client builds of the JSON of
  # the server - pure upstream.
  "sql/streaming/listener.py"

  # Streaming (excluding adapted context, dstream, util)
  "streaming/__init__.py"
  "streaming/kinesis.py"
  "streaming/listener.py"

  # Testing utilities (excluding adapted connectutils, sqlutils, __init__)
  "testing/goldenutils.py"
  "testing/mllibutils.py"
  "testing/mlutils.py"
  "testing/objects.py"
  "testing/pandasutils.py"
  "testing/streamingutils.py"
  "testing/unittestutils.py"
  "testing/utils.py"
)

# Files vendored verbatim from upstream to a path of this repo outside python/pyspark, as
# "<path in this repo>=<path in Apache Spark>". Keep these byte-identical to the tag.
VENDORED_MAPPED_FILES=(
  # Spark's spatial reference system registry, which the Rust crate embeds (srs.rs) to map the
  # SRIDs of the spatial types to their CRS strings. Spark generates it from PROJ
  # (dev/generate_srs_registry.py) and ships the same file in sql/api and in PySpark.
  "crates/spark-connect/src/srs_registry.csv=python/pyspark/sql/srs_registry.csv"
)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Sparse-checkout the dirs + the parent dirs of the individual files (cone mode operates on
# directories, so a file path alone would not materialize).
sparse_paths=("${VENDORED_DIRS[@]}")
for f in "${VENDORED_FILES[@]}"; do
  sparse_paths+=("$(dirname "${f}")")
done
sparse_paths=("${sparse_paths[@]/#/python/pyspark/}")
for m in "${VENDORED_MAPPED_FILES[@]}"; do
  sparse_paths+=("$(dirname "${m#*=}")")
done

echo "Fetching Apache Spark ${SPARK_TAG} (sparse, blobless) ..."
git clone --quiet --depth 1 --branch "${SPARK_TAG}" --filter=blob:none --sparse \
  "${SPARK_REPO}" "${WORK}/spark"
( cd "${WORK}/spark" && git sparse-checkout set "${sparse_paths[@]}" >/dev/null )

status=0
# Compare the path $1 of this repo with the path $2 of Apache Spark at the tag.
check_paths() {
  local ours_path="$1" theirs_path="$2" flag="$3" extra_exclude="${4:-}"
  local ours="${REPO_ROOT}/${ours_path}"
  local theirs="${WORK}/spark/${theirs_path}"
  local name="${ours_path}"
  [[ "${ours_path}" != "${theirs_path}" ]] && name="${ours_path} (upstream ${theirs_path})"
  # Always ignore __pycache__; a caller may add one more --exclude (e.g. `tests` for the
  # whole pandas tree, which upstream ships but we intentionally do not vendor).
  local excludes=(--exclude=__pycache__)
  [[ -n "${extra_exclude}" ]] && excludes+=(--exclude="${extra_exclude}")
  if [[ ! -e "${theirs}" ]]; then
    echo "ERROR: upstream path ${theirs_path} not found at ${SPARK_TAG}"
    status=1
    return
  fi
  if diff ${flag} "${excludes[@]}" "${ours}" "${theirs}" >/dev/null; then
    echo "OK    ${name} matches Apache Spark ${SPARK_TAG}"
  else
    echo "DRIFT ${name} differs from Apache Spark ${SPARK_TAG}:"
    diff ${flag} "${excludes[@]}" "${ours}" "${theirs}" || true
    status=1
  fi
}

# Compare the path $1 under python/pyspark with the same path upstream.
check_one() {
  check_paths "python/pyspark/$1" "python/pyspark/$1" "$2" "${3:-}"
}

for p in "${VENDORED_DIRS[@]}"; do
  # The pandas-on-Spark tree is vendored in full EXCEPT its tests/ (which we do not ship).
  if [[ "${p}" == "pandas" ]]; then
    check_one "${p}" "-r" "tests"
  else
    check_one "${p}" "-r"
  fi
done
for p in "${VENDORED_FILES[@]}"; do
  check_one "${p}" ""
done
for m in "${VENDORED_MAPPED_FILES[@]}"; do
  check_paths "${m%%=*}" "${m#*=}" ""
done

if [[ "${status}" -ne 0 ]]; then
  echo
  echo "Vendored files have drifted from Apache Spark ${SPARK_TAG}."
  echo "Re-vendor from upstream (git subtree pull) or revert the local edit."
fi
exit "${status}"
