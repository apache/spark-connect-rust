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
# byte-identical files whose sibling files are fork-adapted) as more is vendored.
# Do NOT list Rust-backed shims here (anything importing `pyspark._pyspark`), nor
# fork-adapted files (e.g. pipelines/api.py) — those intentionally differ from upstream.
set -euo pipefail

SPARK_TAG="v4.2.0"
SPARK_REPO="https://github.com/apache/spark.git"

# Directories under python/pyspark vendored verbatim from upstream (whole tree byte-identical).
VENDORED_DIRS=(
  "cloudpickle"
)

# Individual files under python/pyspark vendored verbatim from upstream. Listed per-file
# (not per-dir) because their parent dirs also hold fork-adapted files that intentionally
# differ (e.g. pipelines/api.py's connect->functions import repointing, and the Rust-backed
# errors/exceptions/__init__.py; upstream's errors/exceptions/{captured,connect}.py are
# omitted here as they need py4j/grpc). Keep these byte-identical to the tag.
VENDORED_FILES=(
  "logger/__init__.py"
  "logger/logger.py"
  "logger/worker_io.py"
  "errors/__init__.py"
  "errors/error_classes.py"
  "errors/error-conditions.json"
  "errors/utils.py"
  "errors/exceptions/base.py"
  "errors/exceptions/tblib.py"
  "pipelines/__init__.py"
  "pipelines/flow.py"
  "pipelines/output.py"
  "pipelines/source_code_location.py"
  "pipelines/graph_element_registry.py"
  "pipelines/type_error_utils.py"
  "pipelines/logging_utils.py"
  "pipelines/tests/__init__.py"
  "pipelines/tests/local_graph_element_registry.py"
  "pipelines/tests/test_decorators.py"
  "pipelines/tests/test_graph_element_registry.py"
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

echo "Fetching Apache Spark ${SPARK_TAG} (sparse, blobless) ..."
git clone --quiet --depth 1 --branch "${SPARK_TAG}" --filter=blob:none --sparse \
  "${SPARK_REPO}" "${WORK}/spark"
( cd "${WORK}/spark" && git sparse-checkout set \
    "${sparse_paths[@]/#/python/pyspark/}" >/dev/null )

status=0
check_one() {
  local p="$1" flag="$2"
  local ours="${REPO_ROOT}/python/pyspark/${p}"
  local theirs="${WORK}/spark/python/pyspark/${p}"
  if [[ ! -e "${theirs}" ]]; then
    echo "ERROR: upstream path python/pyspark/${p} not found at ${SPARK_TAG}"
    status=1
    return
  fi
  if diff ${flag} --exclude=__pycache__ "${ours}" "${theirs}" >/dev/null; then
    echo "OK    python/pyspark/${p} matches Apache Spark ${SPARK_TAG}"
  else
    echo "DRIFT python/pyspark/${p} differs from Apache Spark ${SPARK_TAG}:"
    diff ${flag} --exclude=__pycache__ "${ours}" "${theirs}" || true
    status=1
  fi
}

for p in "${VENDORED_DIRS[@]}"; do
  check_one "${p}" "-r"
done
for p in "${VENDORED_FILES[@]}"; do
  check_one "${p}" ""
done

if [[ "${status}" -ne 0 ]]; then
  echo
  echo "Vendored files have drifted from Apache Spark ${SPARK_TAG}."
  echo "Re-vendor from upstream (git subtree pull) or revert the local edit."
fi
exit "${status}"
