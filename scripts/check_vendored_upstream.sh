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
# Verify that directories vendored verbatim from Apache Spark match the pinned
# upstream tag. These are pure upstream copies (brought in via `git subtree` from
# the SPARK_TAG below) with no fork-specific edits, so any difference means either
# an accidental local edit or an upstream bump that needs re-vendoring. Fails CI on
# drift.
#
# Extend VENDORED_PATHS (paths under python/pyspark) as more dirs are vendored.
# Do NOT list Rust-backed shims here (anything importing `pyspark._pyspark`) — those
# intentionally differ from upstream.
set -euo pipefail

SPARK_TAG="v4.2.0"
SPARK_REPO="https://github.com/apache/spark.git"

# Paths under python/pyspark that are vendored verbatim from upstream v4.2.0.
VENDORED_PATHS=(
  "cloudpickle"
)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Fetching Apache Spark ${SPARK_TAG} (sparse, blobless) ..."
git clone --quiet --depth 1 --branch "${SPARK_TAG}" --filter=blob:none --sparse \
  "${SPARK_REPO}" "${WORK}/spark"
( cd "${WORK}/spark" && git sparse-checkout set \
    "${VENDORED_PATHS[@]/#/python/pyspark/}" >/dev/null )

status=0
for p in "${VENDORED_PATHS[@]}"; do
  ours="${REPO_ROOT}/python/pyspark/${p}"
  theirs="${WORK}/spark/python/pyspark/${p}"
  if [[ ! -e "${theirs}" ]]; then
    echo "ERROR: upstream path python/pyspark/${p} not found at ${SPARK_TAG}"
    status=1
    continue
  fi
  if diff -r --exclude=__pycache__ "${ours}" "${theirs}" >/dev/null; then
    echo "OK    python/pyspark/${p} matches Apache Spark ${SPARK_TAG}"
  else
    echo "DRIFT python/pyspark/${p} differs from Apache Spark ${SPARK_TAG}:"
    diff -r --exclude=__pycache__ "${ours}" "${theirs}" || true
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo
  echo "Vendored files have drifted from Apache Spark ${SPARK_TAG}."
  echo "Re-vendor from upstream (git subtree pull) or revert the local edit."
fi
exit "${status}"
