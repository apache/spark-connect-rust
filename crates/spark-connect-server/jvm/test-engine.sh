#!/usr/bin/env bash
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

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_PYTHON="${NATIVE_TEST_PYTHON:-python3}"
# Compile and run against the same assembly when an explicit CLI path overrides
# the environment. Let argparse handle both option spellings and repeated flags.
SPARK_HOME="$("$TEST_PYTHON" -B - "$@" <<'PY'
import argparse
import os

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--spark-home", default=os.environ.get("SPARK_HOME"))
args, _ = parser.parse_known_args()
if args.spark_home is None:
    parser.error("set SPARK_HOME or pass --spark-home")
print(args.spark_home)
PY
)"
export SPARK_HOME
"$TEST_PYTHON" -B "$PROJECT_DIR/jvm/test_acceptance_guards.py"
"$PROJECT_DIR/jvm/build.sh" --tests
exec "$TEST_PYTHON" -B "$PROJECT_DIR/jvm/run_acceptance.py" "$@"
