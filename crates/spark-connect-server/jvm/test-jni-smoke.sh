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

if [[ $# -ne 1 || "$1" != /* || ! -f "$1" ]]; then
  echo "usage: $0 /absolute/path/to/libspark_connect_jni.{so,dylib}" >&2
  exit 1
fi
: "${SPARK_HOME:?Set SPARK_HOME to the pinned Spark assembly checkout}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$PROJECT_DIR/jvm/build.sh" --tests
"${NATIVE_TEST_PYTHON:-python3}" -B - "$PROJECT_DIR" "$SPARK_HOME" "$1" <<'PY'
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

root, spark_home, library = (Path(value).resolve(strict=True) for value in sys.argv[1:])
reports = root / ".build" / "jvm-native"
reports.mkdir(parents=True, exist_ok=True)
build = Path(tempfile.mkdtemp(prefix="jni-smoke-", dir=reports))
log_path = build / "native-jni-smoke.log"
command = [str(spark_home / "bin" / "spark-submit"), "--master", "local[2]",
           "--driver-memory", "1g", "--conf", "spark.ui.enabled=false",
           "--conf", "spark.driver.extraJavaOptions=-Xcheck:jni",
           "--conf", "spark.driver.host=127.0.0.1",
           "--conf", "spark.driver.bindAddress=127.0.0.1",
           "--conf", "spark.sql.classic.shuffleDependency.fileCleanup.enabled=false",
           "--conf", "spark.sql.warehouse.dir=" + str(build / "warehouse"),
           "--class", "org.apache.spark.sql.connect.nativebridge.engine.NativeJniSmokeTest",
           str(root / ".build" / "jvm" / "bridge-test.jar"), str(library)]
env = os.environ.copy()
env.update(SPARK_HOME=str(spark_home), SPARK_LOCAL_IP="127.0.0.1", SPARK_CONNECT_MODE="0")
env.pop("SPARK_REMOTE", None)
env.pop("PYSPARK_REMOTE_INIT_CONF_LEN", None)
started = time.monotonic()
report = {"command": command, "log": str(log_path), "passed": False}
try:
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=build, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=180)
    report["exit_code"] = result.returncode
    content = log_path.read_text(errors="replace")
    print(content, end="", flush=True)
    if result.returncode != 0:
        raise RuntimeError("real JNI smoke failed; see " + str(log_path))
    if re.search(r"WARNING in native method|FATAL ERROR in native method", content):
        raise RuntimeError("real JNI smoke triggered a JNI diagnostic; see " + str(log_path))
    if "NATIVE_JNI_SMOKE_OK:" not in content:
        raise RuntimeError("real JNI smoke success marker missing; see " + str(log_path))
    report["passed"] = True
except subprocess.TimeoutExpired as error:
    report["timed_out"] = True
    print(log_path.read_text(errors="replace"), end="", flush=True)
    raise RuntimeError("real JNI smoke exceeded 180 seconds; see " + str(log_path)) from error
finally:
    report["seconds"] = round(time.monotonic() - started, 3)
    (build / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
PY
