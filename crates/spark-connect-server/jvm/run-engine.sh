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

usage() {
  echo "usage: $0 <absolute-native-library> [listen-address]" >&2
  echo "       $0 --native-jar <absolute-native-jar> [listen-address]" >&2
  exit 1
}
[[ $# -ge 1 ]] || usage
NATIVE_JAR=""
if [[ "$1" == "--native-jar" ]]; then
  [[ $# -ge 2 && $# -le 3 && "$2" == /* && -f "$2" && "$2" != *,* ]] || usage
  NATIVE_JAR="$2"
  NATIVE_INPUT=--bundled
  LISTEN_ADDRESS="${3:-127.0.0.1:15002}"
else
  [[ $# -le 2 && "$1" == /* && -f "$1" ]] || usage
  NATIVE_INPUT="$1"
  LISTEN_ADDRESS="${2:-127.0.0.1:15002}"
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$PROJECT_DIR/jvm/build.sh"
unset SPARK_REMOTE
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"
export SPARK_CONNECT_MODE=0
SPARK_ARGUMENTS=(--class org.apache.spark.sql.connect.nativebridge.engine.EmbeddedSparkEngine
  --master "${SPARK_MASTER:-local[2]}")
if [[ -n "$NATIVE_JAR" ]]; then
  SPARK_ARGUMENTS+=(--jars "$NATIVE_JAR")
fi
exec "$SPARK_HOME/bin/spark-submit" "${SPARK_ARGUMENTS[@]}" \
  "$PROJECT_DIR/.build/jvm/spark-connect-native-bridge.jar" "$NATIVE_INPUT" "$LISTEN_ADDRESS"
