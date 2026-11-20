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

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--tests" ) ]]; then
  echo "usage: $0 [--tests]" >&2
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SPARK_HOME:?Set SPARK_HOME to an existing Spark 4.2 assembly with shaded Connect gRPC jars}"
SPARK_JARS="$SPARK_HOME/jars"
if [[ -d "$SPARK_HOME/assembly/target/scala-2.13/jars" ]]; then
  SPARK_JARS="$SPARK_HOME/assembly/target/scala-2.13/jars"
fi
if [[ ! -d "$SPARK_JARS" ]]; then
  echo "Spark jar directory does not exist: $SPARK_JARS" >&2
  exit 1
fi
JAVAC="${JAVA_HOME:+$JAVA_HOME/bin/}javac"
JAR="${JAVA_HOME:+$JAVA_HOME/bin/}jar"
BUILD_DIR="$PROJECT_DIR/.build/jvm"
mkdir -p "$BUILD_DIR"
# Concurrent builders keep their classes private. Publish only complete jars so
# a running launcher never observes a removed or partially written application jar.
BUILD_STAGE="$(mktemp -d "$BUILD_DIR/.build.XXXXXX")"
trap 'rm -rf -- "$BUILD_STAGE"' EXIT
mkdir -p "$BUILD_STAGE/classes"
MAIN_SOURCES=()
while IFS= read -r -d '' source; do
  MAIN_SOURCES+=("$source")
done < <(find "$PROJECT_DIR/jvm/src/main/java" -name '*.java' -type f -print0)
"$JAVAC" -proc:none -cp "$SPARK_JARS/*" -d "$BUILD_STAGE/classes" \
  "${MAIN_SOURCES[@]}"
mkdir -p "$BUILD_STAGE/classes/META-INF"
cp "$PROJECT_DIR/../../LICENSE" "$BUILD_STAGE/classes/META-INF/LICENSE"
if [[ -f "$PROJECT_DIR/../../NOTICE" ]]; then
  cp "$PROJECT_DIR/../../NOTICE" "$BUILD_STAGE/classes/META-INF/NOTICE"
fi
NATIVE_VERSION="$("${NATIVE_BUILD_PYTHON:-python3}" "$PROJECT_DIR/jvm/package-native.py" --print-version)"
printf 'artifact.version=%s\nnative.abi.version=1\n' "$NATIVE_VERSION" \
  > "$BUILD_STAGE/classes/META-INF/spark-connect-native-adapter.properties"
"$JAR" --create --file "$BUILD_STAGE/spark-connect-native-bridge.jar" -C "$BUILD_STAGE/classes" .
if [[ "${1:-}" == "--tests" ]]; then
  mkdir -p "$BUILD_STAGE/test-classes"
  TEST_SOURCES=()
  while IFS= read -r -d '' source; do
    TEST_SOURCES+=("$source")
  done < <(find "$PROJECT_DIR/jvm/src/test/java" -name '*.java' -type f -print0)
  "$JAVAC" -proc:none -cp "$SPARK_JARS/*:$BUILD_STAGE/classes" -d "$BUILD_STAGE/test-classes" \
    "${TEST_SOURCES[@]}"
  "$JAR" --create --file "$BUILD_STAGE/bridge-test.jar" \
    -C "$BUILD_STAGE/classes" . -C "$BUILD_STAGE/test-classes" .
fi
# A compilation failure preserves previously published jars. Callers must honor
# a nonzero exit status and must not launch those older jars after a failed build.
# Both jars are self-contained, so publishing them independently is safe.
mv -f -- "$BUILD_STAGE/spark-connect-native-bridge.jar" "$BUILD_DIR/spark-connect-native-bridge.jar"
echo "$BUILD_DIR/spark-connect-native-bridge.jar"
if [[ "${1:-}" == "--tests" ]]; then
  mv -f -- "$BUILD_STAGE/bridge-test.jar" "$BUILD_DIR/bridge-test.jar"
  echo "$BUILD_DIR/bridge-test.jar"
fi
