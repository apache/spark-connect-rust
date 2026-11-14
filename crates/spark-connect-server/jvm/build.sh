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
rm -rf -- "$BUILD_DIR/classes" "$BUILD_DIR/test-classes"
# A failed compile must not leave a runnable jar from an earlier source revision.
rm -f -- "$BUILD_DIR/spark-connect-native-bridge.jar" "$BUILD_DIR/bridge-test.jar"
mkdir -p "$BUILD_DIR/classes"
MAIN_SOURCES=()
while IFS= read -r -d '' source; do
  MAIN_SOURCES+=("$source")
done < <(find "$PROJECT_DIR/jvm/src/main/java" -name '*.java' -type f -print0)
"$JAVAC" -proc:none -cp "$SPARK_JARS/*" -d "$BUILD_DIR/classes" \
  "${MAIN_SOURCES[@]}"
mkdir -p "$BUILD_DIR/classes/META-INF"
cp "$PROJECT_DIR/../../LICENSE" "$BUILD_DIR/classes/META-INF/LICENSE"
if [[ -f "$PROJECT_DIR/../../NOTICE" ]]; then
  cp "$PROJECT_DIR/../../NOTICE" "$BUILD_DIR/classes/META-INF/NOTICE"
fi
NATIVE_VERSION="$("${NATIVE_BUILD_PYTHON:-python3}" "$PROJECT_DIR/jvm/package-native.py" --print-version)"
printf 'artifact.version=%s\nnative.abi.version=1\n' "$NATIVE_VERSION" \
  > "$BUILD_DIR/classes/META-INF/spark-connect-native-adapter.properties"
"$JAR" --create --file "$BUILD_DIR/spark-connect-native-bridge.jar" -C "$BUILD_DIR/classes" .
echo "$BUILD_DIR/spark-connect-native-bridge.jar"
if [[ "${1:-}" == "--tests" ]]; then
  mkdir -p "$BUILD_DIR/test-classes"
  TEST_SOURCES=()
  while IFS= read -r -d '' source; do
    TEST_SOURCES+=("$source")
  done < <(find "$PROJECT_DIR/jvm/src/test/java" -name '*.java' -type f -print0)
  "$JAVAC" -proc:none -cp "$SPARK_JARS/*:$BUILD_DIR/classes" -d "$BUILD_DIR/test-classes" \
    "${TEST_SOURCES[@]}"
  "$JAR" --create --file "$BUILD_DIR/bridge-test.jar" \
    -C "$BUILD_DIR/classes" . -C "$BUILD_DIR/test-classes" .
  echo "$BUILD_DIR/bridge-test.jar"
fi
