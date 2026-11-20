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

if [[ $# -ne 0 && ( $# -ne 2 || "$1" != "--native-jar" ) ]]; then
  echo "usage: $0 [--native-jar <platform-or-all-native-jar>]" >&2
  exit 1
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${JAVA_HOME:?Set JAVA_HOME to a JDK 21 installation}"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/spark-connect-loader-test.XXXXXX")"
trap 'rm -rf -- "$BUILD_DIR"' EXIT
mkdir -p "$BUILD_DIR/classes/META-INF" "$BUILD_DIR/tmp"
SOURCE_ROOT="$PROJECT_DIR/jvm/src/main/java/org/apache/spark/sql/connect/nativebridge/engine"
TEST_ROOT="$PROJECT_DIR/jvm/src/test/java/org/apache/spark/sql/connect/nativebridge/engine"
"$JAVA_HOME/bin/javac" -proc:none -d "$BUILD_DIR/classes" \
  "$SOURCE_ROOT/NativeLibraryLoader.java" "$TEST_ROOT/NativeLibraryLoaderTest.java"
NATIVE_VERSION="$("${NATIVE_BUILD_PYTHON:-python3}" "$PROJECT_DIR/jvm/package-native.py" --print-version)"
printf 'artifact.version=%s\nnative.abi.version=1\n' "$NATIVE_VERSION" \
  > "$BUILD_DIR/classes/META-INF/spark-connect-native-adapter.properties"
cat > "$BUILD_DIR/fixture.c" <<'C'
#include <jni.h>
JNIEXPORT jint JNICALL
Java_org_apache_spark_sql_connect_nativebridge_engine_NativeLibraryLoaderTest_fixtureValue(
    JNIEnv *env, jclass cls) {
  (void)env;
  (void)cls;
  return 37;
}
C
case "$(uname -s)" in
  Darwin) JNI_PLATFORM=darwin; LIBRARY="$BUILD_DIR/libfixture.dylib"; LINK_FLAGS=(-dynamiclib) ;;
  Linux) JNI_PLATFORM=linux; LIBRARY="$BUILD_DIR/libfixture.so"; LINK_FLAGS=(-shared -fPIC) ;;
  *) echo "loader tests require macOS or Linux" >&2; exit 1 ;;
esac
"${CC:-cc}" -std=c11 -Wall -Wextra -Werror "${LINK_FLAGS[@]}" \
  -I "$JAVA_HOME/include" -I "$JAVA_HOME/include/$JNI_PLATFORM" \
  "$BUILD_DIR/fixture.c" -o "$LIBRARY"
TEST_CLASS=org.apache.spark.sql.connect.nativebridge.engine.NativeLibraryLoaderTest
run_loader_test() {
  local name="$1" marker="$2"
  shift 2
  local log="$BUILD_DIR/$name.log"
  "$JAVA_HOME/bin/java" -Xcheck:jni "-Djava.io.tmpdir=$BUILD_DIR/tmp" \
    "$@" 2>&1 | tee "$log" || return
  if grep -Eq 'WARNING in native method|WARNING: JNI local refs:|FATAL ERROR in native method' "$log"; then
    echo "native loader $name triggered a JNI diagnostic" >&2
    return 1
  fi
  if ! grep -Eq "$marker" "$log"; then
    echo "native loader $name success marker missing" >&2
    return 1
  fi
}
run_loader_test fixture '^NATIVE_LIBRARY_LOADER_OK checks=[1-9][0-9]*$' \
  -cp "$BUILD_DIR/classes" "$TEST_CLASS" "$LIBRARY"
if [[ $# -eq 2 ]]; then
  NATIVE_JAR="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
  [[ -f "$NATIVE_JAR" ]] || { echo "native JAR not found: $NATIVE_JAR" >&2; exit 1; }
  run_loader_test packaged '^NATIVE_PACKAGED_LOAD_OK$' \
    -cp "$BUILD_DIR/classes:$NATIVE_JAR" "$TEST_CLASS" --load-packaged
fi
if [[ -n "$(find "$BUILD_DIR/tmp" -mindepth 1 -print -quit)" ]]; then
  echo "native loader left extracted files after JVM exit" >&2
  exit 1
fi
