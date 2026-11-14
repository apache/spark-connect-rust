#!/bin/sh
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

set -eu

case "${1:-}" in
  '') sanitize=0 ;;
  --sanitize) sanitize=1 ;;
  *) echo "Usage: $0 [--sanitize]" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [--sanitize]" >&2
  exit 2
fi

: "${JAVA_HOME:?Set JAVA_HOME to a JDK containing JNI headers}"
case "$(uname -s)" in
  Darwin) jni_platform=darwin ;;
  Linux) jni_platform=linux ;;
  *) echo 'The C ownership test requires macOS or Linux with pthreads' >&2; exit 2 ;;
esac

task_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output_dir="$task_root/.build/c-export"
mkdir -p "$output_dir"
set -- -std=c11 -Wall -Wextra -Werror -pthread
test_binary="$output_dir/ownership"
if [ "$sanitize" -eq 1 ]; then
  set -- "$@" -fsanitize=address,undefined -fno-omit-frame-pointer -g
  test_binary="$output_dir/ownership-sanitized"
fi
"${CC:-cc}" "$@" \
  -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/$jni_platform" \
  "$task_root/jni/tests/c_export_ownership.c" -o "$test_binary"
"$test_binary"
