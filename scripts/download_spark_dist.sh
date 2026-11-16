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
# Downloads the Apache Spark distribution PACKAGE of VERSION and unpacks it into DIR:
#
#   scripts/download_spark_dist.sh VERSION PACKAGE DIR
#
# The Apache CDN serves the current releases quickly. archive.apache.org keeps every release, but
# a download from it can take long enough to run a CI job out of time, so it only serves what the
# CDN does not have.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 VERSION PACKAGE DIR" >&2
  exit 2
fi
version=$1
package=$2
dir=$3

tgz=$(mktemp "${RUNNER_TEMP:-/tmp}/spark-dist.XXXXXX")
trap 'rm -f "$tgz"' EXIT
mkdir -p "$dir"
for base in https://dlcdn.apache.org/spark https://archive.apache.org/dist/spark; do
  url="$base/spark-$version/$package.tgz"
  echo "Downloading $url"
  if wget -q --timeout=60 --tries=3 -O "$tgz" "$url" && tar -xzf "$tgz" -C "$dir"; then
    exit 0
  fi
  echo "Could not download and unpack $url"
done
echo "::error::Could not download $package.tgz of Spark $version" >&2
exit 1
