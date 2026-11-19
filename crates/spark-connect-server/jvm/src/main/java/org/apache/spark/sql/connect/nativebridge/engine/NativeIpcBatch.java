// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

package org.apache.spark.sql.connect.nativebridge.engine;

/** Driver-side holder for a self-contained Spark Arrow IPC stream and its row count. */
public final class NativeIpcBatch {
  private final byte[] data;
  private final long rowCount;

  NativeIpcBatch(byte[] data, long rowCount) {
    this.data = data;
    this.rowCount = rowCount;
  }

  /** Read-only bytes owned by this batch; JNI copies them before returning to native code. */
  public byte[] data() { return data; }
  public long rowCount() { return rowCount; }
}
