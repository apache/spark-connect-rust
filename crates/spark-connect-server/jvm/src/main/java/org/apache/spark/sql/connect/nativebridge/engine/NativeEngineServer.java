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

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Objects;

/**
 * Loads the Rust protocol frontend with the typed JVM Engine implementation.
 *
 * The supplied engine is dedicated to this server: native startup acquires its ownership,
 * successful server close closes it, and startup failure after acquisition closes it as well.
 * Do not reuse or independently close the engine while this server is running. If construction
 * throws, the caller should close the engine to cover failures before native acquisition
 * (for example, library loading); engine close is idempotent. The caller continues to own the
 * original SparkSession and SparkContext.
 */
public final class NativeEngineServer implements AutoCloseable {
  private long handle;

  public NativeEngineServer(String absoluteLibraryPath, String listenAddress, SparkJvmEngine engine) {
    Path library = Path.of(Objects.requireNonNull(absoluteLibraryPath, "absoluteLibraryPath"));
    if (!library.isAbsolute()) throw new IllegalArgumentException("native library path must be absolute");
    Objects.requireNonNull(listenAddress, "listenAddress");
    Objects.requireNonNull(engine, "engine");
    System.load(library.toString());
    handle = nativeStart(listenAddress.getBytes(StandardCharsets.UTF_8), engine);
    if (handle == 0) throw new IllegalStateException("native engine returned an invalid handle");
  }

  public synchronized String address() {
    if (handle == 0) throw new IllegalStateException("native listener is closed");
    return new String(nativeAddress(handle), StandardCharsets.UTF_8);
  }

  /**
   * Drain native work and close the owned engine. Repeated successful closes are harmless;
   * failed teardown retains the handle and engine resources so the owner can retry.
   */
  @Override
  public synchronized void close() {
    if (handle != 0) {
      nativeStop(handle);
      handle = 0;
    }
  }

  private static native long nativeStart(byte[] utf8Address, SparkJvmEngine engine);
  private static native byte[] nativeAddress(long handle);
  private static native void nativeStop(long handle);
}
