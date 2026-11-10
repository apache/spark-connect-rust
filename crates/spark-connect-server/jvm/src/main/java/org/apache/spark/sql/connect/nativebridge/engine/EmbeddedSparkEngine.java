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

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.connect.service.SparkConnectService$;
import org.apache.spark.util.ShutdownHookManager;

/** Spark-submit launcher: Rust protocol/session/operation ownership with Catalyst execution. */
public final class EmbeddedSparkEngine {
  static final String SHUTDOWN_COMPLETE = "Embedded Spark Engine shutdown complete";
  static final String STARTUP_CLEANUP_COMPLETE =
      "Embedded Spark Engine startup failed; cleanup complete";
  private EmbeddedSparkEngine() {}

  public static void main(String[] arguments) throws InterruptedException {
    if (arguments.length < 1 || arguments.length > 2) {
      throw new IllegalArgumentException(
          "usage: EmbeddedSparkEngine <absolute-native-library> [listen-address]");
    }
    String address = arguments.length == 2 ? arguments[1] : "127.0.0.1:15002";
    SparkSession session = SparkSession.builder().appName("Rust Spark Connect with Catalyst")
        .config("spark.sql.artifact.isolation.enabled", "true")
        .config("spark.sql.artifact.isolation.alwaysApplyClassloader", "true")
        .getOrCreate();
    SparkJvmEngine engine = null;
    NativeEngineServer server = null;
    CountDownLatch stopped = new CountDownLatch(1);
    AtomicBoolean launched = new AtomicBoolean();
    Object hook = null;
    try {
      engine = new SparkJvmEngine(session);
      server = new NativeEngineServer(arguments[0], address, engine);
      // NativeEngineServer owns the engine after acquisition; the extra idempotent close also
      // covers Java-side startup failures before acquisition and keeps one cleanup helper.
      SparkJvmEngine shutdownEngine = engine;
      NativeEngineServer shutdownServer = server;
      hook = ShutdownHookManager.addShutdownHook(
          ShutdownHookManager.SPARK_CONTEXT_SHUTDOWN_PRIORITY() + 1, () -> {
        shutdown(shutdownServer::close, shutdownEngine::close, session::stop, stopped,
            !launched.get());
        return scala.runtime.BoxedUnit.UNIT;
      });
      if (SparkConnectService$.MODULE$.server() != null) {
        throw new IllegalStateException("a Java Spark Connect listener was started");
      }
      System.out.println("Embedded Spark Connect listening on " + server.address());
      System.out.println("Embedded Spark Connect JVM PID " + ProcessHandle.current().pid()
          + "; Java Connect listener absent");
      launched.set(true);
      stopped.await();
    } catch (Throwable error) {
      if (hook != null) {
        try {
          ShutdownHookManager.removeShutdownHook(hook);
        } catch (Throwable cleanup) {
          if (cleanup != error) error.addSuppressed(cleanup);
        }
      }
      NativeEngineServer failedServer = server;
      SparkJvmEngine failedEngine = engine;
      shutdown(() -> { if (failedServer != null) failedServer.close(); },
          () -> { if (failedEngine != null) failedEngine.close(); }, session::stop, stopped,
          !launched.get());
      throw error;
    }
  }

  static boolean shutdown(Runnable stopNative, Runnable stopEngine, Runnable stopSession,
      CountDownLatch stopped, boolean startupFailure) {
    try {
      try {
        stopNative.run();
      } catch (Throwable failure) {
        System.err.println("Embedded Spark Engine shutdown failed: native stop incomplete; "
            + "retaining engine and Spark resources while JNI work may remain");
        failure.printStackTrace(System.err);
        return false;
      }
      Throwable failure = null;
      try {
        stopEngine.run();
      } catch (Throwable error) {
        failure = error;
      }
      try {
        stopSession.run();
      } catch (Throwable error) {
        if (failure == null) failure = error;
        else if (error != failure) failure.addSuppressed(error);
      }
      if (failure != null) {
        System.err.println("Embedded Spark Engine shutdown failed: Spark cleanup did not finish");
        failure.printStackTrace(System.err);
        return false;
      }
      System.out.println(startupFailure ? STARTUP_CLEANUP_COMPLETE : SHUTDOWN_COMPLETE);
      return true;
    } finally {
      stopped.countDown();
    }
  }
}
