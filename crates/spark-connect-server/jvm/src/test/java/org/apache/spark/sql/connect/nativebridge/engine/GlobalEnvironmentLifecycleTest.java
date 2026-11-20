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

import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.cleanup;
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.field;
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.registry;
import static org.apache.spark.sql.connect.nativebridge.engine.TestSupport.require;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.spark.SparkEnv;
import org.apache.spark.sql.SparkSession;

/** Leaving a task scope must not restore a process-global environment stopped by its host. */
public final class GlobalEnvironmentLifecycleTest {
  private static final long TIMEOUT_MILLIS = TimeUnit.SECONDS.toMillis(15);

  private GlobalEnvironmentLifecycleTest() {}

  public static void main(String[] arguments) throws Exception {
    require(arguments.length == 0, "global environment test takes no arguments");
    SparkSession spark = SparkSession.builder().appName("Global SparkEnv lifecycle").getOrCreate();
    spark.sparkContext().setLogLevel("ERROR");
    SparkJvmEngine engine = new SparkJvmEngine(spark);
    SparkJvmEngine.Task task = engine.newTask("task scope crossing host stop");
    CountDownLatch entered = new CountDownLatch(1);
    CountDownLatch release = new CountDownLatch(1);
    AtomicReference<Throwable> workerFailure = new AtomicReference<>();
    AtomicReference<Object> result = new AtomicReference<>();
    Object expected = new Object();
    Thread worker = new Thread(() -> {
      try {
        result.set(runTaskScope(task, entered, release, expected));
      } catch (Throwable error) {
        workerFailure.set(error);
      }
    }, "global-environment-lifecycle-worker");
    worker.setDaemon(true);
    Throwable failure = null;
    try {
      SparkEnv initial = SparkEnv.get();
      require(initial != null, "host did not install its SparkEnv");
      worker.start();
      require(entered.await(TIMEOUT_MILLIS, TimeUnit.MILLISECONDS),
          "worker did not enter Task.run");
      require(field(task, "running").get(task) == worker && worker.isAlive(),
          "fixture did not keep Task.run active");
      require(SparkEnv.get() == initial, "entering Task.run changed the host environment");

      // No Spark job or SessionHolder is used: only Task.run's scope is held across host stop.
      // This isolates the process-global environment from session/artifact cleanup behavior.
      spark.sparkContext().stop();
      require(spark.sparkContext().isStopped(), "host did not stop its SparkContext");
      require(SparkEnv.get() == null, "host stop did not clear the process-global SparkEnv");
      require(worker.isAlive(), "worker left its task scope before host stop completed");
      release.countDown();
      worker.join(TIMEOUT_MILLIS);
      require(!worker.isAlive(), "worker did not leave Task.run after its gate opened");
      throwFailure(workerFailure.get());
      require(result.get() == expected, "task scope changed its action's result");
      require(field(task, "running").get(task) == null, "task retained its execution thread");
      require(SparkEnv.get() == null, "Task.run resurrected the host's stopped SparkEnv");

      task.close();
      task.close();
      engine.close();
      engine.close();
      require(engine.isClosed() && registry(engine, "tasks").isEmpty()
              && registry(engine, "sessions").isEmpty(),
          "stopped-context task/engine owners did not close");
      require(SparkEnv.get() == null, "owner cleanup restored the stopped SparkEnv");
    } catch (Throwable error) {
      failure = error;
    } finally {
      release.countDown();
      failure = cleanup(failure, () -> {
        worker.join(TIMEOUT_MILLIS);
        require(!worker.isAlive(), "fixture retained its task worker");
      });
      if (!worker.isAlive()) {
        failure = cleanup(failure, task::close);
        failure = cleanup(failure, engine::close);
      }
      failure = cleanup(failure, spark::stop);
    }
    throwFailure(failure);
    System.out.println("GLOBAL_ENVIRONMENT_LIFECYCLE_OK: Task.run remains active across host "
        + "SparkContext.stop; stopped SparkEnv remains null after worker exit and repeated "
        + "owner close; no SessionHolder or artifact cleanup dependency");
  }

  private static Object runTaskScope(SparkJvmEngine.Task task, CountDownLatch entered,
      CountDownLatch release, Object expected) throws Throwable {
    Class<?> actionType = Callable.class;
    Object action = Proxy.newProxyInstance(actionType.getClassLoader(), new Class<?>[] {actionType},
        (proxy, method, arguments) -> {
          if (!method.getName().equals("call")) {
            throw new AssertionError("unexpected task action method: " + method.getName());
          }
          entered.countDown();
          require(release.await(2 * TIMEOUT_MILLIS, TimeUnit.MILLISECONDS),
              "host did not release the task scope");
          return expected;
        });
    Method run = task.getClass().getDeclaredMethod("run", SparkJvmEngine.Session.class, actionType);
    run.setAccessible(true);
    try {
      return run.invoke(task, new Object[] {null, action});
    } catch (InvocationTargetException error) {
      throw error.getCause();
    }
  }

  private static void throwFailure(Throwable error) throws Exception {
    if (error instanceof Exception) throw (Exception) error;
    if (error != null) throw (Error) error;
  }
}
