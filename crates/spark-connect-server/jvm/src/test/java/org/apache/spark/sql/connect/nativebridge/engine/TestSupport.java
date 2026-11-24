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

import java.lang.reflect.Field;
import java.util.Set;

import org.apache.spark.connect.proto.Relation;
import org.apache.spark.connect.proto.SQL;

final class TestSupport {
  private TestSupport() {}

  static void require(boolean condition, String message) {
    if (!condition) throw new AssertionError(message);
  }

  static Field field(Object owner, String name) throws ReflectiveOperationException {
    Field field = owner.getClass().getDeclaredField(name);
    field.setAccessible(true);
    return field;
  }

  static Set<?> registry(Object owner, String name) throws ReflectiveOperationException {
    return (Set<?>) field(owner, name).get(owner);
  }

  @FunctionalInterface
  interface CheckedAction { void run() throws Exception; }

  static Throwable cleanup(Throwable original, CheckedAction action) {
    try { action.run(); } catch (Throwable error) {
      if (original == null) return error;
      if (error != original) original.addSuppressed(error);
    }
    return original;
  }

  static Relation sql(String text) {
    return Relation.newBuilder().setSql(SQL.newBuilder().setQuery(text)).build();
  }
}
