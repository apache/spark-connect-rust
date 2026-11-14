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

import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.arrow.memory.ArrowBuf;
import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;

/**
 * An immutable Arrow vector batch leased to native code. Only descriptor metadata crosses JNI;
 * addresses refer to the vectors' original off-heap buffers. The native ArrowArray release
 * callback must close this object exactly once after every imported buffer/clone/slice is gone.
 * Query and session teardown deliberately do not own this batch's allocator.
 */
public final class NativeArrowBatch implements AutoCloseable {
  private static final AtomicLong LIVE_BATCHES = new AtomicLong();
  private final VectorSchemaRoot root;
  private final BufferAllocator allocator;
  private final Node descriptor;
  private final AtomicBoolean closed = new AtomicBoolean();
  private final Set<NativeArrowBatch> owners;
  private Throwable closeFailure;
  private boolean closeComplete;

  NativeArrowBatch(VectorSchemaRoot root, BufferAllocator allocator, Set<NativeArrowBatch> owners) {
    this.root = root;
    this.allocator = allocator;
    this.owners = owners;
    List<FieldVector> fields = root.getFieldVectors();
    Node[] children = new Node[fields.size()];
    for (int i = 0; i < fields.size(); i++) children[i] = node(fields.get(i));
    // A RecordBatch is exported as one non-null struct with the field arrays as children.
    descriptor = new Node(root.getRowCount(), 0, new long[] {0}, new long[] {0}, children);
    owners.add(this);
    LIVE_BATCHES.incrementAndGet();
  }

  public Node descriptor() {
    if (closed.get()) throw new SparkJvmEngine.EngineException(9, "native Arrow batch is closed");
    return descriptor;
  }

  public static long liveBatchCount() { return LIVE_BATCHES.get(); }

  // Package-local value inspection allows tests to prove query-close independence without
  // exposing mutable vectors in the JNI/public engine boundary.
  Object valueAt(int column, int row) {
    if (closed.get()) throw new SparkJvmEngine.EngineException(9, "native Arrow batch is closed");
    return root.getVector(column).getObject(row);
  }

  long vectorBufferAddress(int column, int buffer) {
    if (closed.get()) throw new SparkJvmEngine.EngineException(9, "native Arrow batch is closed");
    ArrowBuf value = root.getVector(column).getFieldBuffers().get(buffer);
    return value.capacity() == 0 ? 0 : value.memoryAddress();
  }

  private static Node node(FieldVector vector) {
    if (vector.getField().getDictionary() != null) {
      throw new SparkJvmEngine.EngineException(12, "dictionary vectors are unsupported by native Arrow export");
    }
    List<ArrowBuf> fields = vector.getFieldBuffers();
    long[] addresses = new long[fields.size()];
    long[] sizes = new long[fields.size()];
    for (int i = 0; i < fields.size(); i++) {
      ArrowBuf buffer = fields.get(i);
      addresses[i] = buffer.capacity() == 0 ? 0 : buffer.memoryAddress();
      sizes[i] = buffer.capacity();
    }
    List<FieldVector> vectors = vector.getChildrenFromFields();
    Node[] children = new Node[vectors.size()];
    for (int i = 0; i < vectors.size(); i++) children[i] = node(vectors.get(i));
    return new Node(vector.getValueCount(), vector.getNullCount(), addresses, sizes, children);
  }

  @Override
  public synchronized void close() {
    checkCloseFailure();
    if (closeComplete) return;
    closed.set(true);
    Throwable failure = null;
    try {
      root.close();
    } catch (Throwable error) {
      failure = error;
    }
    try {
      allocator.close();
    } catch (Throwable error) {
      if (failure == null) failure = error;
      else if (failure != error) failure.addSuppressed(error);
    }
    // Arrow cleanup can partially release reference-counted buffers before throwing. Retain
    // the owner and the original failure; repeating that cleanup could release them twice.
    closeFailure = failure;
    checkCloseFailure();
    owners.remove(this);
    LIVE_BATCHES.decrementAndGet();
    closeComplete = true;
  }

  synchronized void checkCloseFailure() {
    if (closeFailure instanceof Error) throw (Error) closeFailure;
    if (closeFailure instanceof RuntimeException) throw (RuntimeException) closeFailure;
    if (closeFailure != null) {
      throw new SparkJvmEngine.EngineException(13, "native Arrow batch cleanup failed", closeFailure);
    }
  }

  /** Arrow C Data Interface array descriptor; all vector offsets are zero for fresh writers. */
  public static final class Node {
    private final long length;
    private final long nullCount;
    private final long[] buffers;
    private final long[] bufferSizes;
    private final Node[] children;

    private Node(long length, long nullCount, long[] buffers, long[] bufferSizes, Node[] children) {
      this.length = length;
      this.nullCount = nullCount;
      this.buffers = buffers;
      this.bufferSizes = bufferSizes;
      this.children = children;
    }

    public long length() { return length; }
    public long nullCount() { return nullCount; }
    public long offset() { return 0; }
    public long[] buffers() { return buffers.clone(); }
    public long[] bufferSizes() { return bufferSizes.clone(); }
    public Node[] children() { return children.clone(); }
  }
}
