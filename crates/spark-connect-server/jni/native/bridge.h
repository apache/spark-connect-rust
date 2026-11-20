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

#ifndef SPARK_CONNECT_BRIDGE_H
#define SPARK_CONNECT_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

/* Keep synchronized with MAX_BYTES in jni/src/lib.rs and
 * SparkJvmEngine.MAX_IPC_BYTES in the Java adapter. */
#define SPARK_CONNECT_JNI_MAX_IPC_BYTES (64 * 1024 * 1024)

typedef struct {
  uint8_t *data;
  size_t len;
} BridgeBuffer;

void spark_connect_buffer_free(BridgeBuffer *buffer);

/* Typed query engine ABI; these handles never carry service RPC envelopes. */
/* Retain an attachment for a dedicated native worker. Release on the same OS
 * thread, in reverse order for nested attachments. This does not change its
 * class loader; individual engine calls still restore their scoped context.
 * Only a guard that attached the thread detaches it. A failed detach retains
 * the guard so its owner can retry. Callers that cannot retry must report the
 * failure; the guard and retained context remain allocated for safety. The
 * runtime must release guards before the hosting JVM is destroyed. */
void *spark_connect_engine_thread_attach(void *engine, BridgeBuffer *error);
int spark_connect_engine_thread_detach(void *attachment, BridgeBuffer *error);
void *spark_connect_engine_task_new(void *engine, BridgeBuffer *error);
void *spark_connect_engine_open_session(void *engine, const uint8_t *user, size_t user_len,
    const uint8_t *id, size_t id_len, void *task, BridgeBuffer *error);
void *spark_connect_engine_prepare(void *session, const uint8_t *relation, size_t len,
    void *task, BridgeBuffer *error);
/* Opcodes 0/1/2: schema/isLocal/isStreaming. Output: bool byte, then schema IPC.
 * Schema is always 0; streaming is currently 0 (streaming relations are rejected). */
int spark_connect_engine_analyze(void *session, int32_t opcode, const uint8_t *relation, size_t len,
    void *task, BridgeBuffer *out, BridgeBuffer *error);
int spark_connect_engine_schema(void *query, BridgeBuffer *out, BridgeBuffer *error);

typedef uint8_t *(*IpcAllocate)(void *context, size_t len);
/* One complete Arrow IPC stream and its row count. Returns 1 for a batch, 0 for EOF,
 * or -1 for error. After validating the batch, calls allocate synchronously once,
 * including for an empty batch. It must return a nonnull pointer writable for len
 * bytes, or NULL if allocation is refused or fails. C never retains or frees the
 * callback, context or destination. The caller owns the destination on success and
 * failure and must only consider its bytes initialized when this returns 1. The
 * row_count is zero on EOF/error. No callback is made for EOF or invalid batches. */
int spark_connect_engine_next_ipc(void *query, IpcAllocate allocate, void *context,
    int64_t *row_count, BridgeBuffer *error);
int spark_connect_engine_config(void *session, int32_t opcode, const uint8_t *fields, size_t len,
    void *task, BridgeBuffer *out, BridgeBuffer *error);
/* Cancellation retains the handle. Failure may mean JNI delivery failed or Java
 * partially cancelled; preserve the adapter's non-replay policy for the latter. */
int spark_connect_engine_cancel(void *task, BridgeBuffer *error);
void spark_connect_engine_free(void *handle);
/* Success consumes the handle. Failure retains its Java global reference and API ownership.
 * Retrying is safe only when the Java owner's close contract permits it; the supplied adapter
 * preserves non-retryable Java close errors rather than repeating partial side effects. */
int spark_connect_engine_free_checked(void *handle, BridgeBuffer *error);

int64_t spark_connect_engine_native_start(void *env, void *clazz, void *address, void *factory);
void *spark_connect_engine_native_address(void *env, void *clazz, int64_t id);
void spark_connect_engine_native_stop(void *env, void *clazz, int64_t id);

#endif
