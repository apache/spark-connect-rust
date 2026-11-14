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

/* Arrow C Data Interface ABI. These descriptors never own/copy value buffers. */
/* Keep in sync with MAX_ARROW_NODES in src/engine/zero_copy.rs. */
#define SPARK_CONNECT_MAX_ARROW_NODES 65536

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE
struct ArrowSchema {
  const char *format;
  const char *name;
  const char *metadata;
  int64_t flags;
  int64_t n_children;
  struct ArrowSchema **children;
  struct ArrowSchema *dictionary;
  void (*release)(struct ArrowSchema *);
  void *private_data;
};
struct ArrowArray {
  int64_t length;
  int64_t null_count;
  int64_t offset;
  int64_t n_buffers;
  int64_t n_children;
  const void **buffers;
  struct ArrowArray **children;
  struct ArrowArray *dictionary;
  void (*release)(struct ArrowArray *);
  void *private_data;
};
#endif

/* A non-null lease/callback pair transfers ownership on every outcome, including EOF/error.
 * defer_release runs on the final consumer thread (or immediately when no array is returned)
 * and must queue owner disposal. With no Java batch, owner may be NULL or a non-NULL
 * owner without a batch; either form still consumes the lease exactly once.
 * Checked cleanup retains the owner on failure, so its caller can retry without losing the lease. */
int spark_connect_engine_next_arrow(void *query, struct ArrowArray *out, void *lease,
    void (*defer_release)(void *lease, void *owner), BridgeBuffer *error);
int64_t spark_connect_engine_arrow_buffer_size(const struct ArrowArray *array, int64_t index);
void spark_connect_engine_arrow_owner_free(void *owner);
int spark_connect_engine_arrow_owner_free_checked(void *owner, BridgeBuffer *error);
int spark_connect_engine_config(void *session, int32_t opcode, const uint8_t *fields, size_t len,
    void *task, BridgeBuffer *out, BridgeBuffer *error);
void spark_connect_engine_cancel(void *task);
void spark_connect_engine_free(void *handle);
/* Success consumes the handle. Failure retains its Java global reference and API ownership.
 * Retrying is safe only when the Java owner's close contract permits it; the supplied adapter
 * preserves non-retryable Java close errors rather than repeating partial side effects. */
int spark_connect_engine_free_checked(void *handle, BridgeBuffer *error);

int64_t spark_connect_engine_native_start(void *env, void *clazz, void *address, void *factory);
void *spark_connect_engine_native_address(void *env, void *clazz, int64_t id);
void spark_connect_engine_native_stop(void *env, void *clazz, int64_t id);

#endif
