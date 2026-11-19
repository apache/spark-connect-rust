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

/* Exercise native ownership and pre-JNI validation without starting a JVM. */
#undef NDEBUG
#include "../native/bridge.c"
#include <assert.h>
#include <pthread.h>
#include <stdarg.h>

/* The C bridge links to these Rust entry points; this test does not invoke them. */
int64_t spark_connect_engine_start(const uint8_t *p, size_t n, void *b, BridgeBuffer *e) {
  (void) p; (void) n; (void) b; (void) e; return 0;
}
int spark_connect_address(int64_t id, BridgeBuffer *o, BridgeBuffer *e) {
  (void) id; (void) o; (void) e; return -1;
}
int spark_connect_stop(int64_t id, BridgeBuffer *e) { (void) id; (void) e; return -1; }

static void expect_analysis_failure(void *session, void *task, int provide_output,
    const char *expected) {
  uint8_t untouched = 0x5a;
  BridgeBuffer out = {.data = &untouched, .len = 1};
  BridgeBuffer error = {0};
  assert(spark_connect_engine_analyze(session, 0, NULL, 0, task,
      provide_output ? &out : NULL, &error) == -1);
  if (provide_output) assert(out.data == NULL && out.len == 0);
  assert(untouched == 0x5a);
  assert(error.data != NULL && error.len == strlen(expected));
  assert(memcmp(error.data, expected, error.len) == 0);
  spark_connect_buffer_free(&error);
  assert(error.data == NULL && error.len == 0);
  spark_connect_buffer_free(&error);
  assert(error.data == NULL && error.len == 0);
}

static void analysis_preconditions(void) {
  /* No JavaVM exists: every rejection must occur before entering a JNI scope. */
  EngineApi api = {0};
  EngineApi other_api = {0};
  EngineHandle session = {.api = &api, .kind = ENGINE_SESSION};
  EngineHandle query = {.api = &api, .kind = ENGINE_QUERY};
  EngineHandle task = {.api = &api, .kind = ENGINE_TASK};
  EngineHandle other_task = {.api = &other_api, .kind = ENGINE_TASK};
  const char *invalid = "JVM_ENGINE_ERROR:13:invalid Java engine handle";
  expect_analysis_failure(NULL, &task, 1, invalid);
  expect_analysis_failure(&query, &task, 1, invalid);
  expect_analysis_failure(&session, NULL, 1, invalid);
  expect_analysis_failure(&session, &session, 1, invalid);
  expect_analysis_failure(&session, &other_task, 1,
      "JVM_ENGINE_ERROR:13:Java engine task belongs to another engine");
  expect_analysis_failure(&session, &task, 0,
      "JVM_ENGINE_ERROR:13:missing Java engine analysis buffer");
  puts("C_ANALYZE_PRECONDITIONS_OK: invalid handles and cross-engine tasks rejected before JNI; error buffers released");
}

/* A small JNI invocation table verifies native-worker ownership without a JVM. */
static int mock_attached;
static int mock_attaches;
static int mock_named_attaches;
static int mock_detaches;
static int mock_attach_result;
static int mock_detach_result;
static int mock_get_env_result;
static int mock_push_result;
static int mock_frame_depth;
static int mock_frame_capacities[256];
static int mock_frame_references[256];
static int mock_frame_exception_references[256];
static int mock_max_frame_depth;
static int mock_max_frame_references;
static int mock_max_exception_references;
static int mock_fail_frame_depth = -1;
static int mock_exception_pending;
static int mock_pushes;
static int mock_pops;
static jobject mock_loader;
static JNIEnv mock_env;
static char mock_worker_name[64];
static uintmax_t mock_worker_index;
static const char mock_exception_message[] = "injected child access failure";
static jchar mock_exception_chars[sizeof(mock_exception_message)];
static int mock_close_calls;
static int mock_close_failure;
static int mock_globals_deleted;
static int mock_loader_deletions;
static int mock_new_global_failure;
static int mock_next_failure;
static jobject mock_next_batch;
static jbyteArray mock_ipc_data;
static jlong mock_row_count;
static int mock_copy_failure;
static int mock_copy_calls;
static jbyte *mock_copy_destination;
static int mock_cancel_calls;
static int mock_cancel_failure;
static int mock_message_allocation_calls;
static int mock_throw_new_calls;
static int mock_throw_new_failure;
static const char *mock_expected_native_message;

#define MOCK_THROWABLE ((jthrowable) (uintptr_t) 20)
#define MOCK_THROWABLE_CLASS ((jclass) (uintptr_t) 21)
#define MOCK_DESCRIPTION ((jstring) (uintptr_t) 22)
#define MOCK_DESCRIBE ((jmethodID) (uintptr_t) 14)
#define MOCK_NEXT ((jmethodID) (uintptr_t) 15)
#define MOCK_DATA ((jmethodID) (uintptr_t) 16)
#define MOCK_CLOSE ((jmethodID) (uintptr_t) 17)
#define MOCK_CANCEL ((jmethodID) (uintptr_t) 18)
#define MOCK_MESSAGE_BYTES ((jbyteArray) (uintptr_t) 23)

typedef struct {
  jsize len;
  uint8_t *data;
} MockBytes;

static jobject mock_new_local(jobject object) {
  if (object == NULL) return NULL;
  assert(mock_frame_depth > 0);
  int *references = &mock_frame_references[mock_frame_depth - 1];
  (*references)++;
  assert(*references <= mock_frame_capacities[mock_frame_depth - 1]);
  if (*references > mock_max_frame_references) mock_max_frame_references = *references;
  if (object == MOCK_THROWABLE || object == MOCK_THROWABLE_CLASS
      || object == MOCK_DESCRIPTION) {
    int *exceptions = &mock_frame_exception_references[mock_frame_depth - 1];
    (*exceptions)++;
    assert(*exceptions <= 3);
    if (*exceptions > mock_max_exception_references)
      mock_max_exception_references = *exceptions;
  }
  return object;
}

static jint JNICALL mock_get_env(JavaVM *vm, void **out, jint version) {
  (void) vm;
  assert(version == JNI_VERSION_1_8);
  if (mock_get_env_result != JNI_OK) return mock_get_env_result;
  if (!mock_attached) return JNI_EDETACHED;
  *out = &mock_env;
  return JNI_OK;
}

static jint JNICALL mock_attach(JavaVM *vm, void **out, void *args) {
  (void) vm;
  if (args != NULL) {
    JavaVMAttachArgs *attachment = args;
    assert(attachment->version == JNI_VERSION_1_8);
    static const char prefix[] = "spark-connect-engine-worker-";
    assert(strncmp(attachment->name, prefix, sizeof(prefix) - 1) == 0);
    char *end;
    uintmax_t index = strtoumax(attachment->name + sizeof(prefix) - 1, &end, 10);
    assert(*end == '\0' && index > mock_worker_index);
    mock_worker_index = index;
    assert(strlen(attachment->name) < sizeof(mock_worker_name));
    strcpy(mock_worker_name, attachment->name);
    assert(attachment->group == NULL);
  }
  if (mock_attach_result != JNI_OK) return mock_attach_result;
  mock_attached = 1;
  mock_attaches++;
  if (args != NULL) mock_named_attaches++;
  *out = &mock_env;
  return JNI_OK;
}

static jint JNICALL mock_detach(JavaVM *vm) {
  (void) vm;
  if (mock_detach_result != JNI_OK) return mock_detach_result;
  assert(mock_attached && mock_frame_depth == 0);
  mock_attached = 0;
  mock_detaches++;
  return JNI_OK;
}

static jboolean JNICALL mock_exception_check(JNIEnv *env) {
  assert(env == &mock_env);
  return mock_exception_pending ? JNI_TRUE : JNI_FALSE;
}

static jthrowable JNICALL mock_exception_occurred(JNIEnv *env) {
  assert(env == &mock_env && mock_exception_pending);
  return mock_new_local(MOCK_THROWABLE);
}

static void JNICALL mock_exception_clear(JNIEnv *env) {
  assert(env == &mock_env && mock_exception_pending);
  mock_exception_pending = 0;
}

static jclass JNICALL mock_object_class(JNIEnv *env, jobject object) {
  assert(env == &mock_env && !mock_exception_pending && object == MOCK_THROWABLE);
  return mock_new_local(MOCK_THROWABLE_CLASS);
}

static jmethodID JNICALL mock_method_id(JNIEnv *env, jclass cls, const char *name,
    const char *signature) {
  assert(env == &mock_env && !mock_exception_pending && cls == MOCK_THROWABLE_CLASS);
  assert(strcmp(name, "toString") == 0);
  assert(strcmp(signature, "()Ljava/lang/String;") == 0);
  return MOCK_DESCRIBE;
}

static jsize JNICALL mock_string_length(JNIEnv *env, jstring string) {
  assert(env == &mock_env && !mock_exception_pending && string == MOCK_DESCRIPTION);
  return sizeof(mock_exception_message) - 1;
}

static const jchar *JNICALL mock_string_chars(JNIEnv *env, jstring string,
    jboolean *copied) {
  assert(env == &mock_env && !mock_exception_pending && string == MOCK_DESCRIPTION);
  for (size_t i = 0; i < sizeof(mock_exception_message); i++)
    mock_exception_chars[i] = (unsigned char) mock_exception_message[i];
  if (copied != NULL) *copied = JNI_FALSE;
  return mock_exception_chars;
}

static void JNICALL mock_release_string_chars(JNIEnv *env, jstring string,
    const jchar *chars) {
  assert(env == &mock_env && string == MOCK_DESCRIPTION && chars == mock_exception_chars);
}

static jint JNICALL mock_push_frame(JNIEnv *env, jint capacity) {
  assert(env == &mock_env && mock_attached && capacity > 0);
  mock_pushes++;
  if (mock_push_result != JNI_OK) return mock_push_result;
  if (mock_frame_depth == mock_fail_frame_depth) return JNI_ERR;
  assert(mock_frame_depth < 256);
  mock_frame_capacities[mock_frame_depth] = capacity;
  mock_frame_references[mock_frame_depth] = 0;
  mock_frame_exception_references[mock_frame_depth] = 0;
  mock_frame_depth++;
  if (mock_frame_depth > mock_max_frame_depth) mock_max_frame_depth = mock_frame_depth;
  return JNI_OK;
}

static jobject JNICALL mock_pop_frame(JNIEnv *env, jobject result) {
  assert(env == &mock_env && mock_frame_depth > 0 && result == NULL);
  assert(mock_frame_exception_references[mock_frame_depth - 1] == 0);
  mock_frame_depth--;
  mock_pops++;
  return NULL;
}

static jobject JNICALL mock_current_thread(JNIEnv *env, jclass cls, jmethodID method, ...) {
  (void) cls; (void) method;
  assert(env == &mock_env);
  return mock_new_local((jobject) (uintptr_t) 1);
}

static jobject JNICALL mock_call_object(JNIEnv *env, jobject object, jmethodID method, ...) {
  assert(env == &mock_env);
  if (method == MOCK_DESCRIBE) {
    assert(!mock_exception_pending && object == MOCK_THROWABLE);
    return mock_new_local(MOCK_DESCRIPTION);
  }
  if (method == MOCK_NEXT) {
    if (mock_next_failure) mock_exception_pending = 1;
    return mock_new_local(mock_next_batch);
  }
  if (method == MOCK_DATA) return mock_new_local((jobject) mock_ipc_data);
  return mock_new_local(mock_loader);
}

static void JNICALL mock_set_loader(JNIEnv *env, jobject object, jmethodID method, ...) {
  (void) object;
  assert(env == &mock_env);
  if (method == MOCK_CLOSE) {
    assert(!mock_exception_pending);
    mock_close_calls++;
    if (mock_close_failure) mock_exception_pending = 1;
    return;
  }
  if (method == MOCK_CANCEL) {
    assert(!mock_exception_pending);
    mock_cancel_calls++;
    if (mock_cancel_failure) mock_exception_pending = 1;
    return;
  }
  va_list args;
  va_start(args, method);
  mock_loader = va_arg(args, jobject);
  va_end(args);
}

static jobject JNICALL mock_new_global(JNIEnv *env, jobject object) {
  assert(env == &mock_env && object != NULL);
  return mock_new_global_failure ? NULL : object;
}

static void JNICALL mock_delete_global(JNIEnv *env, jobject object) {
  assert(env == &mock_env && object != NULL && mock_attached && mock_frame_depth > 0);
  assert(!mock_exception_pending);
  if (object == (jobject) (uintptr_t) 2) {
    /* Final teardown must restore the caller's loader before deleting ours. */
    assert(mock_loader == (jobject) (uintptr_t) 3);
    mock_loader_deletions++;
  }
  mock_globals_deleted++;
}

static jboolean JNICALL mock_same_object(JNIEnv *env, jobject left, jobject right) {
  assert(env == &mock_env);
  return left == right;
}

static void JNICALL mock_delete_local(JNIEnv *env, jobject object) {
  assert(env == &mock_env && object != NULL && mock_frame_depth > 0);
  assert(mock_frame_references[mock_frame_depth - 1] > 0);
  mock_frame_references[mock_frame_depth - 1]--;
  if (object == MOCK_THROWABLE || object == MOCK_THROWABLE_CLASS
      || object == MOCK_DESCRIPTION) {
    assert(mock_frame_exception_references[mock_frame_depth - 1] > 0);
    mock_frame_exception_references[mock_frame_depth - 1]--;
  }
}

static jlong JNICALL mock_batch_row_count(JNIEnv *env, jobject object, jmethodID method, ...) {
  (void) method;
  assert(env == &mock_env && object == mock_next_batch);
  return mock_row_count;
}

static jsize JNICALL mock_array_length(JNIEnv *env, jarray array) {
  assert(env == &mock_env && array != NULL);
  return ((MockBytes *) array)->len;
}

static void JNICALL mock_byte_array_region(JNIEnv *env, jbyteArray array,
    jsize start, jsize length, jbyte *out) {
  assert(env == &mock_env && array != NULL && start == 0);
  mock_copy_calls++;
  mock_copy_destination = out;
  if (mock_copy_failure) {
    mock_exception_pending = 1;
    return;
  }
  MockBytes *bytes = (MockBytes *) array;
  assert(bytes->len == length);
  memcpy(out, bytes->data, (size_t) length);
}

static void attachment_error(BridgeBuffer *error, const char *message) {
  assert(error->data != NULL);
  assert(error->len == strlen(message));
  assert(memcmp(error->data, message, error->len) == 0);
  spark_connect_buffer_free(error);
}

static void *detach_on_wrong_thread(void *attachment) {
  BridgeBuffer error = {0};
  assert(spark_connect_engine_thread_detach(attachment, &error) == -1);
  attachment_error(&error,
      "JVM_ENGINE_ERROR:13:Java worker attachment belongs to another thread");
  return NULL;
}

static jbyteArray JNICALL mock_fail_message_allocation(JNIEnv *env, jsize length) {
  (void) length;
  assert(env == &mock_env && !mock_exception_pending);
  mock_message_allocation_calls++;
  mock_exception_pending = 1;
  return NULL;
}

static jbyteArray JNICALL mock_allocate_message(JNIEnv *env, jsize length) {
  assert(env == &mock_env && !mock_exception_pending);
  assert(length == (jsize) strlen(mock_expected_native_message));
  return mock_new_local(MOCK_MESSAGE_BYTES);
}

static void JNICALL mock_copy_message(JNIEnv *env, jbyteArray array, jsize start,
    jsize length, const jbyte *data) {
  assert(env == &mock_env && !mock_exception_pending && array == MOCK_MESSAGE_BYTES);
  assert(start == 0 && length == (jsize) strlen(mock_expected_native_message));
  assert(memcmp(data, mock_expected_native_message, (size_t) length) == 0);
  /* Stop after capturing the exact UTF-8 payload; the fallback path is checked below. */
  mock_exception_pending = 1;
}

static jclass JNICALL mock_find_exception_class(JNIEnv *env, const char *name) {
  assert(env == &mock_env && !mock_exception_pending);
  assert(strcmp(name, "java/lang/IllegalStateException") == 0);
  return mock_new_local(MOCK_THROWABLE_CLASS);
}

static jint JNICALL mock_throw_new(JNIEnv *env, jclass type, const char *message) {
  assert(env == &mock_env && !mock_exception_pending && type == MOCK_THROWABLE_CLASS);
  assert(strcmp(message, "native Spark Connect operation failed") == 0);
  mock_throw_new_calls++;
  mock_exception_pending = 1;
  return mock_throw_new_failure ? JNI_ERR : JNI_OK;
}

static JavaVM *mock_vm(void) {
  static const struct JNIInvokeInterface_ invocation = {
    .GetEnv = mock_get_env,
    .AttachCurrentThreadAsDaemon = mock_attach,
    .DetachCurrentThread = mock_detach,
  };
  static const struct JNINativeInterface_ native = {
    .ExceptionCheck = mock_exception_check,
    .ExceptionOccurred = mock_exception_occurred,
    .ExceptionClear = mock_exception_clear,
    .GetObjectClass = mock_object_class,
    .GetMethodID = mock_method_id,
    .GetStringLength = mock_string_length,
    .GetStringChars = mock_string_chars,
    .ReleaseStringChars = mock_release_string_chars,
    .PushLocalFrame = mock_push_frame,
    .PopLocalFrame = mock_pop_frame,
    .CallStaticObjectMethod = mock_current_thread,
    .CallObjectMethod = mock_call_object,
    .CallVoidMethod = mock_set_loader,
    .IsSameObject = mock_same_object,
    .DeleteLocalRef = mock_delete_local,
    .NewGlobalRef = mock_new_global,
    .DeleteGlobalRef = mock_delete_global,
    .CallLongMethod = mock_batch_row_count,
    .GetArrayLength = mock_array_length,
    .GetByteArrayRegion = mock_byte_array_region,
  };
  mock_env = &native;
  static JavaVM vm = &invocation;
  return &vm;
}

static void worker_attachments(void) {
  JavaContext context = {.vm = mock_vm(), .loader = (jobject) (uintptr_t) 2};
  /* Two external owners prevent stack-backed context finalization in this test. */
  atomic_init(&context.references, 2);
  EngineApi api = {.context = &context};
  EngineHandle factory = {.api = &api, .kind = ENGINE_FACTORY};
  EngineHandle query = {.api = &api, .kind = ENGINE_QUERY};
  BridgeBuffer error = {0};
  const char *invalid = "JVM_ENGINE_ERROR:13:invalid Java engine handle";
  assert(spark_connect_engine_thread_attach(NULL, &error) == NULL);
  attachment_error(&error, invalid);
  assert(spark_connect_engine_thread_attach(&query, &error) == NULL);
  attachment_error(&error, invalid);
  assert(mock_attaches == 0);

  mock_attach_result = JNI_ERR;
  assert(spark_connect_engine_thread_attach(&factory, &error) == NULL);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:cannot attach Java engine worker");
  assert(atomic_load(&context.references) == 2 && mock_attaches == 0);
  mock_attach_result = JNI_OK;
  mock_get_env_result = JNI_EVERSION;
  assert(spark_connect_engine_thread_attach(&factory, &error) == NULL);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:cannot access Java engine worker context");
  assert(atomic_load(&context.references) == 2);
  mock_get_env_result = JNI_OK;

  void *outer = spark_connect_engine_thread_attach(&factory, &error);
  assert(outer != NULL && error.data == NULL);
  assert(mock_attaches == 1 && mock_named_attaches == 1);
  char first_worker_name[sizeof(mock_worker_name)];
  strcpy(first_worker_name, mock_worker_name);
  assert(atomic_load(&context.references) == 3);
  void *inner = spark_connect_engine_thread_attach(&factory, &error);
  assert(inner != NULL && mock_attaches == 1);
  assert(atomic_load(&context.references) == 4);
  assert(spark_connect_engine_thread_detach(inner, &error) == 0);
  assert(mock_detaches == 0 && atomic_load(&context.references) == 3);

  mock_loader = (jobject) (uintptr_t) 3;
  JavaScope first, second;
  assert(scope_enter(&context, &first, &error) == 0);
  assert(mock_loader == context.loader && mock_frame_depth == 1);
  assert(scope_enter(&context, &second, &error) == 0);
  assert(mock_frame_depth == 2 && mock_attaches == 1);
  scope_leave(&second);
  assert(mock_frame_depth == 1 && mock_loader == context.loader);
  scope_leave(&first);
  assert(mock_frame_depth == 0 && mock_loader == (jobject) (uintptr_t) 3);
  assert(mock_pushes == 2 && mock_pops == 2 && mock_detaches == 0);

  mock_push_result = JNI_ERR;
  assert(scope_enter(&context, &first, &error) == -1);
  attachment_error(&error, "cannot allocate Java local reference frame");
  assert(mock_frame_depth == 0 && mock_pops == 2 && mock_detaches == 0);
  mock_push_result = JNI_OK;

  pthread_t other;
  assert(pthread_create(&other, NULL, detach_on_wrong_thread, outer) == 0);
  assert(pthread_join(other, NULL) == 0);
  assert(mock_detaches == 0 && atomic_load(&context.references) == 3);
  mock_detach_result = JNI_ERR;
  assert(spark_connect_engine_thread_detach(outer, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:cannot detach Java engine worker");
  assert(mock_detaches == 0 && atomic_load(&context.references) == 3);
  mock_detach_result = JNI_OK;
  assert(spark_connect_engine_thread_detach(outer, &error) == 0);
  assert(mock_detaches == 1 && atomic_load(&context.references) == 2);
  assert(spark_connect_engine_thread_detach(NULL, &error) == 0);

  /* An attachment borrowed from Java must remain attached after guard release. */
  mock_attached = 1;
  outer = spark_connect_engine_thread_attach(&factory, &error);
  assert(outer != NULL && mock_attaches == 1);
  assert(spark_connect_engine_thread_detach(outer, &error) == 0);
  assert(mock_attached && mock_detaches == 1 && atomic_load(&context.references) == 2);
  mock_attached = 0;

  /* A new native attachment gets a distinct name, copied by the VM on entry. */
  outer = spark_connect_engine_thread_attach(&factory, &error);
  assert(outer != NULL && error.data == NULL);
  assert(mock_attaches == 2 && mock_named_attaches == 2);
  assert(strcmp(first_worker_name, mock_worker_name) != 0);
  assert(spark_connect_engine_thread_detach(outer, &error) == 0);
  assert(mock_detaches == 2 && atomic_load(&context.references) == 2);

  /* The existing non-persistent path still balances attachment and local frames. */
  assert(scope_enter(&context, &first, &error) == 0);
  assert(mock_attaches == 3 && mock_frame_depth == 1);
  scope_leave(&first);
  assert(mock_detaches == 3 && mock_frame_depth == 0 && mock_pushes == mock_pops + 1);
  assert(mock_named_attaches == 2);
  puts("C_WORKER_ATTACHMENTS_OK: nested ownership, same-thread release, scoped frames/loaders, failure retry");
}

static EngineHandle *mock_handle(EngineKind kind) {
  JavaContext *context = calloc(1, sizeof(*context));
  EngineApi *api = calloc(1, sizeof(*api));
  EngineHandle *handle = calloc(1, sizeof(*handle));
  assert(context != NULL && api != NULL && handle != NULL);
  context->vm = mock_vm();
  context->loader = (jobject) (uintptr_t) 2;
  context->thread_class = (jclass) (uintptr_t) 5;
  atomic_init(&context->references, 1);
  api->context = context;
  atomic_init(&api->references, 1);
  for (int i = 0; i < ENGINE_EXCEPTION; i++) api->close[i] = MOCK_CLOSE;
  api->next_ipc = MOCK_NEXT;
  api->batch_data = MOCK_DATA;
  api->batch_row_count = (jmethodID) (uintptr_t) 11;
  api->cancel = MOCK_CANCEL;
  handle->api = api;
  handle->kind = kind;
  handle->object = (jobject) (uintptr_t) 30;
  mock_loader = (jobject) (uintptr_t) 3;
  mock_close_calls = mock_globals_deleted = mock_loader_deletions = 0;
  mock_close_failure = mock_new_global_failure = mock_next_failure = 0;
  mock_next_batch = NULL;
  mock_ipc_data = NULL;
  mock_row_count = 0;
  mock_copy_failure = 0;
  mock_copy_calls = 0;
  mock_copy_destination = NULL;
  mock_cancel_calls = mock_cancel_failure = 0;
  assert(!mock_attached && mock_frame_depth == 0 && !mock_exception_pending);
  return handle;
}

static void checked_cleanup_ownership(void) {
  EngineHandle *handle = mock_handle(ENGINE_FACTORY);
  EngineApi *api = handle->api;
  JavaContext *context = api->context;
  BridgeBuffer error = {0};
  int attaches = mock_attaches;
  mock_attach_result = JNI_ERR;
  assert(spark_connect_engine_free_checked(handle, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:cannot attach Rust worker to JVM");
  assert(mock_close_calls == 0 && mock_globals_deleted == 0);
  assert(atomic_load(&api->references) == 1 && atomic_load(&context->references) == 1);
  mock_attach_result = JNI_OK;

  mock_push_result = JNI_ERR;
  assert(spark_connect_engine_free_checked(handle, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:cannot allocate Java local reference frame");
  assert(mock_close_calls == 0 && mock_globals_deleted == 0);
  assert(atomic_load(&api->references) == 1 && atomic_load(&context->references) == 1);
  mock_push_result = JNI_OK;

  mock_close_failure = 1;
  assert(spark_connect_engine_free_checked(handle, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:injected child access failure");
  assert(mock_close_calls == 1 && mock_globals_deleted == 0);
  assert(handle->api == api && handle->object == (jobject) (uintptr_t) 30);
  assert(atomic_load(&api->references) == 1 && atomic_load(&context->references) == 1);
  assert(!mock_attached && mock_frame_depth == 0);
  mock_close_failure = 0;
  assert(spark_connect_engine_free_checked(handle, &error) == 0 && error.data == NULL);
  assert(mock_close_calls == 2 && mock_globals_deleted == 3 && mock_loader_deletions == 1);
  /* Each attempt has one scope; final global deletion must not reattach. */
  assert(mock_attaches == attaches + 3 && !mock_attached && mock_frame_depth == 0);

  handle = mock_handle(ENGINE_FACTORY);
  context = handle->api->context;
  void *worker = spark_connect_engine_thread_attach(handle, &error);
  assert(worker != NULL && atomic_load(&context->references) == 2);
  assert(spark_connect_engine_free_checked(handle, &error) == -1);
  attachment_error(&error,
      "JVM_ENGINE_ERROR:13:Java engine worker attachments remain during factory cleanup");
  assert(mock_close_calls == 0 && mock_globals_deleted == 0);
  assert(spark_connect_engine_thread_detach(worker, &error) == 0);
  assert(atomic_load(&context->references) == 1);
  assert(spark_connect_engine_free_checked(handle, &error) == 0);
  assert(mock_close_calls == 1 && mock_globals_deleted == 3);

  puts("C_CHECKED_CLEANUP_OK: attach/Java failures retain owners; worker guards prevent factory free; final globals use the original scope");
}

typedef struct {
  uint8_t data[4];
  size_t len;
  int calls;
  int fail;
} MockIpcDestination;

static uint8_t *mock_allocate_ipc(void *context, size_t len) {
  MockIpcDestination *destination = context;
  assert(destination != NULL);
  destination->calls++;
  destination->len = len;
  if (destination->fail) return NULL;
  assert(len <= sizeof(destination->data));
  return destination->data;
}

static void ipc_transfer(void) {
  EngineHandle *query = mock_handle(ENGINE_QUERY);
  BridgeBuffer error = {0};
  MockIpcDestination destination = {0};
  int64_t count = -1;
  assert(spark_connect_engine_next_ipc(NULL, mock_allocate_ipc, &destination,
      &count, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:invalid Java engine handle");
  assert(destination.calls == 0 && count == 0);
  assert(spark_connect_engine_next_ipc(query, NULL, &destination, &count, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:missing Java engine IPC destination");
  assert(spark_connect_engine_next_ipc(query, mock_allocate_ipc, &destination,
      NULL, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:missing Java engine IPC destination");
  mock_attach_result = JNI_ERR;
  assert(spark_connect_engine_next_ipc(query, mock_allocate_ipc, &destination,
      &count, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:cannot attach Rust worker to JVM");
  mock_attach_result = JNI_OK;

  assert(spark_connect_engine_next_ipc(query, mock_allocate_ipc, &destination,
      &count, &error) == 0);
  assert(error.data == NULL && destination.calls == 0 && count == 0);
  mock_next_failure = 1;
  assert(spark_connect_engine_next_ipc(query, mock_allocate_ipc, &destination,
      &count, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:injected child access failure");
  assert(destination.calls == 0 && mock_copy_calls == 0 && count == 0);
  mock_next_failure = 0;

  uint8_t data[] = {0, 255, 3, 7};
  MockBytes bytes = {.len = sizeof(data), .data = data};
  mock_next_batch = (jobject) (uintptr_t) 31;
  mock_ipc_data = (jbyteArray) &bytes;
  mock_row_count = 2;
  assert(spark_connect_engine_next_ipc(query, mock_allocate_ipc, &destination,
      &count, &error) == 1);
  assert(error.data == NULL && destination.len == sizeof(data) && count == 2);
  assert(destination.calls == 1 && mock_copy_calls == 1);
  assert(mock_copy_destination == (jbyte *) destination.data);
  assert(memcmp(destination.data, data, sizeof(data)) == 0);
  /* The returned copy remains valid independently of Java's next-batch storage. */
  data[0] = 99;
  assert(destination.data[0] == 0);

  bytes.len = 0;
  mock_row_count = 0;
  assert(spark_connect_engine_next_ipc(query, mock_allocate_ipc, &destination,
      &count, &error) == 1);
  assert(error.data == NULL && destination.len == 0 && count == 0);
  assert(destination.calls == 2 && mock_copy_calls == 1);
  bytes.len = sizeof(data);

  mock_row_count = -1;
  assert(spark_connect_engine_next_ipc(query, mock_allocate_ipc, &destination,
      &count, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:negative Java engine IPC row count");
  assert(destination.calls == 2 && mock_copy_calls == 1 && count == 0);
  mock_row_count = 0;
  bytes.len = 64 * 1024 * 1024 + 1;
  assert(spark_connect_engine_next_ipc(query, mock_allocate_ipc, &destination,
      &count, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:8:Spark Arrow batch exceeds JNI limit");
  assert(destination.calls == 2 && mock_copy_calls == 1 && count == 0);
  bytes.len = sizeof(data);
  mock_ipc_data = NULL;
  assert(spark_connect_engine_next_ipc(query, mock_allocate_ipc, &destination,
      &count, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:Java engine returned no IPC data");
  assert(destination.calls == 2 && mock_copy_calls == 1 && count == 0);
  mock_ipc_data = (jbyteArray) &bytes;

  destination.fail = 1;
  count = -1;
  assert(spark_connect_engine_next_ipc(query, mock_allocate_ipc, &destination,
      &count, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:8:cannot allocate Rust IPC buffer");
  assert(destination.calls == 3 && mock_copy_calls == 1 && count == 0);
  assert(mock_frame_depth == 0 && !mock_exception_pending);
  destination.fail = 0;

  mock_copy_failure = 1;
  assert(spark_connect_engine_next_ipc(query, mock_allocate_ipc, &destination,
      &count, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:injected child access failure");
  assert(destination.calls == 4 && mock_copy_calls == 2 && count == 0);
  assert(mock_copy_destination == (jbyte *) destination.data);
  assert(mock_frame_depth == 0 && !mock_exception_pending);
  mock_copy_failure = 0;
  assert(spark_connect_engine_free_checked(query, &error) == 0);
  puts("C_IPC_TRANSFER_OK: direct caller-owned destination, empty batches, allocation/copy errors, and local-reference cleanup");
}

static void checked_cancellation(void) {
  EngineHandle *task = mock_handle(ENGINE_TASK);
  EngineApi *api = task->api;
  JavaContext *context = api->context;
  BridgeBuffer error = {0};
  assert(spark_connect_engine_cancel(NULL, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:invalid Java engine handle");
  EngineHandle wrong_kind = {.api = api, .kind = ENGINE_QUERY};
  assert(spark_connect_engine_cancel(&wrong_kind, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:invalid Java engine handle");
  assert(mock_cancel_calls == 0);

  mock_attach_result = JNI_ERR;
  assert(spark_connect_engine_cancel(task, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:cannot attach Rust worker to JVM");
  assert(mock_cancel_calls == 0 && mock_globals_deleted == 0);
  assert(atomic_load(&api->references) == 1 && atomic_load(&context->references) == 1);
  mock_attach_result = JNI_OK;
  // A failed delivery retains the same task, allowing a later caller to retry.
  assert(spark_connect_engine_cancel(task, &error) == 0 && error.data == NULL);
  assert(mock_cancel_calls == 1);

  mock_push_result = JNI_ERR;
  assert(spark_connect_engine_cancel(task, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:cannot allocate Java local reference frame");
  assert(mock_cancel_calls == 1);
  mock_push_result = JNI_OK;
  mock_cancel_failure = 1;
  assert(spark_connect_engine_cancel(task, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:injected child access failure");
  assert(mock_cancel_calls == 2 && mock_globals_deleted == 0);
  assert(!mock_attached && mock_frame_depth == 0 && !mock_exception_pending);
  assert(atomic_load(&api->references) == 1 && atomic_load(&context->references) == 1);
  mock_cancel_failure = 0;
  assert(spark_connect_engine_free_checked(task, &error) == 0);
  puts("C_CHECKED_CANCEL_OK: failed delivery and Java exceptions reported without consuming task ownership");
}

static void throw_error_allocation_failure(void) {
  (void) mock_vm();
  JNIEnv saved_env = mock_env;
  struct JNINativeInterface_ error_native = *mock_env;
  // Install these restrictive fault mocks only for the throw_error cases.
  error_native.NewByteArray = mock_allocate_message;
  error_native.SetByteArrayRegion = mock_copy_message;
  error_native.FindClass = mock_find_exception_class;
  error_native.ThrowNew = mock_throw_new;
  mock_env = &error_native;
  assert(!mock_attached && mock_frame_depth == 0 && !mock_exception_pending);
  mock_attached = 1;
  assert(mock_push_frame(&mock_env, 32) == JNI_OK);
  const char *messages[][2] = {
    {"plain startup failure", "plain startup failure"},
    {"JVM_ENGINE_ERROR:3:invalid address", "invalid address"},
    {"JVM_ENGINE_ERROR:13:cannot initialize engine", "cannot initialize engine"},
    {"JVM_ENGINE_ERROR:16:message:with:colons", "message:with:colons"},
    {"JVM_ENGINE_ERROR:0:invalid code", "JVM_ENGINE_ERROR:0:invalid code"},
    {"JVM_ENGINE_ERROR:17:invalid code", "JVM_ENGINE_ERROR:17:invalid code"},
    {"JVM_ENGINE_ERROR:123:invalid code", "JVM_ENGINE_ERROR:123:invalid code"},
    {"JVM_ENGINE_ERROR:text:literal prefix", "JVM_ENGINE_ERROR:text:literal prefix"},
  };
  for (size_t i = 0; i < sizeof(messages) / sizeof(messages[0]); i++) {
    BridgeBuffer error = {0};
    set_error(&error, messages[i][0]);
    mock_expected_native_message = messages[i][1];
    throw_error(&mock_env, &error);
    assert(error.data == NULL && error.len == 0 && mock_exception_pending);
    assert(mock_frame_references[mock_frame_depth - 1] == 0);
    mock_exception_clear(&mock_env);
  }
  const int message_checks = mock_throw_new_calls;
  error_native.NewByteArray = mock_fail_message_allocation;
  for (int failure = 0; failure < 2; failure++) {
    BridgeBuffer error = {0};
    set_error(&error, "injected native stop failure");
    mock_throw_new_failure = failure;
    throw_error(&mock_env, &error);
    assert(error.data == NULL && error.len == 0);
    // Both the fallback throwable and a fallback allocation failure stay pending.
    assert(mock_exception_pending);
    assert(mock_frame_references[mock_frame_depth - 1] == 0);
    mock_exception_clear(&mock_env);
  }
  assert(mock_message_allocation_calls == 2 && mock_throw_new_calls == message_checks + 2);
  // An already pending exception is preserved without allocating another message.
  BridgeBuffer error = {0};
  set_error(&error, "injected native stop failure");
  mock_exception_pending = 1;
  throw_error(&mock_env, &error);
  assert(mock_exception_pending && error.data == NULL && error.len == 0);
  assert(mock_message_allocation_calls == 2 && mock_throw_new_calls == message_checks + 2);
  mock_exception_clear(&mock_env);
  mock_throw_new_failure = 0;
  mock_pop_frame(&mock_env, NULL);
  mock_attached = 0;
  mock_env = saved_env;
  puts("C_THROW_ERROR_OK: engine framing removed from Java messages; allocation fallback and pending exceptions preserved");
}

int main(void) {
  analysis_preconditions();
  worker_attachments();
  checked_cleanup_ownership();
  ipc_transfer();
  checked_cancellation();
  throw_error_allocation_failure();
  return 0;
}
