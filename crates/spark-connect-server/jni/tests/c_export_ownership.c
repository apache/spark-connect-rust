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

static atomic_int finalizations;
static void *released_owner;

static void defer(void *lease, void *owner) {
  assert(lease == &finalizations);
  released_owner = owner;
  atomic_fetch_add(&finalizations, 1);
}

static ArrowExportOwner *owner(unsigned refs) {
  ArrowExportOwner *result = calloc(1, sizeof(*result));
  assert(result != NULL);
  atomic_init(&result->references, refs);
  result->lease = &finalizations;
  result->defer_release = defer;
  return result;
}

static struct ArrowArray node(ArrowExportOwner *shared, int64_t *values) {
  struct ArrowArray result = {0};
  result.length = 3;
  result.n_buffers = 2;
  result.buffers = calloc(2, sizeof(*result.buffers));
  assert(result.buffers != NULL);
  result.buffers[1] = values;
  ArrowNodePrivate *private = calloc(1, sizeof(*private));
  assert(private != NULL);
  private->owner = shared;
  private->capacities = calloc(2, sizeof(*private->capacities));
  assert(private->capacities != NULL);
  private->capacities[1] = 3 * sizeof(*values);
  result.private_data = private;
  result.release = arrow_node_release;
  return result;
}

static void *release_on_worker(void *pointer) {
  struct ArrowArray *array = pointer;
  array->release(array);
  return NULL;
}

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
static int mock_fail_array_depth = -1;
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
static jobject mock_descriptor;

#define MOCK_THROWABLE ((jthrowable) (uintptr_t) 20)
#define MOCK_THROWABLE_CLASS ((jclass) (uintptr_t) 21)
#define MOCK_DESCRIPTION ((jstring) (uintptr_t) 22)
#define MOCK_DESCRIBE ((jmethodID) (uintptr_t) 14)
#define MOCK_NEXT ((jmethodID) (uintptr_t) 15)
#define MOCK_DESCRIPTOR ((jmethodID) (uintptr_t) 16)
#define MOCK_CLOSE ((jmethodID) (uintptr_t) 17)

typedef struct MockNode MockNode;
typedef struct {
  jsize count;
  MockNode *child;
} MockArray;
struct MockNode {
  int valid;
  MockArray buffers;
  MockArray capacities;
  MockArray children;
};

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
  if (method == MOCK_DESCRIPTOR) return mock_new_local(mock_descriptor);
  if (method == (jmethodID) (uintptr_t) 11)
    return mock_new_local((jobject) &((MockNode *) object)->buffers);
  if (method == (jmethodID) (uintptr_t) 12)
    return mock_new_local((jobject) &((MockNode *) object)->capacities);
  if (method == (jmethodID) (uintptr_t) 13)
    return mock_new_local((jobject) &((MockNode *) object)->children);
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

static jboolean JNICALL mock_node_instance(JNIEnv *env, jobject object, jclass cls) {
  assert(env == &mock_env && object != NULL && cls == (jclass) (uintptr_t) 4);
  return ((MockNode *) object)->valid;
}

static jlong JNICALL mock_node_long(JNIEnv *env, jobject object, jmethodID method, ...) {
  (void) method;
  assert(env == &mock_env && object != NULL);
  return 0;
}

static jsize JNICALL mock_array_length(JNIEnv *env, jarray array) {
  assert(env == &mock_env && array != NULL);
  return ((MockArray *) array)->count;
}

static jobject JNICALL mock_array_child(JNIEnv *env, jobjectArray array, jsize index) {
  assert(env == &mock_env && array != NULL && index == 0);
  MockArray *children = (MockArray *) array;
  assert(children->count == 1 && children->child != NULL);
  if (mock_frame_depth == mock_fail_array_depth) {
    assert(!mock_exception_pending);
    mock_exception_pending = 1;
    return NULL;
  }
  return mock_new_local((jobject) children->child);
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
    .IsInstanceOf = mock_node_instance,
    .CallLongMethod = mock_node_long,
    .GetArrayLength = mock_array_length,
    .GetObjectArrayElement = mock_array_child,
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

static void descriptor_reference_frames(void) {
  EngineApi api = {0};
  api.classes[ENGINE_NODE] = (jclass) (uintptr_t) 4;
  api.node_buffers = (jmethodID) (uintptr_t) 11;
  api.node_buffer_sizes = (jmethodID) (uintptr_t) 12;
  api.node_children = (jmethodID) (uintptr_t) 13;
  MockNode nodes[130] = {0};
  for (size_t i = 0; i < 130; i++) {
    nodes[i].valid = 1;
    if (i + 1 < 130) {
      nodes[i].children.count = 1;
      nodes[i].children.child = &nodes[i + 1];
    }
  }
  ArrowExportOwner owner = {0};
  atomic_init(&owner.references, 1);
  struct ArrowArray out = {0};
  BridgeBuffer error = {0};
  mock_attached = 1;
  mock_max_frame_depth = mock_max_frame_references = 0;
  assert(mock_push_frame(&mock_env, 32) == JNI_OK);
  /* Model the outer scope's thread, loader, batch and root-node references. */
  for (int i = 0; i < 4; i++) mock_new_local((jobject) (uintptr_t) 1);

  nodes[128].children.count = 0;
  size_t remaining = SPARK_CONNECT_MAX_ARROW_NODES;
  assert(arrow_build_node(&mock_env, &api, (jobject) &nodes[0], &owner,
      &out, 0, &remaining, &error) == 0);
  assert(error.data == NULL && remaining == SPARK_CONNECT_MAX_ARROW_NODES - 129);
  assert(mock_frame_depth == 1 && mock_max_frame_depth == 130);
  assert(mock_max_frame_references == 4 && atomic_load(&owner.references) == 130);
  out.release(&out);
  assert(atomic_load(&owner.references) == 1);

  /* Exhausting the node budget must unwind descriptors independently of depth. */
  memset(&out, 0, sizeof(out));
  remaining = 128;
  assert(arrow_build_node(&mock_env, &api, (jobject) &nodes[0], &owner,
      &out, 0, &remaining, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:invalid or excessive Java Arrow descriptor tree");
  assert(remaining == 0 && mock_frame_depth == 1 && out.release == NULL);
  assert(atomic_load(&owner.references) == 1);

  /* A rejected child at depth 129 unwinds all frames and partial owners. */
  nodes[128].children.count = 1;
  memset(&out, 0, sizeof(out));
  remaining = SPARK_CONNECT_MAX_ARROW_NODES;
  assert(arrow_build_node(&mock_env, &api, (jobject) &nodes[0], &owner,
      &out, 0, &remaining, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:invalid or excessive Java Arrow descriptor tree");
  assert(mock_frame_depth == 1 && out.release == NULL);
  assert(atomic_load(&owner.references) == 1);

  /* Frame-allocation failure at an interior node must also unwind cleanly. */
  mock_fail_frame_depth = 7;
  memset(&out, 0, sizeof(out));
  remaining = SPARK_CONNECT_MAX_ARROW_NODES;
  assert(arrow_build_node(&mock_env, &api, (jobject) &nodes[0], &owner,
      &out, 0, &remaining, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:cannot reserve Java Arrow descriptor references");
  assert(mock_frame_depth == 1 && out.release == NULL);
  assert(atomic_load(&owner.references) == 1);
  mock_fail_frame_depth = -1;

  /* An interior JNI exception needs all three temporary exception references
   * beside the descriptor arrays, then unwinds every partial owner and frame. */
  mock_fail_array_depth = 7;
  mock_max_frame_references = mock_max_exception_references = 0;
  int before_pushes = mock_pushes;
  int before_pops = mock_pops;
  memset(&out, 0, sizeof(out));
  remaining = SPARK_CONNECT_MAX_ARROW_NODES;
  assert(arrow_build_node(&mock_env, &api, (jobject) &nodes[0], &owner,
      &out, 0, &remaining, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:injected child access failure");
  assert(!mock_exception_pending && mock_max_exception_references == 3);
  assert(mock_max_frame_references == 6);
  assert(mock_pushes - before_pushes == 6 && mock_pops - before_pops == 6);
  assert(mock_frame_depth == 1 && out.release == NULL);
  assert(atomic_load(&owner.references) == 1);
  mock_fail_array_depth = -1;

  nodes[0].valid = 0;
  remaining = SPARK_CONNECT_MAX_ARROW_NODES;
  assert(arrow_build_node(&mock_env, &api, (jobject) &nodes[0], &owner,
      &out, 0, &remaining, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:invalid Java Arrow node type");
  assert(mock_frame_depth == 1 && atomic_load(&owner.references) == 1);
  mock_pop_frame(&mock_env, NULL);
  assert(mock_frame_depth == 0);
  mock_attached = 0;
  puts("C_ARROW_LOCAL_FRAMES_OK: depth128 bounded refs; depth, frame allocation and JNI exception failures unwind owners");
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
  api->classes[ENGINE_NODE] = (jclass) (uintptr_t) 4;
  for (int i = 0; i < ENGINE_EXCEPTION; i++) api->close[i] = MOCK_CLOSE;
  api->next_arrow = MOCK_NEXT;
  api->descriptor = MOCK_DESCRIPTOR;
  api->node_buffers = (jmethodID) (uintptr_t) 11;
  api->node_buffer_sizes = (jmethodID) (uintptr_t) 12;
  api->node_children = (jmethodID) (uintptr_t) 13;
  handle->api = api;
  handle->kind = kind;
  handle->object = (jobject) (uintptr_t) 30;
  mock_loader = (jobject) (uintptr_t) 3;
  mock_close_calls = mock_globals_deleted = mock_loader_deletions = 0;
  mock_close_failure = mock_new_global_failure = mock_next_failure = 0;
  mock_next_batch = mock_descriptor = NULL;
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
  assert(mock_close_calls == 2 && mock_globals_deleted == 4 && mock_loader_deletions == 1);
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
  assert(mock_close_calls == 1 && mock_globals_deleted == 4);

  ArrowExportOwner *shared = owner(1);
  shared->batch = mock_handle(ENGINE_BATCH);
  mock_close_failure = 1;
  assert(spark_connect_engine_arrow_owner_free_checked(shared, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:injected child access failure");
  assert(shared->batch != NULL && mock_close_calls == 1 && mock_globals_deleted == 0);
  mock_close_failure = 0;
  assert(spark_connect_engine_arrow_owner_free_checked(shared, &error) == 0);
  assert(mock_close_calls == 2 && mock_globals_deleted == 4);
  puts("C_CHECKED_CLEANUP_OK: attach/Java failures retain owners; worker guards prevent factory free; final globals use the original scope");
}

static void arrow_export_error_ownership(void) {
  EngineHandle *query = mock_handle(ENGINE_QUERY);
  BridgeBuffer error = {0};
  struct ArrowArray out = {0};
  atomic_store(&finalizations, 0);
  released_owner = (void *) (uintptr_t) 1;
  assert(spark_connect_engine_next_arrow(NULL, &out, &finalizations, defer, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:invalid Java engine handle");
  assert(atomic_load(&finalizations) == 1 && released_owner == NULL);
  assert(spark_connect_engine_next_arrow(query, NULL, &finalizations, defer, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:invalid Java Arrow export destination or lease");
  assert(atomic_load(&finalizations) == 2 && released_owner == NULL);
  mock_attach_result = JNI_ERR;
  assert(spark_connect_engine_next_arrow(query, &out, &finalizations, defer, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:cannot attach Rust worker to JVM");
  assert(atomic_load(&finalizations) == 3 && released_owner == NULL);
  mock_attach_result = JNI_OK;

  /* EOF and a Java next-batch failure both consume the caller's lease once. */
  assert(spark_connect_engine_next_arrow(query, &out, &finalizations, defer, &error) == 0);
  assert(error.data == NULL && out.release == NULL);
  assert(atomic_load(&finalizations) == 4 && released_owner != NULL);
  assert(spark_connect_engine_arrow_owner_free_checked(released_owner, &error) == 0);
  mock_next_failure = 1;
  assert(spark_connect_engine_next_arrow(query, &out, &finalizations, defer, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:injected child access failure");
  assert(atomic_load(&finalizations) == 5 && released_owner != NULL);
  assert(spark_connect_engine_arrow_owner_free_checked(released_owner, &error) == 0);
  mock_next_failure = 0;

  MockNode invalid = {0};
  MockNode root = {.valid = 1, .children = {.count = 1, .child = &invalid}};
  mock_next_batch = (jobject) (uintptr_t) 31;
  mock_descriptor = (jobject) &root;
  assert(spark_connect_engine_next_arrow(query, &out, &finalizations, defer, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:invalid Java Arrow node type");
  assert(out.release == NULL && atomic_load(&finalizations) == 6);
  ArrowExportOwner *retained = released_owner;
  assert(retained != NULL && retained->batch != NULL);
  assert(atomic_load(&query->api->references) == 2 && mock_close_calls == 0);
  mock_close_failure = 1;
  assert(spark_connect_engine_arrow_owner_free_checked(retained, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:injected child access failure");
  assert(retained->batch != NULL && atomic_load(&query->api->references) == 2);
  mock_close_failure = 0;
  assert(spark_connect_engine_arrow_owner_free_checked(retained, &error) == 0);
  assert(atomic_load(&query->api->references) == 1);

  /* Failure to retain a Java batch must also dispose its local reference and lease. */
  mock_new_global_failure = 1;
  assert(spark_connect_engine_next_arrow(query, &out, &finalizations, defer, &error) == -1);
  attachment_error(&error, "JVM_ENGINE_ERROR:13:cannot retain Java engine handle");
  assert(atomic_load(&finalizations) == 7 && released_owner != NULL);
  assert(((ArrowExportOwner *) released_owner)->batch == NULL);
  assert(spark_connect_engine_arrow_owner_free_checked(released_owner, &error) == 0);
  mock_new_global_failure = 0;

  /* Successful export transfers the same lease until the consumer releases it. */
  root.children.count = 0;
  assert(spark_connect_engine_next_arrow(query, &out, &finalizations, defer, &error) == 1);
  assert(out.release != NULL && atomic_load(&finalizations) == 7);
  out.release(&out);
  assert(atomic_load(&finalizations) == 8 && released_owner != NULL);
  assert(spark_connect_engine_arrow_owner_free_checked(released_owner, &error) == 0);
  assert(spark_connect_engine_free_checked(query, &error) == 0);
  assert(mock_close_calls == 5 && mock_globals_deleted == 6 && mock_frame_depth == 0);
  atomic_store(&finalizations, 0);
  puts("C_ARROW_ERROR_OWNERSHIP_OK: invalid destinations, attachment/Java failures, EOF and partial exports consume exactly one lease; failed batch close remains retryable");
}

int main(void) {
  analysis_preconditions();
  worker_attachments();
  descriptor_reference_frames();
  checked_cleanup_ownership();
  arrow_export_error_ownership();
  int64_t data[] = {7, 11, 13};
  ArrowExportOwner *shared = owner(2);
  struct ArrowArray parent = node(shared, data);
  parent.n_children = 1;
  parent.children = calloc(1, sizeof(*parent.children));
  assert(parent.children != NULL);
  parent.children[0] = malloc(sizeof(*parent.children[0]));
  assert(parent.children[0] != NULL);
  *parent.children[0] = node(shared, data);
  assert(spark_connect_engine_arrow_buffer_size(&parent, 1) == (int64_t) sizeof(data));
  assert(spark_connect_engine_arrow_buffer_size(&parent, -1) == -1);
  assert(parent.buffers[1] == data);

  /* A C Data consumer moves a descriptor by clearing its original release slot. */
  struct ArrowArray moved = *parent.children[0];
  parent.children[0]->release = NULL;
  parent.release(&parent);
  assert(atomic_load(&finalizations) == 0);
  assert(spark_connect_engine_arrow_buffer_size(&parent, 1) == -1);
  assert(moved.buffers[1] == data);
  assert(data[0] == 7 && data[1] == 11 && data[2] == 13);
  moved.release(&moved);
  assert(atomic_load(&finalizations) == 1);
  arrow_node_release(&parent);
  arrow_node_release(&moved);
  assert(atomic_load(&finalizations) == 1);
  spark_connect_engine_arrow_owner_free(released_owner);

  /* Independent descriptors may drop on different consumer threads. */
  shared = owner(2);
  struct ArrowArray first = node(shared, data);
  struct ArrowArray second = node(shared, data);
  pthread_t a, b;
  assert(pthread_create(&a, NULL, release_on_worker, &first) == 0);
  assert(pthread_create(&b, NULL, release_on_worker, &second) == 0);
  assert(pthread_join(a, NULL) == 0 && pthread_join(b, NULL) == 0);
  assert(atomic_load(&finalizations) == 2);
  assert(data[0] == 7 && data[1] == 11 && data[2] == 13);
  spark_connect_engine_arrow_owner_free(released_owner);
  puts("C_ARROW_OWNERSHIP_OK: data pointers shared, moved children retained, one final release across threads");
  return 0;
}
