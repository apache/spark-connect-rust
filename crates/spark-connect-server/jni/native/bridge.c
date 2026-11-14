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

#include "bridge.h"

#include <jni.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  JavaVM *vm;
  jobject loader;
  jclass thread_class;
  jmethodID current_thread;
  jmethodID get_loader;
  jmethodID set_loader;
  atomic_uint references;
} JavaContext;

typedef struct {
  JavaContext *context;
  JNIEnv *env;
  jobject thread;
  jobject previous_loader;
  int attached;
  int framed;
  int loader_changed;
} JavaScope;

typedef struct {
  JavaContext *context;
  pthread_t owner;
  int attached;
} EngineThread;

extern int64_t spark_connect_engine_start(const uint8_t *, size_t, void *, BridgeBuffer *);
extern int spark_connect_address(int64_t, BridgeBuffer *, BridgeBuffer *);
extern int spark_connect_stop(int64_t, BridgeBuffer *);

static void clear_buffer(BridgeBuffer *buffer) {
  if (buffer != NULL) {
    buffer->data = NULL;
    buffer->len = 0;
  }
}

static void set_error(BridgeBuffer *error, const char *message) {
  if (error == NULL || error->data != NULL) {
    return;
  }
  size_t len = strlen(message);
  uint8_t *data = malloc(len == 0 ? 1 : len);
  if (data != NULL) {
    memcpy(data, message, len);
    error->data = data;
    error->len = len;
  }
}

void spark_connect_buffer_free(BridgeBuffer *buffer) {
  if (buffer != NULL) {
    free(buffer->data);
    clear_buffer(buffer);
  }
}

/* GetStringUTFChars uses modified UTF-8, so encode UTF-16 explicitly instead. */
static int string_buffer(JNIEnv *env, jstring value, BridgeBuffer *out) {
  jsize len = (*env)->GetStringLength(env, value);
  const jchar *chars = (*env)->GetStringChars(env, value, NULL);
  if (chars == NULL) {
    return -1;
  }
  uint8_t *data = malloc((size_t) len * 3 + 1);
  if (data == NULL) {
    (*env)->ReleaseStringChars(env, value, chars);
    return -1;
  }
  size_t written = 0;
  for (jsize i = 0; i < len; i++) {
    uint32_t code = chars[i];
    if (code >= 0xd800 && code <= 0xdbff && i + 1 < len
        && chars[i + 1] >= 0xdc00 && chars[i + 1] <= 0xdfff) {
      code = 0x10000 + ((code - 0xd800) << 10) + chars[++i] - 0xdc00;
    } else if (code >= 0xd800 && code <= 0xdfff) {
      code = 0xfffd;
    }
    if (code < 0x80) {
      data[written++] = (uint8_t) code;
    } else if (code < 0x800) {
      data[written++] = (uint8_t) (0xc0 | (code >> 6));
      data[written++] = (uint8_t) (0x80 | (code & 0x3f));
    } else if (code < 0x10000) {
      data[written++] = (uint8_t) (0xe0 | (code >> 12));
      data[written++] = (uint8_t) (0x80 | ((code >> 6) & 0x3f));
      data[written++] = (uint8_t) (0x80 | (code & 0x3f));
    } else {
      data[written++] = (uint8_t) (0xf0 | (code >> 18));
      data[written++] = (uint8_t) (0x80 | ((code >> 12) & 0x3f));
      data[written++] = (uint8_t) (0x80 | ((code >> 6) & 0x3f));
      data[written++] = (uint8_t) (0x80 | (code & 0x3f));
    }
  }
  (*env)->ReleaseStringChars(env, value, chars);
  out->data = data;
  out->len = written;
  return 0;
}

static int exception_error(JNIEnv *env, BridgeBuffer *error) {
  if (!(*env)->ExceptionCheck(env)) {
    return 0;
  }
  jthrowable throwable = (*env)->ExceptionOccurred(env);
  (*env)->ExceptionClear(env);
  jclass type = throwable == NULL ? NULL : (*env)->GetObjectClass(env, throwable);
  jmethodID to_string = type == NULL ? NULL
      : (*env)->GetMethodID(env, type, "toString", "()Ljava/lang/String;");
  jstring description = to_string == NULL ? NULL
      : (*env)->CallObjectMethod(env, throwable, to_string);
  if (!(*env)->ExceptionCheck(env) && description != NULL && error != NULL
      && error->data == NULL) {
    (void) string_buffer(env, description, error);
  }
  if ((*env)->ExceptionCheck(env)) {
    (*env)->ExceptionClear(env);
  }
  if (description != NULL) (*env)->DeleteLocalRef(env, description);
  if (type != NULL) (*env)->DeleteLocalRef(env, type);
  if (throwable != NULL) (*env)->DeleteLocalRef(env, throwable);
  set_error(error, "Java bridge operation failed");
  return -1;
}

static void scope_restore_loader(JavaScope *scope) {
  JNIEnv *env = scope->env;
  if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
  if (scope->loader_changed) {
    (*env)->CallVoidMethod(env, scope->thread, scope->context->set_loader,
        scope->previous_loader);
    if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
    scope->loader_changed = 0;
  }
}

static void scope_leave(JavaScope *scope) {
  if (scope->env == NULL) return;
  JNIEnv *env = scope->env;
  scope_restore_loader(scope);
  if (scope->previous_loader != NULL) (*env)->DeleteLocalRef(env, scope->previous_loader);
  if (scope->thread != NULL) (*env)->DeleteLocalRef(env, scope->thread);
  if (scope->framed) (*env)->PopLocalFrame(env, NULL);
  if (scope->attached) (*scope->context->vm)->DetachCurrentThread(scope->context->vm);
  scope->env = NULL;
}

static int scope_enter(JavaContext *context, JavaScope *scope, BridgeBuffer *error) {
  memset(scope, 0, sizeof(*scope));
  scope->context = context;
  jint state = (*context->vm)->GetEnv(context->vm, (void **) &scope->env, JNI_VERSION_1_8);
  if (state == JNI_EDETACHED) {
    if ((*context->vm)->AttachCurrentThreadAsDaemon(
        context->vm, (void **) &scope->env, NULL) != JNI_OK) {
      scope->env = NULL;
      set_error(error, "cannot attach Rust worker to JVM");
      return -1;
    }
    scope->attached = 1;
  } else if (state != JNI_OK) {
    scope->env = NULL;
    set_error(error, "cannot access JVM from Rust worker");
    return -1;
  }
  JNIEnv *env = scope->env;
  if ((*env)->PushLocalFrame(env, 32) < 0) {
    (void) exception_error(env, error);
    set_error(error, "cannot allocate Java local reference frame");
    goto failed;
  }
  scope->framed = 1;
  scope->thread = (*env)->CallStaticObjectMethod(env, context->thread_class,
      context->current_thread);
  if (exception_error(env, error) != 0 || scope->thread == NULL) goto failed;
  scope->previous_loader = (*env)->CallObjectMethod(env, scope->thread, context->get_loader);
  if (exception_error(env, error) != 0) goto failed;
  if (!(*env)->IsSameObject(env, scope->previous_loader, context->loader)) {
    (*env)->CallVoidMethod(env, scope->thread, context->set_loader, context->loader);
    if (exception_error(env, error) != 0) goto failed;
    scope->loader_changed = 1;
  }
  return 0;
failed:
  set_error(error, "cannot prepare JVM worker context");
  scope_leave(scope);
  return -1;
}

static void context_release(JavaContext *context) {
  if (atomic_fetch_sub_explicit(&context->references, 1, memory_order_acq_rel) != 1) return;
  // The factory cannot be freed while worker guards exist. Consequently detach
  // never owns the last context reference, and must not reattach just to destroy it.
  // Preserve an unexpected orphan rather than deleting globals without a valid env.
  atomic_store_explicit(&context->references, 1, memory_order_release);
  fputs("JNI cleanup is incomplete; retained cleanup failed: orphaned Java worker context\n",
      stderr);
}

static jbyteArray byte_array(JNIEnv *env, const uint8_t *data, size_t len,
    BridgeBuffer *error) {
  if (len > INT_MAX || (len != 0 && data == NULL)) {
    set_error(error, "Java bridge byte array is too large or invalid");
    return NULL;
  }
  jbyteArray array = (*env)->NewByteArray(env, (jsize) len);
  if (exception_error(env, error) != 0 || array == NULL) return NULL;
  if (len != 0) (*env)->SetByteArrayRegion(env, array, 0, (jsize) len, (const jbyte *) data);
  if (exception_error(env, error) != 0) {
    (*env)->DeleteLocalRef(env, array);
    return NULL;
  }
  return array;
}

static jstring utf8_string(JNIEnv *env, const uint8_t *data, size_t len,
    BridgeBuffer *error) {
  jbyteArray bytes = byte_array(env, data, len, error);
  if (bytes == NULL) return NULL;
  jclass type = (*env)->FindClass(env, "java/lang/String");
  jmethodID constructor = type == NULL ? NULL
      : (*env)->GetMethodID(env, type, "<init>", "([BLjava/lang/String;)V");
  jstring encoding = constructor == NULL ? NULL : (*env)->NewStringUTF(env, "UTF-8");
  jstring value = NULL;
  if (exception_error(env, error) == 0 && encoding != NULL) {
    value = (*env)->NewObject(env, type, constructor, bytes, encoding);
    if (exception_error(env, error) != 0) value = NULL;
  }
  if (encoding != NULL) (*env)->DeleteLocalRef(env, encoding);
  if (type != NULL) (*env)->DeleteLocalRef(env, type);
  (*env)->DeleteLocalRef(env, bytes);
  return value;
}

static int array_buffer(JNIEnv *env, jbyteArray array, BridgeBuffer *out,
    BridgeBuffer *error) {
  jsize len = (*env)->GetArrayLength(env, array);
  if (exception_error(env, error) != 0) return -1;
  uint8_t *data = malloc(len == 0 ? 1 : (size_t) len);
  if (data == NULL) {
    set_error(error, "cannot allocate Java bridge buffer");
    return -1;
  }
  if (len != 0) (*env)->GetByteArrayRegion(env, array, 0, len, (jbyte *) data);
  if (exception_error(env, error) != 0) {
    free(data);
    return -1;
  }
  out->data = data;
  out->len = (size_t) len;
  return 0;
}

static void throw_error(JNIEnv *env, BridgeBuffer *error) {
  if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
  if (error->data == NULL) set_error(error, "native Spark Connect operation failed");
  jstring message = utf8_string(env, error->data, error->len, NULL);
  jclass type = (*env)->FindClass(env, "java/lang/IllegalStateException");
  if (!(*env)->ExceptionCheck(env) && message != NULL && type != NULL) {
    jmethodID constructor = (*env)->GetMethodID(env, type, "<init>", "(Ljava/lang/String;)V");
    if (!(*env)->ExceptionCheck(env) && constructor != NULL) {
      jthrowable throwable = (*env)->NewObject(env, type, constructor, message);
      if (!(*env)->ExceptionCheck(env) && throwable != NULL) {
        (*env)->Throw(env, throwable);
        (*env)->DeleteLocalRef(env, throwable);
      }
    }
  }
  if (message != NULL) (*env)->DeleteLocalRef(env, message);
  if (type != NULL) (*env)->DeleteLocalRef(env, type);
  spark_connect_buffer_free(error);
}

void *spark_connect_engine_native_address(void *raw_env, void *clazz, int64_t id) {
  (void) clazz;
  JNIEnv *env = raw_env;
  BridgeBuffer address = {0};
  BridgeBuffer error = {0};
  if (spark_connect_address(id, &address, &error) != 0) {
    spark_connect_buffer_free(&address);
    throw_error(env, &error);
    return NULL;
  }
  jbyteArray result = byte_array(env, address.data, address.len, &error);
  spark_connect_buffer_free(&address);
  if (result == NULL) throw_error(env, &error);
  else spark_connect_buffer_free(&error);
  return result;
}

void spark_connect_engine_native_stop(void *raw_env, void *clazz, int64_t id) {
  (void) clazz;
  BridgeBuffer error = {0};
  if (spark_connect_stop(id, &error) != 0) throw_error(raw_env, &error);
  else spark_connect_buffer_free(&error);
}

/* Typed JVM query engine and Arrow ownership implementation. */
typedef enum {
  ENGINE_FACTORY,
  ENGINE_TASK,
  ENGINE_SESSION,
  ENGINE_QUERY,
  ENGINE_BATCH,
  ENGINE_EXCEPTION,
  ENGINE_NODE,
  ENGINE_CLASS_COUNT
} EngineKind;

typedef struct {
  JavaContext *context;
  atomic_uint references;
  jclass classes[ENGINE_CLASS_COUNT];
  jmethodID close[ENGINE_EXCEPTION];
  jmethodID new_task;
  jmethodID open_session;
  jmethodID prepare;
  jmethodID analyze;
  jmethodID config;
  jmethodID schema;
  jmethodID next_arrow;
  jmethodID descriptor;
  jmethodID node_length;
  jmethodID node_null_count;
  jmethodID node_offset;
  jmethodID node_buffers;
  jmethodID node_buffer_sizes;
  jmethodID node_children;
  jmethodID cancel;
  jmethodID exception_code;
} EngineApi;

typedef struct {
  EngineApi *api;
  jobject object;
  EngineKind kind;
} EngineHandle;

static void engine_error_bytes(BridgeBuffer *error, int code,
    const uint8_t *message, size_t len) {
  if (error == NULL || error->data != NULL) return;
  char prefix[40];
  int prefix_len = snprintf(prefix, sizeof(prefix), "JVM_ENGINE_ERROR:%d:", code);
  if (prefix_len < 0 || (size_t) prefix_len >= sizeof(prefix)
      || len > SIZE_MAX - (size_t) prefix_len) return;
  uint8_t *data = malloc((size_t) prefix_len + len);
  if (data == NULL) return;
  memcpy(data, prefix, (size_t) prefix_len);
  if (len != 0) memcpy(data + prefix_len, message, len);
  error->data = data;
  error->len = (size_t) prefix_len + len;
}

static void engine_error(BridgeBuffer *error, int code, const char *message) {
  engine_error_bytes(error, code, (const uint8_t *) message, strlen(message));
}

/* Shared low-level helpers report plain text. Frame it at this ABI's boundary. */
static void engine_normalize_error(BridgeBuffer *error, const char *fallback) {
  static const char prefix[] = "JVM_ENGINE_ERROR:";
  if (error == NULL) return;
  if (error->len >= sizeof(prefix) - 1 && error->data != NULL
      && memcmp(error->data, prefix, sizeof(prefix) - 1) == 0) return;
  BridgeBuffer previous = *error;
  clear_buffer(error);
  if (previous.data != NULL) {
    engine_error_bytes(error, 13, previous.data, previous.len);
    spark_connect_buffer_free(&previous);
  } else {
    engine_error(error, 13, fallback);
  }
}

static int engine_exception(JNIEnv *env, EngineApi *api, BridgeBuffer *error) {
  if (!(*env)->ExceptionCheck(env)) return 0;
  jthrowable throwable = (*env)->ExceptionOccurred(env);
  (*env)->ExceptionClear(env);
  int code = 13;
  int typed = throwable != NULL && api != NULL
      && api->classes[ENGINE_EXCEPTION] != NULL && api->exception_code != NULL
      && (*env)->IsInstanceOf(env, throwable, api->classes[ENGINE_EXCEPTION]);
  if (typed) {
    jint reported = (*env)->CallIntMethod(env, throwable, api->exception_code);
    if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
    else if (reported >= 1 && reported <= 16) code = reported;
  }
  jclass type = throwable == NULL ? NULL : (*env)->GetObjectClass(env, throwable);
  if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
  jmethodID describe = type == NULL ? NULL : (*env)->GetMethodID(env, type,
      typed ? "getMessage" : "toString", "()Ljava/lang/String;");
  if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
  jstring description = describe == NULL ? NULL
      : (*env)->CallObjectMethod(env, throwable, describe);
  if ((*env)->ExceptionCheck(env)) {
    (*env)->ExceptionClear(env);
    description = NULL;
  }
  BridgeBuffer message = {0};
  if (description != NULL) (void) string_buffer(env, description, &message);
  if ((*env)->ExceptionCheck(env)) (*env)->ExceptionClear(env);
  if (message.data != NULL) engine_error_bytes(error, code, message.data, message.len);
  else engine_error(error, code, "Java engine operation failed");
  spark_connect_buffer_free(&message);
  if (description != NULL) (*env)->DeleteLocalRef(env, description);
  if (type != NULL) (*env)->DeleteLocalRef(env, type);
  if (throwable != NULL) (*env)->DeleteLocalRef(env, throwable);
  return -1;
}

static void engine_delete_classes(JNIEnv *env, EngineApi *api) {
  for (int i = 0; i < ENGINE_CLASS_COUNT; i++) {
    if (api->classes[i] != NULL) (*env)->DeleteGlobalRef(env, api->classes[i]);
  }
}

static void engine_api_release(JNIEnv *env, EngineApi *api) {
  if (atomic_fetch_sub_explicit(&api->references, 1, memory_order_acq_rel) != 1) return;
  // Startup already supplies a valid Java environment. Do not risk another
  // attachment after disposing the last handle.
  engine_delete_classes(env, api);
  JavaContext *context = api->context;
  if (atomic_fetch_sub_explicit(&context->references, 1, memory_order_acq_rel) == 1) {
    if (context->loader != NULL) (*env)->DeleteGlobalRef(env, context->loader);
    (*env)->DeleteGlobalRef(env, context->thread_class);
    free(context);
  }
  free(api);
}

static jclass engine_class(JNIEnv *env, EngineApi *api, const char *name,
    BridgeBuffer *error) {
  jclass local = (*env)->FindClass(env, name);
  if (engine_exception(env, api, error) != 0 || local == NULL) return NULL;
  jclass global = (*env)->NewGlobalRef(env, local);
  (*env)->DeleteLocalRef(env, local);
  if (engine_exception(env, api, error) != 0) return NULL;
  return global;
}

static int engine_method(JNIEnv *env, EngineApi *api, EngineKind kind,
    jmethodID *out, const char *name, const char *signature, BridgeBuffer *error) {
  *out = (*env)->GetMethodID(env, api->classes[kind], name, signature);
  if (engine_exception(env, api, error) != 0 || *out == NULL) return -1;
  return 0;
}

static EngineHandle *engine_handle(JNIEnv *env, EngineApi *api, jobject object,
    EngineKind kind, BridgeBuffer *error) {
  EngineHandle *handle = calloc(1, sizeof(*handle));
  if (handle != NULL) {
    handle->object = (*env)->NewGlobalRef(env, object);
    (void) engine_exception(env, api, error);
  }
  if (handle == NULL || handle->object == NULL) {
    engine_error(error, 13, "cannot retain Java engine handle");
    (*env)->CallVoidMethod(env, object, api->close[kind]);
    (void) engine_exception(env, api, NULL);
    free(handle);
    return NULL;
  }
  handle->api = api;
  handle->kind = kind;
  atomic_fetch_add_explicit(&api->references, 1, memory_order_relaxed);
  return handle;
}

static EngineHandle *engine_factory(JNIEnv *env, jobject factory, BridgeBuffer *error) {
  EngineApi *api = calloc(1, sizeof(*api));
  JavaContext *context = calloc(1, sizeof(*context));
  if (api == NULL || context == NULL) {
    free(api);
    free(context);
    engine_error(error, 13, "cannot allocate Java engine context");
    return NULL;
  }
  api->context = context;
  atomic_init(&api->references, 1);
  atomic_init(&context->references, 1);
  if ((*env)->GetJavaVM(env, &context->vm) != JNI_OK) goto failed;
  context->thread_class = engine_class(env, api, "java/lang/Thread", error);
  if (context->thread_class == NULL) goto failed;
  context->current_thread = (*env)->GetStaticMethodID(env, context->thread_class,
      "currentThread", "()Ljava/lang/Thread;");
  if (engine_exception(env, api, error) != 0 || context->current_thread == NULL) goto failed;
  context->get_loader = (*env)->GetMethodID(env, context->thread_class,
      "getContextClassLoader", "()Ljava/lang/ClassLoader;");
  if (engine_exception(env, api, error) != 0 || context->get_loader == NULL) goto failed;
  context->set_loader = (*env)->GetMethodID(env, context->thread_class,
      "setContextClassLoader", "(Ljava/lang/ClassLoader;)V");
  if (engine_exception(env, api, error) != 0 || context->set_loader == NULL) goto failed;
  jobject current = (*env)->CallStaticObjectMethod(env, context->thread_class,
      context->current_thread);
  if (engine_exception(env, api, error) != 0 || current == NULL) goto failed;
  jobject loader = (*env)->CallObjectMethod(env, current, context->get_loader);
  (*env)->DeleteLocalRef(env, current);
  if (engine_exception(env, api, error) != 0) goto failed;
  if (loader != NULL) {
    context->loader = (*env)->NewGlobalRef(env, loader);
    (*env)->DeleteLocalRef(env, loader);
    if (engine_exception(env, api, error) != 0 || context->loader == NULL) goto failed;
  }
  jclass factory_class = (*env)->GetObjectClass(env, factory);
  if (engine_exception(env, api, error) != 0 || factory_class == NULL) goto failed;
  api->classes[ENGINE_FACTORY] = (*env)->NewGlobalRef(env, factory_class);
  (*env)->DeleteLocalRef(env, factory_class);
  if (engine_exception(env, api, error) != 0 || api->classes[ENGINE_FACTORY] == NULL) goto failed;
  api->classes[ENGINE_TASK] = engine_class(env, api,
      "org/apache/spark/sql/connect/nativebridge/engine/SparkJvmEngine$Task", error);
  if (api->classes[ENGINE_TASK] == NULL) goto failed;
  api->classes[ENGINE_SESSION] = engine_class(env, api,
      "org/apache/spark/sql/connect/nativebridge/engine/SparkJvmEngine$Session", error);
  if (api->classes[ENGINE_SESSION] == NULL) goto failed;
  api->classes[ENGINE_QUERY] = engine_class(env, api,
      "org/apache/spark/sql/connect/nativebridge/engine/SparkJvmEngine$Query", error);
  if (api->classes[ENGINE_QUERY] == NULL) goto failed;
  api->classes[ENGINE_BATCH] = engine_class(env, api,
      "org/apache/spark/sql/connect/nativebridge/engine/NativeArrowBatch", error);
  if (api->classes[ENGINE_BATCH] == NULL) goto failed;
  api->classes[ENGINE_NODE] = engine_class(env, api,
      "org/apache/spark/sql/connect/nativebridge/engine/NativeArrowBatch$Node", error);
  if (api->classes[ENGINE_NODE] == NULL) goto failed;
  api->classes[ENGINE_EXCEPTION] = engine_class(env, api,
      "org/apache/spark/sql/connect/nativebridge/engine/SparkJvmEngine$EngineException", error);
  if (api->classes[ENGINE_EXCEPTION] == NULL) goto failed;
  for (int i = ENGINE_FACTORY; i < ENGINE_EXCEPTION; i++) {
    if (engine_method(env, api, (EngineKind) i, &api->close[i], "close", "()V", error) != 0)
      goto failed;
  }
  if (engine_method(env, api, ENGINE_EXCEPTION, &api->exception_code, "code", "()I", error) != 0
      || engine_method(env, api, ENGINE_FACTORY, &api->new_task, "newTask",
          "(Ljava/lang/String;)Lorg/apache/spark/sql/connect/nativebridge/engine/SparkJvmEngine$Task;",
          error) != 0
      || engine_method(env, api, ENGINE_FACTORY, &api->open_session, "openSession",
          "(Ljava/lang/String;Ljava/lang/String;Lorg/apache/spark/sql/connect/nativebridge/engine/SparkJvmEngine$Task;)Lorg/apache/spark/sql/connect/nativebridge/engine/SparkJvmEngine$Session;",
          error) != 0
      || engine_method(env, api, ENGINE_SESSION, &api->prepare, "prepare",
          "([BLorg/apache/spark/sql/connect/nativebridge/engine/SparkJvmEngine$Task;)Lorg/apache/spark/sql/connect/nativebridge/engine/SparkJvmEngine$Query;",
          error) != 0
      || engine_method(env, api, ENGINE_SESSION, &api->analyze, "analyze",
          "(I[BLorg/apache/spark/sql/connect/nativebridge/engine/SparkJvmEngine$Task;)[B",
          error) != 0
      || engine_method(env, api, ENGINE_SESSION, &api->config, "config",
          "(I[BLorg/apache/spark/sql/connect/nativebridge/engine/SparkJvmEngine$Task;)[B",
          error) != 0
      || engine_method(env, api, ENGINE_TASK, &api->cancel, "cancel", "()V", error) != 0
      || engine_method(env, api, ENGINE_QUERY, &api->schema, "schemaIpc", "()[B", error) != 0
      || engine_method(env, api, ENGINE_QUERY, &api->next_arrow, "nextVectorBatch",
          "()Lorg/apache/spark/sql/connect/nativebridge/engine/NativeArrowBatch;", error) != 0
      || engine_method(env, api, ENGINE_BATCH, &api->descriptor, "descriptor",
          "()Lorg/apache/spark/sql/connect/nativebridge/engine/NativeArrowBatch$Node;", error) != 0
      || engine_method(env, api, ENGINE_NODE, &api->node_length, "length", "()J", error) != 0
      || engine_method(env, api, ENGINE_NODE, &api->node_null_count, "nullCount", "()J", error) != 0
      || engine_method(env, api, ENGINE_NODE, &api->node_offset, "offset", "()J", error) != 0
      || engine_method(env, api, ENGINE_NODE, &api->node_buffers, "buffers", "()[J", error) != 0
      || engine_method(env, api, ENGINE_NODE, &api->node_buffer_sizes, "bufferSizes", "()[J", error) != 0
      || engine_method(env, api, ENGINE_NODE, &api->node_children, "children",
          "()[Lorg/apache/spark/sql/connect/nativebridge/engine/NativeArrowBatch$Node;", error) != 0)
    goto failed;
  EngineHandle *handle = engine_handle(env, api, factory, ENGINE_FACTORY, error);
  engine_api_release(env, api);
  return handle;
failed:
  (void) engine_exception(env, api, error);
  engine_normalize_error(error, "cannot initialize Java engine ABI");
  engine_delete_classes(env, api);
  if (context->loader != NULL) (*env)->DeleteGlobalRef(env, context->loader);
  if (context->thread_class != NULL) (*env)->DeleteGlobalRef(env, context->thread_class);
  free(context);
  free(api);
  return NULL;
}

static int engine_check(EngineHandle *handle, EngineKind kind, BridgeBuffer *error) {
  if (handle != NULL && handle->kind == kind) return 0;
  engine_error(error, 13, "invalid Java engine handle");
  return -1;
}

static int engine_task_check(EngineHandle *handle, EngineHandle *task, BridgeBuffer *error) {
  if (engine_check(task, ENGINE_TASK, error) != 0) return -1;
  if (handle->api == task->api) return 0;
  engine_error(error, 13, "Java engine task belongs to another engine");
  return -1;
}

static int engine_scope(EngineHandle *handle, JavaScope *scope, BridgeBuffer *error) {
  if (scope_enter(handle->api->context, scope, error) == 0) return 0;
  engine_normalize_error(error, "cannot enter Java engine context");
  return -1;
}

void *spark_connect_engine_thread_attach(void *raw_engine, BridgeBuffer *error) {
  clear_buffer(error);
  EngineHandle *engine = raw_engine;
  if (engine_check(engine, ENGINE_FACTORY, error) != 0) return NULL;
  EngineThread *thread = calloc(1, sizeof(*thread));
  if (thread == NULL) {
    engine_error(error, 8, "cannot allocate Java worker attachment");
    return NULL;
  }
  JavaContext *context = engine->api->context;
  JNIEnv *env = NULL;
  jint state = (*context->vm)->GetEnv(context->vm, (void **) &env, JNI_VERSION_1_8);
  if (state == JNI_EDETACHED) {
    static atomic_uint_fast64_t next_worker = 1;
    uint_fast64_t id = atomic_load_explicit(&next_worker, memory_order_relaxed);
    do {
      if (id == UINT_FAST64_MAX) {
        engine_error(error, 8, "Java engine worker identifiers are exhausted");
        free(thread);
        return NULL;
      }
    } while (!atomic_compare_exchange_weak_explicit(&next_worker, &id, id + 1,
        memory_order_relaxed, memory_order_relaxed));
    char name[64];
    (void) snprintf(name, sizeof(name), "spark-connect-engine-worker-%" PRIuFAST64, id);
    /* JNI consumes the name during attachment; the VM keeps its own copy. */
    JavaVMAttachArgs args = {
      .version = JNI_VERSION_1_8,
      .name = name,
      .group = NULL,
    };
    if ((*context->vm)->AttachCurrentThreadAsDaemon(
        context->vm, (void **) &env, &args) != JNI_OK) {
      engine_error(error, 13, "cannot attach Java engine worker");
      free(thread);
      return NULL;
    }
    thread->attached = 1;
  } else if (state != JNI_OK) {
    engine_error(error, 13, "cannot access Java engine worker context");
    free(thread);
    return NULL;
  }
  if (env == NULL) {
    if (thread->attached) (*context->vm)->DetachCurrentThread(context->vm);
    engine_error(error, 13, "Java engine worker has no JNI environment");
    free(thread);
    return NULL;
  }
  thread->context = context;
  thread->owner = pthread_self();
  atomic_fetch_add_explicit(&context->references, 1, memory_order_relaxed);
  return thread;
}

int spark_connect_engine_thread_detach(void *raw_attachment, BridgeBuffer *error) {
  clear_buffer(error);
  EngineThread *thread = raw_attachment;
  if (thread == NULL) return 0;
  if (!pthread_equal(thread->owner, pthread_self())) {
    engine_error(error, 13, "Java worker attachment belongs to another thread");
    return -1;
  }
  JavaContext *context = thread->context;
  if (thread->attached && (*context->vm)->DetachCurrentThread(context->vm) != JNI_OK) {
    engine_error(error, 13, "cannot detach Java engine worker");
    return -1;
  }
  context_release(context);
  free(thread);
  return 0;
}

void *spark_connect_engine_task_new(void *raw_engine, BridgeBuffer *error) {
  clear_buffer(error);
  EngineHandle *engine = raw_engine;
  if (engine_check(engine, ENGINE_FACTORY, error) != 0) return NULL;
  static atomic_uint_fast64_t next_task = 1;
  uint_fast64_t id = atomic_load_explicit(&next_task, memory_order_relaxed);
  do {
    if (id == UINT_FAST64_MAX) {
      engine_error(error, 8, "Java engine task identifiers are exhausted");
      return NULL;
    }
  } while (!atomic_compare_exchange_weak_explicit(&next_task, &id, id + 1,
      memory_order_relaxed, memory_order_relaxed));
  char name[64];
  int len = snprintf(name, sizeof(name), "spark-engine-%" PRIuFAST64, id);
  JavaScope scope;
  if (engine_scope(engine, &scope, error) != 0) return NULL;
  JNIEnv *env = scope.env;
  jstring description = utf8_string(env, (const uint8_t *) name, (size_t) len, error);
  jobject object = description == NULL ? NULL : (*env)->CallObjectMethod(env,
      engine->object, engine->api->new_task, description);
  EngineHandle *task = NULL;
  if (engine_exception(env, engine->api, error) == 0 && object != NULL)
    task = engine_handle(env, engine->api, object, ENGINE_TASK, error);
  if (object != NULL) (*env)->DeleteLocalRef(env, object);
  if (description != NULL) (*env)->DeleteLocalRef(env, description);
  scope_leave(&scope);
  if (task == NULL) engine_normalize_error(error, "Java engine did not create a task");
  return task;
}

void *spark_connect_engine_open_session(void *raw_engine, const uint8_t *user, size_t user_len,
    const uint8_t *id, size_t id_len, void *raw_task, BridgeBuffer *error) {
  clear_buffer(error);
  EngineHandle *engine = raw_engine;
  EngineHandle *task = raw_task;
  if (engine_check(engine, ENGINE_FACTORY, error) != 0
      || engine_task_check(engine, task, error) != 0) return NULL;
  JavaScope scope;
  if (engine_scope(engine, &scope, error) != 0) return NULL;
  JNIEnv *env = scope.env;
  jstring user_string = utf8_string(env, user, user_len, error);
  jstring id_string = user_string == NULL ? NULL : utf8_string(env, id, id_len, error);
  jobject object = id_string == NULL ? NULL : (*env)->CallObjectMethod(env,
      engine->object, engine->api->open_session, user_string, id_string, task->object);
  EngineHandle *session = NULL;
  if (engine_exception(env, engine->api, error) == 0 && object != NULL)
    session = engine_handle(env, engine->api, object, ENGINE_SESSION, error);
  if (object != NULL) (*env)->DeleteLocalRef(env, object);
  if (id_string != NULL) (*env)->DeleteLocalRef(env, id_string);
  if (user_string != NULL) (*env)->DeleteLocalRef(env, user_string);
  scope_leave(&scope);
  if (session == NULL) engine_normalize_error(error, "Java engine did not open a session");
  return session;
}

void *spark_connect_engine_prepare(void *raw_session, const uint8_t *relation, size_t len,
    void *raw_task, BridgeBuffer *error) {
  clear_buffer(error);
  EngineHandle *session = raw_session;
  EngineHandle *task = raw_task;
  if (engine_check(session, ENGINE_SESSION, error) != 0
      || engine_task_check(session, task, error) != 0) return NULL;
  JavaScope scope;
  if (engine_scope(session, &scope, error) != 0) return NULL;
  JNIEnv *env = scope.env;
  jbyteArray input = byte_array(env, relation, len, error);
  jobject object = input == NULL ? NULL : (*env)->CallObjectMethod(env,
      session->object, session->api->prepare, input, task->object);
  EngineHandle *query = NULL;
  if (engine_exception(env, session->api, error) == 0 && object != NULL)
    query = engine_handle(env, session->api, object, ENGINE_QUERY, error);
  if (object != NULL) (*env)->DeleteLocalRef(env, object);
  if (input != NULL) (*env)->DeleteLocalRef(env, input);
  scope_leave(&scope);
  if (query == NULL) engine_normalize_error(error, "Java engine did not prepare a query");
  return query;
}

int spark_connect_engine_schema(void *raw_query, BridgeBuffer *out, BridgeBuffer *error) {
  EngineHandle *query = raw_query;
  clear_buffer(out);
  clear_buffer(error);
  if (engine_check(query, ENGINE_QUERY, error) != 0) return -1;
  if (out == NULL) {
    engine_error(error, 13, "missing Java engine result buffer");
    return -1;
  }
  JavaScope scope;
  if (engine_scope(query, &scope, error) != 0) return -1;
  JNIEnv *env = scope.env;
  jbyteArray bytes = (*env)->CallObjectMethod(env, query->object, query->api->schema);
  int result = engine_exception(env, query->api, error);
  if (result == 0 && bytes != NULL) {
    result = array_buffer(env, bytes, out, error);
  } else if (result == 0) {
    engine_error(error, 13, "Java engine returned no schema");
    result = -1;
  }
  if (bytes != NULL) (*env)->DeleteLocalRef(env, bytes);
  scope_leave(&scope);
  if (result < 0) engine_normalize_error(error, "cannot read Java engine query");
  return result;
}

/* Java owns every off-heap value buffer. Native metadata describes its layout and
 * holds the batch alive until the last imported Rust buffer releases its owner. */
typedef struct {
  atomic_uint references;
  EngineHandle *batch;
  void *lease;
  void (*defer_release)(void *lease, void *owner);
} ArrowExportOwner;

typedef struct {
  ArrowExportOwner *owner;
  int64_t *capacities;
} ArrowNodePrivate;

int spark_connect_engine_arrow_owner_free_checked(void *raw_owner, BridgeBuffer *error) {
  clear_buffer(error);
  ArrowExportOwner *owner = raw_owner;
  if (owner == NULL) return 0;
  if (spark_connect_engine_free_checked(owner->batch, error) != 0) return -1;
  free(owner);
  return 0;
}

void spark_connect_engine_arrow_owner_free(void *raw_owner) {
  BridgeBuffer error = {0};
  if (spark_connect_engine_arrow_owner_free_checked(raw_owner, &error) != 0) {
    fputs("JNI cleanup is incomplete; retained cleanup failed: ", stderr);
    (void) fwrite(error.data, 1, error.len, stderr);
    fputc('\n', stderr);
  }
  spark_connect_buffer_free(&error);
}

static void arrow_owner_unref(ArrowExportOwner *owner) {
  if (atomic_fetch_sub_explicit(&owner->references, 1, memory_order_acq_rel) != 1) return;
  if (owner->defer_release != NULL) owner->defer_release(owner->lease, owner);
  else spark_connect_engine_arrow_owner_free(owner);
}

static void arrow_node_release(struct ArrowArray *array) {
  if (array == NULL || array->release == NULL) return;
  array->release = NULL;
  ArrowNodePrivate *private = array->private_data;
  /* Child structs may have been moved by a C Data consumer. In that case their
   * release pointer is NULL; the moved child's private owner remains retained. */
  if (array->children != NULL) {
    for (int64_t i = 0; i < array->n_children; i++) {
      if (array->children[i] != NULL) {
        if (array->children[i]->release != NULL)
          array->children[i]->release(array->children[i]);
        free(array->children[i]);
      }
    }
  }
  if (array->dictionary != NULL) {
    if (array->dictionary->release != NULL) array->dictionary->release(array->dictionary);
    free(array->dictionary);
  }
  free((void *) array->buffers);
  free(array->children);
  array->buffers = NULL;
  array->children = NULL;
  array->dictionary = NULL;
  array->private_data = NULL;
  if (private != NULL) {
    ArrowExportOwner *owner = private->owner;
    free(private->capacities);
    free(private);
    arrow_owner_unref(owner);
  }
}

int64_t spark_connect_engine_arrow_buffer_size(const struct ArrowArray *array, int64_t index) {
  if (array == NULL || array->release != arrow_node_release || array->private_data == NULL
      || index < 0 || index >= array->n_buffers) return -1;
  const ArrowNodePrivate *private = array->private_data;
  return private->capacities[index];
}

static int arrow_read_long(JNIEnv *env, EngineApi *api, jobject node,
    jmethodID method, int64_t *out, BridgeBuffer *error) {
  *out = (int64_t) (*env)->CallLongMethod(env, node, method);
  return engine_exception(env, api, error);
}

static int arrow_build_node(JNIEnv *env, EngineApi *api, jobject node,
    ArrowExportOwner *owner, struct ArrowArray *out, unsigned depth,
    size_t *remaining_nodes, BridgeBuffer *error) {
  if (node == NULL || depth > 128 || *remaining_nodes == 0) {
    engine_error(error, 13, "invalid or excessive Java Arrow descriptor tree");
    return -1;
  }
  (*remaining_nodes)--;
  /* Each level owns at most four descriptor references plus three temporary
   * exception references. Separate frames keep recursion out of the caller's
   * local-reference budget without enlarging metadata calls' 32-slot frame. */
  jint capacity = (*env)->PushLocalFrame(env, 8);
  if (engine_exception(env, api, error) != 0 || capacity != JNI_OK) {
    if (capacity == JNI_OK) (*env)->PopLocalFrame(env, NULL);
    engine_normalize_error(error, "cannot reserve Java Arrow descriptor references");
    return -1;
  }
  if (!(*env)->IsInstanceOf(env, node, api->classes[ENGINE_NODE])) {
    engine_error(error, 13, "invalid Java Arrow node type");
    (*env)->PopLocalFrame(env, NULL);
    return -1;
  }
  ArrowNodePrivate *private = calloc(1, sizeof(*private));
  if (private == NULL) {
    engine_error(error, 8, "cannot allocate Arrow descriptor");
    (*env)->PopLocalFrame(env, NULL);
    return -1;
  }
  private->owner = owner;
  atomic_fetch_add_explicit(&owner->references, 1, memory_order_relaxed);
  out->private_data = private;
  out->release = arrow_node_release;
  jlongArray buffers = NULL;
  jlongArray capacities = NULL;
  jobjectArray children = NULL;
  int result = -1;
  if (arrow_read_long(env, api, node, api->node_length, &out->length, error) != 0
      || arrow_read_long(env, api, node, api->node_null_count, &out->null_count, error) != 0
      || arrow_read_long(env, api, node, api->node_offset, &out->offset, error) != 0) goto done;
  if (out->length < 0 || out->null_count < 0 || out->null_count > out->length
      || out->offset != 0) {
    engine_error(error, 13, "invalid Java Arrow node length, null count, or offset");
    goto done;
  }
  buffers = (*env)->CallObjectMethod(env, node, api->node_buffers);
  if (engine_exception(env, api, error) != 0 || buffers == NULL) goto done;
  capacities = (*env)->CallObjectMethod(env, node, api->node_buffer_sizes);
  if (engine_exception(env, api, error) != 0 || capacities == NULL) goto done;
  children = (*env)->CallObjectMethod(env, node, api->node_children);
  if (engine_exception(env, api, error) != 0 || children == NULL) goto done;
  jsize n_buffers = (*env)->GetArrayLength(env, buffers);
  if (engine_exception(env, api, error) != 0) goto done;
  jsize n_capacities = (*env)->GetArrayLength(env, capacities);
  if (engine_exception(env, api, error) != 0) goto done;
  jsize n_children = (*env)->GetArrayLength(env, children);
  if (engine_exception(env, api, error) != 0) goto done;
  if (n_buffers != n_capacities || n_buffers > 1024 || n_children > 10000) {
    engine_error(error, 13, "invalid Java Arrow descriptor dimensions");
    goto done;
  }
  out->n_buffers = n_buffers;
  out->n_children = n_children;
  if (n_buffers != 0) {
    out->buffers = calloc((size_t) n_buffers, sizeof(*out->buffers));
    private->capacities = calloc((size_t) n_buffers, sizeof(*private->capacities));
    if (out->buffers == NULL || private->capacities == NULL) {
      engine_error(error, 8, "cannot allocate Arrow buffer descriptors");
      goto done;
    }
    for (jsize i = 0; i < n_buffers; i++) {
      jlong address;
      jlong capacity;
      (*env)->GetLongArrayRegion(env, buffers, i, 1, &address);
      if (engine_exception(env, api, error) != 0) goto done;
      (*env)->GetLongArrayRegion(env, capacities, i, 1, &capacity);
      if (engine_exception(env, api, error) != 0) goto done;
      if (capacity < 0 || (uint64_t) capacity > SIZE_MAX
          || ((uint64_t) address > UINTPTR_MAX)
          || (capacity != 0 && address == 0)) {
        engine_error(error, 13, "invalid Java Arrow buffer address or capacity");
        goto done;
      }
      out->buffers[i] = (const void *) (uintptr_t) (uint64_t) address;
      private->capacities[i] = capacity;
    }
  }
  if (n_children != 0) {
    out->children = calloc((size_t) n_children, sizeof(*out->children));
    if (out->children == NULL) {
      engine_error(error, 8, "cannot allocate Arrow child descriptors");
      goto done;
    }
    for (jsize i = 0; i < n_children; i++) {
      jobject child = (*env)->GetObjectArrayElement(env, children, i);
      if (engine_exception(env, api, error) != 0) {
        if (child != NULL) (*env)->DeleteLocalRef(env, child);
        goto done;
      }
      out->children[i] = calloc(1, sizeof(*out->children[i]));
      if (out->children[i] == NULL) {
        if (child != NULL) (*env)->DeleteLocalRef(env, child);
        engine_error(error, 8, "cannot allocate Arrow child descriptor");
        goto done;
      }
      int child_result = arrow_build_node(env, api, child, owner, out->children[i],
          depth + 1, remaining_nodes, error);
      if (child != NULL) (*env)->DeleteLocalRef(env, child);
      if (child_result != 0) goto done;
    }
  }
  result = 0;
done:
  if (children != NULL) (*env)->DeleteLocalRef(env, children);
  if (capacities != NULL) (*env)->DeleteLocalRef(env, capacities);
  if (buffers != NULL) (*env)->DeleteLocalRef(env, buffers);
  if (result != 0) {
    arrow_node_release(out);
    engine_normalize_error(error, "cannot export Java Arrow descriptor");
  }
  (*env)->PopLocalFrame(env, NULL);
  return result;
}

int spark_connect_engine_next_arrow(void *raw_query, struct ArrowArray *out, void *lease,
    void (*defer_release)(void *lease, void *owner), BridgeBuffer *error) {
  clear_buffer(error);
  EngineHandle *query = raw_query;
  if (lease == NULL || defer_release == NULL) {
    engine_error(error, 13, "invalid Java Arrow export destination or lease");
    return -1;
  }
  if (engine_check(query, ENGINE_QUERY, error) != 0) {
    defer_release(lease, NULL);
    return -1;
  }
  if (out == NULL || out->release != NULL) {
    engine_error(error, 13, "invalid Java Arrow export destination or lease");
    defer_release(lease, NULL);
    return -1;
  }
  memset(out, 0, sizeof(*out));
  JavaScope scope;
  if (engine_scope(query, &scope, error) != 0) {
    defer_release(lease, NULL);
    return -1;
  }
  JNIEnv *env = scope.env;
  // Reserve the native owner before Java creates off-heap buffers. Every acquired
  // batch then has a deferred cleanup slot, including descriptor-build failures.
  ArrowExportOwner *owner = calloc(1, sizeof(*owner));
  if (owner == NULL) {
    engine_error(error, 8, "cannot allocate Java Arrow export owner");
    scope_leave(&scope);
    defer_release(lease, NULL);
    return -1;
  }
  atomic_init(&owner->references, 1);
  owner->lease = lease;
  owner->defer_release = defer_release;
  jobject batch = (*env)->CallObjectMethod(env, query->object, query->api->next_arrow);
  int result = engine_exception(env, query->api, error);
  if (result != 0 || batch == NULL) {
    if (batch != NULL) {
      (*env)->CallVoidMethod(env, batch, query->api->close[ENGINE_BATCH]);
      (void) engine_exception(env, query->api, NULL);
      (*env)->DeleteLocalRef(env, batch);
    }
    scope_leave(&scope);
    arrow_owner_unref(owner);
    return result;
  }
  EngineHandle *batch_handle = engine_handle(env, query->api, batch, ENGINE_BATCH, error);
  if (batch_handle == NULL) {
    (*env)->DeleteLocalRef(env, batch);
    scope_leave(&scope);
    arrow_owner_unref(owner);
    return -1;
  }
  owner->batch = batch_handle;
  jobject node = (*env)->CallObjectMethod(env, batch, query->api->descriptor);
  result = engine_exception(env, query->api, error);
  size_t remaining_nodes = SPARK_CONNECT_MAX_ARROW_NODES;
  if (result == 0) result = arrow_build_node(env, query->api, node, owner, out, 0,
      &remaining_nodes, error);
  if (node != NULL) (*env)->DeleteLocalRef(env, node);
  (*env)->DeleteLocalRef(env, batch);
  scope_leave(&scope);
  arrow_owner_unref(owner);
  if (result != 0) {
    engine_normalize_error(error, "cannot export Java Arrow batch");
    return -1;
  }
  return 1;
}

int spark_connect_engine_analyze(void *raw_session, int32_t opcode, const uint8_t *relation, size_t len,
    void *raw_task, BridgeBuffer *out, BridgeBuffer *error) {
  clear_buffer(out);
  clear_buffer(error);
  EngineHandle *session = raw_session;
  EngineHandle *task = raw_task;
  if (engine_check(session, ENGINE_SESSION, error) != 0
      || engine_task_check(session, task, error) != 0) return -1;
  if (out == NULL) {
    engine_error(error, 13, "missing Java engine analysis buffer");
    return -1;
  }
  JavaScope scope;
  if (engine_scope(session, &scope, error) != 0) return -1;
  JNIEnv *env = scope.env;
  jbyteArray input = byte_array(env, relation, len, error);
  jbyteArray bytes = input == NULL ? NULL : (*env)->CallObjectMethod(env,
      session->object, session->api->analyze, (jint) opcode, input, task->object);
  int result = engine_exception(env, session->api, error);
  if (result == 0 && bytes != NULL) result = array_buffer(env, bytes, out, error);
  else if (result == 0) {
    engine_normalize_error(error, "Java engine returned no analysis result");
    result = -1;
  }
  if (bytes != NULL) (*env)->DeleteLocalRef(env, bytes);
  if (input != NULL) (*env)->DeleteLocalRef(env, input);
  scope_leave(&scope);
  if (result < 0) engine_normalize_error(error, "cannot analyze Java engine relation");
  return result;
}

int spark_connect_engine_config(void *raw_session, int32_t opcode, const uint8_t *fields, size_t len,
    void *raw_task, BridgeBuffer *out, BridgeBuffer *error) {
  clear_buffer(out);
  clear_buffer(error);
  EngineHandle *session = raw_session;
  EngineHandle *task = raw_task;
  if (engine_check(session, ENGINE_SESSION, error) != 0
      || engine_task_check(session, task, error) != 0) return -1;
  if (out == NULL) {
    engine_error(error, 13, "missing Java engine config buffer");
    return -1;
  }
  JavaScope scope;
  if (engine_scope(session, &scope, error) != 0) return -1;
  JNIEnv *env = scope.env;
  jbyteArray input = byte_array(env, fields, len, error);
  jbyteArray bytes = input == NULL ? NULL : (*env)->CallObjectMethod(env,
      session->object, session->api->config, (jint) opcode, input, task->object);
  int result = engine_exception(env, session->api, error);
  if (result == 0 && bytes != NULL) result = array_buffer(env, bytes, out, error);
  else if (result == 0) {
    engine_normalize_error(error, "Java engine returned no config result");
    result = -1;
  }
  if (bytes != NULL) (*env)->DeleteLocalRef(env, bytes);
  if (input != NULL) (*env)->DeleteLocalRef(env, input);
  scope_leave(&scope);
  if (result < 0) engine_normalize_error(error, "cannot configure Java engine session");
  return result;
}

void spark_connect_engine_cancel(void *raw_task) {
  EngineHandle *task = raw_task;
  if (engine_check(task, ENGINE_TASK, NULL) != 0) return;
  JavaScope scope;
  if (engine_scope(task, &scope, NULL) != 0) return;
  (*scope.env)->CallVoidMethod(scope.env, task->object, task->api->cancel);
  (void) engine_exception(scope.env, task->api, NULL);
  scope_leave(&scope);
}

int spark_connect_engine_free_checked(void *raw_handle, BridgeBuffer *error) {
  clear_buffer(error);
  EngineHandle *handle = raw_handle;
  if (handle == NULL) return 0;
  EngineApi *api = handle->api;
  if (handle->kind == ENGINE_FACTORY
      && atomic_load_explicit(&api->context->references, memory_order_acquire) != 1) {
    engine_error(error, 13, "Java engine worker attachments remain during factory cleanup");
    return -1;
  }
  JavaScope scope;
  if (engine_scope(handle, &scope, error) != 0) return -1;
  JNIEnv *env = scope.env;
  (*env)->CallVoidMethod(env, handle->object, api->close[handle->kind]);
  if (engine_exception(env, api, error) != 0) {
    scope_leave(&scope);
    return -1;
  }
  (*env)->DeleteGlobalRef(env, handle->object);
  // Delete final API/context globals in this valid scope, never by entering a
  // second fallible JNI scope after ownership has already been consumed.
  int final_api = atomic_fetch_sub_explicit(&api->references, 1, memory_order_acq_rel) == 1;
  JavaContext *context = api->context;
  int final_context = 0;
  if (final_api) {
    engine_delete_classes(env, api);
    final_context = atomic_fetch_sub_explicit(&context->references, 1,
        memory_order_acq_rel) == 1;
    if (final_context) {
      scope_restore_loader(&scope);
      if (context->loader != NULL) (*env)->DeleteGlobalRef(env, context->loader);
      (*env)->DeleteGlobalRef(env, context->thread_class);
    }
  }
  scope_leave(&scope);
  free(handle);
  if (final_context) free(context);
  if (final_api) free(api);
  return 0;
}

void spark_connect_engine_free(void *raw_handle) {
  BridgeBuffer error = {0};
  if (spark_connect_engine_free_checked(raw_handle, &error) != 0) {
    // Startup failures can have no registered Rust owner to retry. Preserve the
    // C/Java resources for process lifetime and make the incomplete cleanup visible.
    fputs("JNI cleanup is incomplete; retained cleanup failed: ", stderr);
    (void) fwrite(error.data, 1, error.len, stderr);
    fputc('\n', stderr);
  }
  spark_connect_buffer_free(&error);
}

int64_t spark_connect_engine_native_start(void *raw_env, void *clazz, void *raw_address, void *raw_factory) {
  (void) clazz;
  JNIEnv *env = raw_env;
  BridgeBuffer error = {0};
  BridgeBuffer address = {0};
  if (raw_address == NULL || raw_factory == NULL) {
    engine_error(&error, 3, "native address and engine factory must be non-null");
    throw_error(env, &error);
    return 0;
  }
  if (array_buffer(env, raw_address, &address, &error) != 0) {
    engine_normalize_error(&error, "cannot read native engine address");
    throw_error(env, &error);
    return 0;
  }
  EngineHandle *factory = engine_factory(env, raw_factory, &error);
  if (factory == NULL) {
    spark_connect_buffer_free(&address);
    throw_error(env, &error);
    return 0;
  }
  /* spark_connect_engine_start owns factory on both success and failure. */
  int64_t id = spark_connect_engine_start(address.data, address.len, factory, &error);
  spark_connect_buffer_free(&address);
  if (id == 0) throw_error(env, &error);
  else spark_connect_buffer_free(&error);
  return id;
}
