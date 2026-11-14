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

//! JVM-loaded JNI host. Java objects are retained by the C shim, never by JNIEnv pointers.

mod engine;

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    ffi::c_void,
    net::SocketAddr,
    ptr,
    sync::{
        atomic::{AtomicBool, AtomicI64, Ordering},
        mpsc as std_mpsc, Arc, Mutex, OnceLock,
    },
    time::Duration,
};

use tokio::sync::oneshot;
use tonic::Code;

const MAX_BYTES: usize = 64 * 1024 * 1024;

#[repr(C)]
pub struct BridgeBuffer {
    data: *mut u8,
    len: usize,
}
impl Default for BridgeBuffer {
    fn default() -> Self {
        Self {
            data: ptr::null_mut(),
            len: 0,
        }
    }
}

extern "C" {
    fn spark_connect_buffer_free(buffer: *mut BridgeBuffer);
}

fn take_buffer(mut buffer: BridgeBuffer) -> Vec<u8> {
    let result = if buffer.data.is_null() || buffer.len == 0 {
        Vec::new()
    } else {
        // C returns malloc-owned buffers valid until spark_connect_buffer_free.
        unsafe { std::slice::from_raw_parts(buffer.data, buffer.len).to_vec() }
    };
    unsafe {
        spark_connect_buffer_free(&mut buffer);
    }
    result
}

struct ServerEntry {
    address: String,
    shutdown: Mutex<Option<oneshot::Sender<()>>>,
    done: Mutex<std_mpsc::Receiver<Result<(), String>>>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    stopping: Mutex<()>,
    drained: AtomicBool,
    backend: Arc<engine::JvmEngine>,
}
fn servers() -> &'static Mutex<HashMap<i64, Arc<ServerEntry>>> {
    static SERVERS: OnceLock<Mutex<HashMap<i64, Arc<ServerEntry>>>> = OnceLock::new();
    SERVERS.get_or_init(Mutex::default)
}

thread_local! {
    // Attach lazily on the first engine action. The runtime's stop hook runs on
    // that same blocking worker; each individual call still restores its context.
    static ENGINE_ATTACHMENT: RefCell<Option<engine::WorkerAttachment>> = const { RefCell::new(None) };
    static ENGINE_ATTACH_ERROR_REPORTED: Cell<bool> = const { Cell::new(false) };
}
fn attach_engine_worker(engine: &Arc<engine::JvmEngine>) {
    ENGINE_ATTACHMENT.with(|slot| {
        if let Some(attachment) = slot.borrow().as_ref() {
            debug_assert!(attachment.belongs_to(engine));
            return;
        }
        match engine.attach_worker() {
            Ok(attachment) => *slot.borrow_mut() = Some(attachment),
            Err(error) if error.code() == Code::Unavailable => {
                // Shutdown can win after blocking work is reserved. Its task
                // cancellation and normal per-call checks handle that race.
            }
            Err(error) => {
                // The existing per-call JNI scope remains a safe fallback.
                // Retry transient failures, but report at most once per worker.
                ENGINE_ATTACH_ERROR_REPORTED.with(|reported| {
                    if !reported.replace(true) {
                        eprintln!(
                            "JVM worker attachment failed; using per-call attachment: {error}"
                        );
                    }
                });
            }
        }
    });
}
fn start_host(address: &[u8], backend: Arc<engine::JvmEngine>) -> Result<i64, String> {
    static NEXT: AtomicI64 = AtomicI64::new(1);
    #[allow(deprecated)] // Renamed only in Rust 1.95; preserve the declared MSRV.
    let id = NEXT
        .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |id| id.checked_add(1))
        .map_err(|_| "JNI server IDs exhausted")?;
    let address: SocketAddr = std::str::from_utf8(address)
        .map_err(|_| "listen address must be UTF-8")?
        .parse()
        .map_err(|_| "listen address must be an IP:port")?;
    let listener = std::net::TcpListener::bind(address).map_err(|e| e.to_string())?;
    listener.set_nonblocking(true).map_err(|e| e.to_string())?;
    let address = listener
        .local_addr()
        .map_err(|e| e.to_string())?
        .to_string();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .max_blocking_threads(256)
        .on_thread_stop(|| {
            let attachment = ENGINE_ATTACHMENT.with(|slot| slot.borrow_mut().take());
            drop(attachment);
        })
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    let (shutdown, stopped) = oneshot::channel();
    let (done, completed) = std_mpsc::channel();
    let worker_backend = backend.clone();
    let thread = std::thread::Builder::new()
        .name("spark-connect-jni-server".into())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.block_on(async {
                    let listener =
                        tokio::net::TcpListener::from_std(listener).map_err(|e| e.to_string())?;
                    spark_connect_server::serve_with_engine(
                        listener,
                        Arc::new(engine::EngineFactory(worker_backend.clone())),
                        async {
                            let _ = stopped.await;
                        },
                    )
                    .await
                    .map_err(|e| e.to_string())
                })
            }))
            .unwrap_or_else(|_| Err("Rust listener panicked".into()));
            worker_backend.shutdown();
            // Never destroy a JVM global reference or runtime while attached work remains.
            worker_backend.wait_drained();
            runtime.shutdown_timeout(Duration::from_secs(1));
            let result = result.and_then(|()| worker_backend.verify_workers_detached());
            let _ = done.send(result);
        })
        .map_err(|e| e.to_string())?;
    servers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(
            id,
            Arc::new(ServerEntry {
                address,
                shutdown: Mutex::new(Some(shutdown)),
                done: Mutex::new(completed),
                thread: Mutex::new(Some(thread)),
                stopping: Mutex::new(()),
                drained: AtomicBool::new(false),
                backend,
            }),
        );
    Ok(id)
}
fn find_server(id: i64) -> Result<Arc<ServerEntry>, String> {
    servers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&id)
        .cloned()
        .ok_or_else(|| "invalid or closed native server handle".into())
}
fn stop_server(id: i64) -> Result<(), String> {
    let server = find_server(id)?;
    let _stopping = server
        .stopping
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    server.backend.retry_failed_cleanup();
    if server
        .thread
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .is_none()
    {
        if !server.drained.load(Ordering::SeqCst) {
            return Err("JNI server drain state is unknown after a listener-thread failure; references retained until JVM exit".into());
        }
        server.backend.shutdown();
        server.backend.verify_workers_detached()?;
        server.backend.release_factory()?;
        servers()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&id);
        return Ok(());
    }
    server.backend.shutdown();
    if let Some(shutdown) = server
        .shutdown
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
    {
        let _ = shutdown.send(());
    }
    let completed = server
        .done
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let result = loop {
        if let Some(error) = server.backend.cleanup_failure() {
            // The listener thread still owns the runtime and waits for the
            // retained handles/leases. A later explicit close can retry them.
            return Err(error);
        }
        let remaining = deadline.saturating_duration_since(std::time::Instant::now());
        if remaining.is_zero() {
            return Err("JNI stop is incomplete; active JVM calls retained safely, retry close after they finish".into());
        }
        match completed.recv_timeout(remaining.min(Duration::from_millis(20))) {
            Ok(result) => {
                // Only this completion certifies drain and runtime stop.
                server.drained.store(true, Ordering::SeqCst);
                break result;
            }
            Err(std_mpsc::RecvTimeoutError::Timeout) => {}
            Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                break Err("JNI server completion channel closed without confirming teardown; drain state unknown".into());
            }
        }
    };
    drop(completed);
    let mut result = result;
    if let Some(thread) = server
        .thread
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .take()
    {
        if thread.join().is_err() {
            result = Err("JNI server thread panicked".into());
        }
    }
    // NativeEngineServer retains its handle on any error. Keep even a fully stopped
    // entry until the owner retries, rather than returning a stale handle.
    if result.is_ok() {
        // Join and confirmed drain precede the synchronous Java factory close.
        // Retaining server here prevents a concurrent final core Drop.
        result = server.backend.release_factory();
    }
    if result.is_ok() {
        servers()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&id);
    }
    result
}

unsafe fn write_buffer(out: *mut BridgeBuffer, bytes: &[u8]) {
    if out.is_null() {
        return;
    }
    if bytes.is_empty() {
        *out = BridgeBuffer::default();
        return;
    }
    let data = libc::malloc(bytes.len()).cast::<u8>();
    if data.is_null() {
        *out = BridgeBuffer::default();
        return;
    }
    ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len());
    *out = BridgeBuffer {
        data,
        len: bytes.len(),
    };
}
fn ffi_result<T>(
    error: *mut BridgeBuffer,
    action: impl FnOnce() -> Result<T, String>,
) -> Option<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)) {
        Ok(Ok(result)) => Some(result),
        Ok(Err(message)) => {
            unsafe {
                write_buffer(error, message.as_bytes());
            }
            None
        }
        Err(_) => {
            unsafe {
                write_buffer(error, b"panic contained at JNI boundary");
            }
            None
        }
    }
}

/// # Safety
/// Output pointers must refer to writable BridgeBuffer objects owned by bridge.c.
#[no_mangle]
pub unsafe extern "C" fn spark_connect_address(
    id: i64,
    out: *mut BridgeBuffer,
    error: *mut BridgeBuffer,
) -> i32 {
    ffi_result(error, || {
        let server = find_server(id)?;
        write_buffer(out, server.address.as_bytes());
        Ok(())
    })
    .map_or(-1, |_| 0)
}
/// # Safety
/// Error must be null or a writable BridgeBuffer owned by bridge.c.
#[no_mangle]
pub unsafe extern "C" fn spark_connect_stop(id: i64, error: *mut BridgeBuffer) -> i32 {
    ffi_result(error, || stop_server(id)).map_or(-1, |_| 0)
}

/// # Safety
/// C supplies a readable address and transfers an owned typed Java engine handle.
#[no_mangle]
pub unsafe extern "C" fn spark_connect_engine_start(
    address: *const u8,
    length: usize,
    factory: *mut c_void,
    error: *mut BridgeBuffer,
) -> i64 {
    ffi_result(error, || {
        if factory.is_null() {
            return Err("null Spark engine".into());
        }
        let backend = engine::JvmEngine::new(factory);
        if address.is_null() && length != 0 {
            return Err("null listen address".into());
        }
        let address = if length == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(address, length)
        };
        start_host(address, backend)
    })
    .unwrap_or(0)
}

extern "C" {
    fn spark_connect_engine_native_start(
        env: *mut c_void,
        class: *mut c_void,
        address: *mut c_void,
        factory: *mut c_void,
    ) -> i64;
    fn spark_connect_engine_native_address(
        env: *mut c_void,
        class: *mut c_void,
        id: i64,
    ) -> *mut c_void;
    fn spark_connect_engine_native_stop(env: *mut c_void, class: *mut c_void, id: i64);
}
/// # Safety
/// The JVM supplies valid JNI arguments for NativeEngineServer.nativeStart.
#[no_mangle]
pub unsafe extern "system" fn Java_org_apache_spark_sql_connect_nativebridge_engine_NativeEngineServer_nativeStart(
    env: *mut c_void,
    class: *mut c_void,
    address: *mut c_void,
    factory: *mut c_void,
) -> i64 {
    spark_connect_engine_native_start(env, class, address, factory)
}
/// # Safety
/// The JVM supplies valid JNI arguments for NativeEngineServer.nativeAddress.
#[no_mangle]
pub unsafe extern "system" fn Java_org_apache_spark_sql_connect_nativebridge_engine_NativeEngineServer_nativeAddress(
    env: *mut c_void,
    class: *mut c_void,
    id: i64,
) -> *mut c_void {
    spark_connect_engine_native_address(env, class, id)
}
/// # Safety
/// The JVM supplies valid JNI arguments for NativeEngineServer.nativeStop.
#[no_mangle]
pub unsafe extern "system" fn Java_org_apache_spark_sql_connect_nativebridge_engine_NativeEngineServer_nativeStop(
    env: *mut c_void,
    class: *mut c_void,
    id: i64,
) {
    spark_connect_engine_native_stop(env, class, id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_server_errors_require_a_confirmed_drain_before_retry_succeeds() {
        for (id, disconnect) in [(-1, false), (-2, true)] {
            let backend = engine::JvmEngine::new(ptr::null_mut());
            let (done, completed) = std_mpsc::channel();
            let thread = std::thread::spawn(move || {
                if !disconnect {
                    done.send(Err("listener failed".into())).unwrap();
                }
            });
            servers().lock().unwrap().insert(
                id,
                Arc::new(ServerEntry {
                    address: "127.0.0.1:0".into(),
                    shutdown: Mutex::new(None),
                    done: Mutex::new(completed),
                    thread: Mutex::new(Some(thread)),
                    stopping: Mutex::new(()),
                    drained: AtomicBool::new(false),
                    backend,
                }),
            );
            assert!(stop_server(id).is_err());
            assert!(find_server(id).unwrap().thread.lock().unwrap().is_none());
            if disconnect {
                let retry = stop_server(id).unwrap_err();
                assert!(retry.contains("drain state is unknown"), "{retry}");
                assert!(find_server(id).is_ok());
                // This synthetic entry owns no JNI objects; remove the test
                // fixture explicitly. Real unknown-drain entries stay retained.
                servers().lock().unwrap().remove(&id);
            } else {
                stop_server(id).unwrap();
                assert!(find_server(id).is_err());
            }
        }
    }

    #[test]
    fn completed_engine_server_retry_requires_detached_workers() {
        let core = engine::JvmEngine::new(ptr::null_mut());
        core.set_worker_attachment_count_for_test(1);
        let (_done, completed) = std_mpsc::channel();
        let id = -303;
        servers().lock().unwrap().insert(
            id,
            Arc::new(ServerEntry {
                address: "127.0.0.1:0".into(),
                shutdown: Mutex::new(None),
                done: Mutex::new(completed),
                thread: Mutex::new(None),
                stopping: Mutex::new(()),
                drained: AtomicBool::new(true),
                backend: core.clone(),
            }),
        );
        let error = stop_server(id).unwrap_err();
        assert!(error.contains("1 JNI worker attachments remain"), "{error}");
        assert!(find_server(id).is_ok());
        // Simulate the previously unfinished worker successfully detaching.
        core.set_worker_attachment_count_for_test(0);
        stop_server(id).unwrap();
        assert!(find_server(id).is_err());
    }

    #[test]
    fn failed_cleanup_preserves_the_listener_and_drain_state_until_explicit_retry() {
        let core = engine::JvmEngine::new(ptr::null_mut());
        let (shutdown, stopping) = oneshot::channel();
        let (done, completed) = std_mpsc::channel();
        let listener_core = core.clone();
        let thread = std::thread::spawn(move || {
            stopping.blocking_recv().unwrap();
            // Failure arrives after stop_server's retry snapshot. The listener
            // must keep waiting with its runtime alive until another close.
            listener_core.fail_export_cleanup_for_test();
            listener_core.wait_drained();
            done.send(Ok(())).unwrap();
        });
        let id = -304;
        servers().lock().unwrap().insert(
            id,
            Arc::new(ServerEntry {
                address: "127.0.0.1:0".into(),
                shutdown: Mutex::new(Some(shutdown)),
                done: Mutex::new(completed),
                thread: Mutex::new(Some(thread)),
                stopping: Mutex::new(()),
                drained: AtomicBool::new(false),
                backend: core,
            }),
        );
        let error = stop_server(id).unwrap_err();
        assert!(error.starts_with("JNI cleanup is incomplete; retained cleanup failed:"));
        let retained = find_server(id).unwrap();
        assert!(!retained.drained.load(Ordering::SeqCst));
        assert!(retained.thread.lock().unwrap().is_some());
        stop_server(id).unwrap();
        assert!(retained.drained.load(Ordering::SeqCst));
        assert!(retained.thread.lock().unwrap().is_none());
        assert!(find_server(id).is_err());
    }
}
