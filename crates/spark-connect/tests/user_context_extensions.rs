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

//! The user context extensions of a session, as PySpark's `SparkConnectClient` keeps them: those
//! of each thread and the global ones, which the requests of a thread carry.

use spark_connect::SparkSession;

fn offline() -> SparkSession {
    // The channel connects lazily: the extensions are kept without a server.
    SparkSession::builder()
        .remote("sc://127.0.0.1:1")
        .get_or_create()
        .unwrap()
}

fn extension(name: &str) -> prost_types::Any {
    prost_types::Any {
        type_url: format!("type.googleapis.com/test.{name}"),
        value: name.as_bytes().to_vec(),
    }
}

#[test]
fn extensions_are_of_their_thread_or_global() {
    let session = offline();
    let local = session.add_threadlocal_user_context_extension(extension("local"));
    let global = session.add_global_user_context_extension(extension("global"));
    assert!(local.starts_with("threadlocal_") && global.starts_with("global_"));
    assert_eq!(
        session.threadlocal_user_context_extensions(),
        vec![(local.clone(), extension("local"))]
    );
    assert_eq!(
        session.global_user_context_extensions(),
        vec![(global.clone(), extension("global"))]
    );

    let other = session.clone();
    let (own, shared) = std::thread::spawn(move || {
        (
            other.threadlocal_user_context_extensions(),
            other.global_user_context_extensions(),
        )
    })
    .join()
    .unwrap();
    assert!(own.is_empty());
    assert_eq!(shared, vec![(global.clone(), extension("global"))]);

    // A thread removes its own extensions only.
    let other = session.clone();
    let removed = local.clone();
    std::thread::spawn(move || other.remove_user_context_extension(&removed))
        .join()
        .unwrap();
    assert_eq!(session.threadlocal_user_context_extensions().len(), 1);
    session.remove_user_context_extension(&local);
    assert!(session.threadlocal_user_context_extensions().is_empty());
    session.remove_user_context_extension(&global);
    assert!(session.global_user_context_extensions().is_empty());
}

#[test]
fn clearing_removes_those_of_the_thread_and_the_global_ones() {
    let session = offline();
    session.add_threadlocal_user_context_extension(extension("a"));
    session.add_global_user_context_extension(extension("b"));
    session.clear_user_context_extensions();
    assert!(session.threadlocal_user_context_extensions().is_empty());
    assert!(session.global_user_context_extensions().is_empty());
}

#[test]
fn a_new_session_has_no_extensions() {
    let session = offline();
    session.add_global_user_context_extension(extension("global"));
    let fresh = session.new_session();
    assert!(fresh.global_user_context_extensions().is_empty());
}
