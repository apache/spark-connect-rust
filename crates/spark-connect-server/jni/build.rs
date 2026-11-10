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

fn main() {
    let platform = match std::env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
        "macos" => "darwin",
        "linux" => "linux",
        other => panic!(
            "unsupported JNI target: {other}; spark-connect-jni requires macOS or Linux with pthreads"
        ),
    };
    let java_home = std::env::var_os("JAVA_HOME")
        .expect("Set JAVA_HOME to a JDK with JNI headers, or exclude spark-connect-jni from workspace commands");
    let include = std::path::PathBuf::from(java_home).join("include");
    cc::Build::new()
        .file("native/bridge.c")
        .include(&include)
        .include(include.join(platform))
        .std("c11")
        .warnings(true)
        .compile("spark_connect_jni_bridge");
    println!("cargo:rerun-if-env-changed=JAVA_HOME");
    println!("cargo:rerun-if-changed=native/bridge.c");
    println!("cargo:rerun-if-changed=native/bridge.h");
}
