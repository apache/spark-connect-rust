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

use spark_connect_macros::spark_wasm_udf;

#[spark_wasm_udf]
mod fixtures {
    /// The value of the environment variable `name` in the module, if it is set.
    pub fn env_var(name: String) -> Option<String> {
        std::env::var(name).ok()
    }

    /// Whether the module can list the directory `path`, which it never can: the
    /// runner preopens no directory.
    pub fn can_list(path: String) -> bool {
        std::fs::read_dir(path).is_ok()
    }

    /// Returns `s` after writing it to standard output and error, which the runner
    /// discards.
    pub fn echo(s: String) -> String {
        println!("{s}");
        eprintln!("{s}");
        s
    }

    /// Exits the module with `code` instead of returning.
    pub fn exit_with(code: i32) -> i32 {
        std::process::exit(code)
    }
}
