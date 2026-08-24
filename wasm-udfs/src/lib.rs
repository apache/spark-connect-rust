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

//! Example UDFs written as plain Rust functions.
//!
//! `#[spark_wasm_udf]` exports each function to WebAssembly and generates a
//! `udf::<name>()` constructor (with the Spark signature inferred). The
//! functions show the range of supported types: scalars, `bool`, `String`,
//! `Vec<T>` (array), and `Option<T>` (nullable).

use spark_connect_macros::spark_wasm_udf;

#[spark_wasm_udf]
mod udfs {
    /// `i64 -> i64`
    pub fn add_one(x: i64) -> i64 {
        x + 1
    }

    /// `f64 -> f64`
    pub fn celsius_to_fahrenheit(c: f64) -> f64 {
        c * 9.0 / 5.0 + 32.0
    }

    /// `String -> String`
    pub fn shout(s: String) -> String {
        format!("{}!", s.to_uppercase())
    }

    /// `String -> bool`
    pub fn is_palindrome(s: String) -> bool {
        let chars: Vec<char> = s.chars().collect();
        chars.iter().eq(chars.iter().rev())
    }

    /// `Vec<i64> -> i64` (array argument)
    pub fn sum(xs: Vec<i64>) -> i64 {
        xs.iter().sum()
    }

    /// `Vec<String> -> String` (array of strings)
    pub fn join_words(words: Vec<String>) -> String {
        words.join(" ")
    }

    /// `Option<i64> -> Option<i64>` (nullable in and out)
    pub fn double_or_null(x: Option<i64>) -> Option<i64> {
        x.map(|v| v * 2)
    }
}
