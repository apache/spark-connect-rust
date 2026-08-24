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
//! `#[spark_wasm_udf]` does the rest: when this crate is built for
//! `wasm32-unknown-unknown` each function is exported from the module; when
//! built for the host it also generates a `<name>_udf(module)` constructor with
//! the Spark signature inferred from the Rust signature.

use spark_connect_macros::spark_wasm_udf;

/// `add_one(x) = x + 1`.
#[spark_wasm_udf]
pub fn add_one(x: i64) -> i64 {
    x + 1
}

/// Convert Celsius to Fahrenheit.
#[spark_wasm_udf]
pub fn celsius_to_fahrenheit(c: f64) -> f64 {
    c * 9.0 / 5.0 + 32.0
}

/// Greatest common divisor.
#[spark_wasm_udf]
pub fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.abs()
}
