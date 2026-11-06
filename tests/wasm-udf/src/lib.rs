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
    use std::cell::RefCell;

    thread_local! {
        static STATE: RefCell<()> = const { RefCell::new(()) };
    }

    pub fn sub_i32(x: i32, y: Option<i32>) -> Option<i32> {
        y.map(|y| x.wrapping_sub(y))
    }

    pub fn sub_i64(x: i64, y: Option<i64>) -> Option<i64> {
        y.map(|y| x.wrapping_sub(y))
    }

    pub fn sub_f32(x: f32, y: Option<f32>) -> Option<f32> {
        y.map(|y| x - y)
    }

    pub fn sub_f64(x: f64, y: Option<f64>) -> Option<f64> {
        y.map(|y| x - y)
    }

    pub fn and_not(x: bool, y: Option<bool>) -> Option<bool> {
        y.map(|y| x && !y)
    }

    pub fn concat(x: String, y: Option<String>) -> Option<String> {
        y.map(|y| x + &y)
    }

    pub fn concat_bytes(x: Vec<u8>, y: Option<Vec<u8>>) -> Option<Vec<u8>> {
        y.map(|y| [x, y].concat())
    }

    pub fn label(name: Option<String>, n: i64) -> String {
        format!("{}#{n}", name.unwrap_or_default())
    }

    pub fn keep_first(x: i64, _: i64) -> i64 {
        x
    }

    pub fn sum(xs: Vec<i64>) -> i64 {
        xs.iter().sum()
    }

    // Panics while `STATE` is borrowed, which stays borrowed in the trapped instance.
    pub fn non_negative(x: i64) -> i64 {
        STATE.with(|s| {
            let _borrow = s.borrow_mut();
            assert!(x >= 0, "negative input");
            x
        })
    }

    // Their exports would collide with `spark_udf_<fn>` ones under prefixes like
    // `spark_udf_batch_<fn>` and `spark_udf_scratch`.
    pub fn scratch(x: i64) -> i64 {
        x
    }

    pub fn batch_scratch(x: i64) -> i64 {
        x
    }
}
