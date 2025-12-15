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

use std::fs;

fn find_spark_version() -> &'static str {
    let version = std::env::var("SPARK_VERSION").unwrap_or("4.1.0".to_string());
    let version_triplet = version.splitn(3, '.').collect::<Vec<_>>();
    match version_triplet[..3] {
        ["3", "5", _] => "./protobuf/spark-3.5/",
        _ => {
            println!("cargo::rustc-cfg=spark41");
            "./protobuf/spark-4.1/"
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spark_version = find_spark_version();
    let file_path = format!("{}spark/connect/", spark_version);
    let files = fs::read_dir(&file_path)?;

    let mut file_paths: Vec<String> = vec![];

    for file in files {
        let entry = file?.path();
        file_paths.push(entry.to_str().unwrap().to_string());
    }

    tonic_prost_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .build_server(false)
        .build_client(true)
        .build_transport(true)
        .compile_protos(file_paths.as_ref(), &[spark_version.to_string()])?;

    Ok(())
}
