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

//! The protobuf messages that the modules of PySpark that build their own, such as `pyspark.ml`,
//! exchange with the client: the `spark.connect.DataType` of a type, as bytes, which the Python
//! side parses with the generated `pyspark.sql.connect.proto`, for
//! `pyspark.sql.connect.types.pyspark_types_to_proto_types` and
//! `proto_schema_to_pyspark_data_type`.

use prost::Message;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use spark_connect::types::DataType;
use spark_connect_proto as proto;

use crate::errors::ResultExt;

/// The bytes of the `spark.connect.DataType` of `data_type`, as the client sends the type.
#[pyfunction]
pub fn _data_type_proto<'py>(
    py: Python<'py>,
    data_type: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let data_type = crate::types::py_to_data_type(data_type)?;
    Ok(PyBytes::new(py, &data_type.to_proto().encode_to_vec()))
}

/// The type of the bytes of a `spark.connect.DataType`, as the client reads the types of the
/// server.
#[pyfunction]
pub fn _data_type_of_proto(py: Python<'_>, data_type: &[u8]) -> PyResult<Py<PyAny>> {
    let data_type = proto::DataType::decode(data_type)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let data_type = DataType::from_proto(&data_type).to_pyerr()?;
    crate::types::data_type_to_py(py, &data_type)
}
