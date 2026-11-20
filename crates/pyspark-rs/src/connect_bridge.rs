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
//! side parses with `pyspark.sql.connect.proto`, for
//! `pyspark.sql.connect.types.pyspark_types_to_proto_types` and
//! `proto_schema_to_pyspark_data_type`.

use prost::Message;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use spark_connect::types::DataType;
use spark_connect_proto as proto;

use crate::errors::ResultExt;

/// The bytes of the `spark.connect.DataType` of `data_type`, as the client sends the type: a type
/// that PySpark's `pyspark_types_to_proto_types` has no proto of, a `StructField` or a type of
/// its class only such as `AtomicType()`, raises its `UNSUPPORTED_OPERATION`.
#[pyfunction]
pub fn _data_type_proto<'py>(
    py: Python<'py>,
    data_type: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let data_type = crate::types::py_to_proto_data_type(data_type)?;
    Ok(PyBytes::new(py, &data_type.to_proto().encode_to_vec()))
}

/// The type of the bytes of a `spark.connect.DataType`, as the client reads the types of the
/// server, with the errors of PySpark's `proto_schema_to_pyspark_data_type` for a type that it
/// does not convert (see [`unconverted`]), such as `pb2.DataType()` of no kind or a spatial type
/// of an SRID that PySpark does not take. PySpark makes a user-defined type of its Python class,
/// of no conversion of its SQL type, which is read only when the core can read it.
#[pyfunction]
pub fn _data_type_of_proto(py: Python<'_>, data_type: &[u8]) -> PyResult<Py<PyAny>> {
    let mut data_type = proto::DataType::decode(data_type)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    match unconverted(Some(&data_type)) {
        Some(Unconverted::Kind(node)) => return Err(unsupported_data_type(py, &node)),
        Some(Unconverted::Srid(srid)) => return Err(crate::types::invalid_schema_srid(py, srid)),
        None => {}
    }
    drop_unread_sql_types(&mut data_type);
    let data_type = DataType::from_proto(&data_type).to_pyerr()?;
    crate::types::schema_to_py(py, &data_type)
}

/// Drops the SQL type of each user-defined type in `data_type` that holds a type of no kind,
/// which the core does not read and PySpark does not convert.
fn drop_unread_sql_types(data_type: &mut proto::DataType) {
    use proto::data_type::Kind;
    match &mut data_type.kind {
        Some(Kind::Udt(udt)) => {
            if udt.sql_type.as_deref().is_some_and(has_no_kind) {
                udt.sql_type = None;
            }
        }
        Some(Kind::Array(array)) => array
            .element_type
            .iter_mut()
            .for_each(|t| drop_unread_sql_types(t)),
        Some(Kind::Struct(st)) => st
            .fields
            .iter_mut()
            .filter_map(|field| field.data_type.as_mut())
            .for_each(drop_unread_sql_types),
        Some(Kind::Map(map)) => {
            map.key_type
                .iter_mut()
                .for_each(|t| drop_unread_sql_types(t));
            map.value_type
                .iter_mut()
                .for_each(|t| drop_unread_sql_types(t));
        }
        _ => {}
    }
}

/// Whether `data_type` is, or holds, a type of no kind, of which the core makes no type.
fn has_no_kind(data_type: &proto::DataType) -> bool {
    use proto::data_type::Kind;
    match &data_type.kind {
        None => true,
        Some(Kind::Array(array)) => array.element_type.as_deref().is_none_or(has_no_kind),
        Some(Kind::Struct(st)) => st
            .fields
            .iter()
            .any(|field| field.data_type.as_ref().is_none_or(has_no_kind)),
        Some(Kind::Map(map)) => {
            map.key_type.as_deref().is_none_or(has_no_kind)
                || map.value_type.as_deref().is_none_or(has_no_kind)
        }
        Some(Kind::Udt(udt)) => udt.sql_type.as_deref().is_some_and(has_no_kind),
        Some(_) => false,
    }
}

/// A type that PySpark's `proto_schema_to_pyspark_data_type` does not convert.
enum Unconverted {
    /// A type of no kind, such as `pb2.DataType()` or the element type that an array does not
    /// set, which PySpark reads as the empty message of the field, or of a kind that it has no
    /// branch of, an unparsed type or a nanosecond-precision timestamp: the bytes of the type, of
    /// whose proto text PySpark's last branch raises `UNSUPPORTED_OPERATION`.
    Kind(Vec<u8>),
    /// A spatial type of an SRID that PySpark's `GeometryType` or `GeographyType` does not take,
    /// which raises `ST_INVALID_SRID_VALUE`.
    Srid(i32),
}

/// The first type of `data_type` that PySpark's `proto_schema_to_pyspark_data_type` does not
/// convert, in the order in which it converts them: the element of an array, the types of the
/// fields of a struct, and the key and then the value of a map. PySpark makes a user-defined type
/// of its Python class, of no conversion of its SQL type.
fn unconverted(data_type: Option<&proto::DataType>) -> Option<Unconverted> {
    use proto::data_type::Kind;
    let Some(data_type) = data_type else {
        return Some(Unconverted::Kind(Vec::new()));
    };
    let spatial = |srid: i32, geographic: bool| {
        (!crate::types::pyspark_converts_srid(srid, geographic)).then_some(Unconverted::Srid(srid))
    };
    match &data_type.kind {
        None
        | Some(Kind::Unparsed(_))
        | Some(Kind::TimestampNtzNanos(_))
        | Some(Kind::TimestampLtzNanos(_)) => Some(Unconverted::Kind(data_type.encode_to_vec())),
        Some(Kind::Geometry(geometry)) => spatial(geometry.srid, false),
        Some(Kind::Geography(geography)) => spatial(geography.srid, true),
        Some(Kind::Array(array)) => unconverted(array.element_type.as_deref()),
        Some(Kind::Struct(st)) => st
            .fields
            .iter()
            .find_map(|field| unconverted(field.data_type.as_ref())),
        Some(Kind::Map(map)) => {
            unconverted(map.key_type.as_deref()).or_else(|| unconverted(map.value_type.as_deref()))
        }
        Some(_) => None,
    }
}

/// PySpark's `UNSUPPORTED_OPERATION` of `data type {schema}`, a `PySparkValueError`, of the proto
/// text of the type of the bytes `node`, as `str()` of its message gives it: empty of a type of no
/// kind.
fn unsupported_data_type(py: Python<'_>, node: &[u8]) -> PyErr {
    let text = || -> PyResult<String> {
        let message = py
            .import("pyspark.sql.connect.proto")?
            .getattr("DataType")?
            .call_method1("FromString", (PyBytes::new(py, node),))?;
        Ok(message.str()?.to_cow()?.into_owned())
    };
    match text() {
        Ok(text) => crate::errors::raise_pyspark(
            py,
            "PySparkValueError",
            "UNSUPPORTED_OPERATION",
            &[("operation", &format!("data type {text}"))],
        ),
        Err(err) => err,
    }
}
