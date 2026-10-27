/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Arrow <-> Python conversion one value at a time, for a per-row UDF loop that has to match the
//! `SQL_ARROW_BATCHED_UDF` loop in `pyspark.worker.read_udfs`.
//!
//! [`InputColumns`] reads a batch's input columns cell by cell, as PySpark's input step,
//! `_convert_arrow_udf_input`, reads them: `ArrowTableToRowsConversion._to_pylist`, then the
//! column's converter. [`ResultColumn`] builds one UDF's results into an Arrow array, as PySpark's
//! output step does: `pa.array(values, type=T)`, retried as
//! `pa.array(values).cast(T, safe=safecheck)` on `ArrowInvalid`. It appends built-in results, and
//! variant results whose parts are `bytes`, to an arrow-rs builder and hands any other result to
//! PyArrow. A caller keeps no list of types of its own: it asks [`input_spec`] about each input
//! column and [`result_spec`] about each result.
//!
//! A spec returns `Ok(Err(reason))` for an unsupported conversion; the caller may select its
//! existing Python implementation in that case. `Err(PyErr)` instead means inspecting the
//! supplied type/converter failed and must not silently become an unsupported-type fallback.
//! Runtime conversion failures retain Python's errors and result ordering. Nested, decimal,
//! and timestamp conversion is not implemented natively by this UDF policy.

use std::collections::HashMap;

use arrow::array::{ArrayRef, StructArray};
use arrow::datatypes::{DataType, Field, Fields, Schema};
use pyo3::prelude::*;

mod input;
mod result;

pub use input::{input_spec, InputColumns, InputSpec};
pub use result::{result_spec, Finished, ResultColumn, ResultSpec};

/// The struct Spark gives `VariantType`, with parts of `part`: `Binary` from PySpark's
/// `to_arrow_type`, or `LargeBinary` from the JVM under `useLargeVarTypes`. Its tagged `metadata`
/// field tells it apart from a struct of two binaries.
fn variant_fields(part: DataType) -> Fields {
    Fields::from(vec![
        Field::new("value", part.clone(), false),
        Field::new("metadata", part, false)
            .with_metadata(HashMap::from([("variant".to_owned(), "true".to_owned())])),
    ])
}

/// Whether `spark_type` is exactly `sql_types.<name>`, a type with a supported native converter.
/// `sql_types` is the caller-supplied backend module that parsed `spark_type`.
fn is_spark_type(
    spark_type: &Bound<'_, PyAny>,
    sql_types: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<bool> {
    Ok(spark_type.get_type().is(&sql_types.getattr(name)?))
}

/// Imports `schema`, a `pa.Schema`, such as the input schema a UDF's batches follow.
pub fn import_schema(schema: &Bound<'_, PyAny>) -> PyResult<Schema> {
    crate::bridge::import_schema(schema)
}

/// Imports `data_type`, a `pa.DataType`, rejecting root extension types whose metadata
/// cannot be represented by a bare Rust data type.
pub fn import_data_type(data_type: &Bound<'_, PyAny>) -> PyResult<DataType> {
    crate::bridge::import_data_type(data_type)
}

fn import_batch(batch: &Bound<'_, PyAny>) -> PyResult<StructArray> {
    Ok(StructArray::from(crate::bridge::import_batch(batch)?))
}

/// Hands `array` to pyarrow, whose `pa.Array` is `array_type`, without copying it. pyarrow moves
/// both structs' contents out, so dropping them afterwards releases nothing.
fn export_array<'py>(
    array_type: &Bound<'py, PyAny>,
    array: ArrayRef,
) -> PyResult<Bound<'py, PyAny>> {
    crate::bridge::export_array_with_type(array_type, array)
}
