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

//! PyArrow Table input through arrow-rs's owned C Stream importer.

use arrow::ipc::reader::StreamReader;
use arrow::pyarrow::{FromPyArrow, PyArrowException, Table};
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use spark_connect::session::SparkSession;

use crate::dataframe::PyDataFrame;

/// The record batches of a pyarrow table, at least one, so that a table without rows keeps its
/// schema, as PySpark's IPC stream does, for the relation to carry it.
pub(crate) fn table_batches(table: &Bound<'_, PyAny>) -> PyResult<Vec<RecordBatch>> {
    let py = table.py();
    let (mut batches, arrow_schema) = if holds_nul_name(py, &table.getattr("schema")?)? {
        // The Arrow C data interface holds each name as a C string, which ends at its first NUL:
        // such a table is read of its IPC stream, which holds the names whole, as the stream of
        // PySpark's createDataFrame does.
        let pa = py.import("pyarrow")?;
        let sink = pa.call_method0("BufferOutputStream")?;
        let writer = pa
            .getattr("ipc")?
            .call_method1("new_stream", (&sink, table.getattr("schema")?))?;
        writer.call_method1("write_table", (table,))?;
        writer.call_method0("close")?;
        let stream = sink.call_method0("getvalue")?.call_method0("to_pybytes")?;
        let arrow_error =
            |error: arrow::error::ArrowError| PyArrowException::new_err(error.to_string());
        let reader = StreamReader::try_new(stream.cast::<PyBytes>()?.as_bytes(), None)
            .map_err(arrow_error)?;
        let arrow_schema = reader.schema();
        let batches = reader.collect::<Result<Vec<_>, _>>().map_err(arrow_error)?;
        (batches, arrow_schema)
    } else {
        // PyArrow 10 readers support _export_to_c, even without PyCapsules.
        Table::from_pyarrow_bound(&table.call_method0("to_reader")?)?.into_inner()
    };
    // A cast drops the rows of a zero-column table, leaving no batches.
    if batches.is_empty() {
        batches.push(RecordBatch::new_empty(arrow_schema));
    }
    Ok(batches)
}

/// Whether the name of a field of the pyarrow `schema`, at any depth, holds a NUL: of a field of a
/// struct, a list, a map, a union or a run-end encoded type, also within the values of a dictionary
/// or the storage of an extension type.
fn holds_nul_name(py: Python<'_>, schema: &Bound<'_, PyAny>) -> PyResult<bool> {
    let pa = py.import("pyarrow")?;
    let dictionary = pa.getattr("DictionaryType")?;
    let extension = pa.getattr("BaseExtensionType")?;
    let mut fields = schema.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    while let Some(field) = fields.pop() {
        if field.getattr("name")?.extract::<String>()?.contains('\0') {
            return Ok(true);
        }
        let mut data_type = field.getattr("type")?;
        loop {
            data_type = if data_type.is_instance(&dictionary)? {
                data_type.getattr("value_type")?
            } else if data_type.is_instance(&extension)? {
                data_type.getattr("storage_type")?
            } else {
                break;
            };
        }
        for i in 0..data_type.getattr("num_fields")?.extract::<usize>()? {
            fields.push(data_type.call_method1("field", (i,))?);
        }
    }
    Ok(false)
}

/// A DataFrame of the pyarrow Table `table`. `options` are the settings that the caller's
/// createDataFrame read already, as PySpark reads them once before it converts any data; without
/// them, a table that is not empty reads them here. A table of the columns of a pandas DataFrame
/// (`pandas`) is cast as PySpark's conversion of the frame casts it (see `prepare_table`).
pub(crate) fn create_dataframe(
    py: Python<'_>,
    session: &SparkSession,
    table: &Bound<'_, PyAny>,
    schema: Option<&Bound<'_, PyAny>>,
    options: Option<&spark_connect::ArrowInputOptions>,
    pandas: bool,
) -> PyResult<PyDataFrame> {
    // As PySpark's createDataFrame, which first encodes each name of a list that is not a str, of
    // which most values raise the `AttributeError` of `encode`, before it reads any setting.
    if let Some(names) = schema.filter(|schema| {
        schema.is_instance_of::<pyo3::types::PyList>()
            || schema.is_instance_of::<pyo3::types::PyTuple>()
    }) {
        for name in names.try_iter()? {
            let name = name?;
            if !name.is_instance_of::<pyo3::types::PyString>() {
                name.call_method1("encode", ("utf-8",))?;
            }
        }
    }
    let helper = py.import("pyspark.sql._arrow_input")?;
    // Resolve DDL with the same converter as the existing local-data path.
    let schema = match schema {
        Some(s) if s.extract::<String>().is_ok() => Some(crate::types::data_type_to_py(
            py,
            &crate::types::schema_data_type(py, session, s)?,
        )?),
        Some(s) => Some(s.clone().unbind()),
        None => None,
    };
    let schema = schema.unwrap_or_else(|| py.None());
    // Empty input needs only the explicit schema, and no server configuration.
    let empty: bool = table.call_method0("__len__")?.extract::<usize>()? == 0;
    let options = match (empty, options) {
        (true, _) => None,
        (false, Some(options)) => Some(options.clone()),
        (false, None) => Some(crate::interrupt::interruptible_without_parts(
            py,
            session,
            || session.arrow_input_options(),
        )?),
    };
    let result = helper.getattr("prepare_table")?.call1((
        table,
        schema,
        options.as_ref().and_then(|o| o.timestamp_type()),
        options.as_ref().and_then(|o| o.timezone()),
        options.as_ref().and_then(|o| o.use_large_var_types()),
        pandas,
    ))?;
    let normalized = result.get_item(0)?;
    let schema = result.get_item(1)?;
    // A type of its class only, such as `AtomicType()`, as the schema: PySpark's relation of the
    // table is of the `json()` of the type, which the server cannot parse.
    if crate::types::is_class_type(&schema)? {
        let json = schema.call_method0("json")?.extract()?;
        return crate::session::schema_text_dataframe(session, json);
    }
    // Of the metadata as it was given, of which PySpark writes the `json()` after the cast, which
    // the relation sends as its schema, also of the attributes that the cast read of a Python
    // subclass of its own or a patched class (`pyspark_schema_json_text`); of a struct whose fields
    // hold an element that is not a `StructField`, as PySpark's `json()` writes it, of the
    // `AttributeError` of such an element of no `jsonValue` (`foreign_schema_json`).
    let (spark_schema, schema_json) = if schema.is_none() {
        (None, None)
    } else if let Some(json) = crate::types::foreign_schema_json(&schema)? {
        let unparsed = spark_connect::types::DataType::Unparsed {
            data_type_string: json.clone(),
        };
        (Some(unparsed), Some(json))
    } else {
        let json = crate::types::pyspark_schema_json_text(Some(&schema))?;
        (Some(crate::types::py_to_data_type_as_given(&schema)?), json)
    };
    if let (true, Some(json)) = (empty, &schema_json) {
        return crate::session::schema_text_dataframe(session, json.clone());
    }
    let options = options.map(|options| crate::session::with_schema_json(options, schema_json));
    // As PySpark's createDataFrame, which applies the names with `toDF`, of whose plan a str that
    // UTF-8 cannot encode fails the requests (`data_name`).
    let mut at_build = crate::plan_args::PlanArgs::new();
    let names = result.get_item(2)?;
    let names = if names.is_none() {
        None
    } else {
        let mut converted = Vec::new();
        for name in names.try_iter()? {
            converted.push(crate::session::data_name(&name?, Some(&mut at_build))?);
        }
        Some(converted)
    };
    let batches = if empty {
        Vec::new()
    } else {
        table_batches(&normalized)?
    };
    let dataframe =
        crate::interrupt::interruptible_without_parts(py, session, || match options.as_ref() {
            Some(options) => {
                session.create_dataframe_from_arrow_with_options(batches, spark_schema, options)
            }
            None => session.create_dataframe_from_arrow(batches, spark_schema),
        })?;
    // As in PySpark, column names are applied with toDF only when there are any.
    let dataframe = match names {
        Some(names) if !names.is_empty() => {
            at_build.plan(dataframe.to_df(names.iter().map(String::as_str).collect::<Vec<_>>()))
        }
        _ => dataframe,
    };
    Ok(PyDataFrame::new(dataframe))
}
