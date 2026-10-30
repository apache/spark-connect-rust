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

use arrow::pyarrow::{FromPyArrow, Table};
use pyo3::prelude::*;
use spark_connect::session::SparkSession;

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

pub(crate) fn create_dataframe(
    py: Python<'_>,
    session: &SparkSession,
    table: &Bound<'_, PyAny>,
    schema: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyDataFrame> {
    let helper = py.import("pyspark.sql._arrow_input")?;
    // Resolve DDL with the same converter as the existing local-data path.
    let schema = match schema {
        Some(s) if s.extract::<String>().is_ok() => Some(crate::types::data_type_to_py(
            py,
            &crate::types::py_to_data_type(s)?,
        )?),
        Some(s) => Some(s.clone().unbind()),
        None => None,
    };
    let schema = schema.unwrap_or_else(|| py.None());
    // Empty input needs only the explicit schema, and no server configuration.
    let empty: bool = table.call_method0("__len__")?.extract::<usize>()? == 0;
    let (timestamp_type, timezone, large_types) = if empty {
        (None, None, None)
    } else {
        py.detach(|| -> PyResult<_> {
            let values = session
                .conf()
                .get_many(&[
                    "spark.sql.timestampType",
                    "spark.sql.session.timeZone",
                    "spark.sql.execution.arrow.useLargeVarTypes",
                ])
                .to_pyerr()?;
            let mut values = values.into_iter();
            Ok((
                values.next().flatten(),
                values.next().flatten(),
                values.next().flatten(),
            ))
        })?
    };
    let result = helper.getattr("prepare_table")?.call1((
        table,
        schema,
        timestamp_type,
        timezone,
        large_types,
    ))?;
    let normalized = result.get_item(0)?;
    let spark_schema = crate::types::py_to_data_type(&result.get_item(1)?)?;
    let names: Option<Vec<String>> = result.get_item(2)?.extract()?;
    let batches = if empty {
        Vec::new()
    } else {
        // PyArrow 10 readers support _export_to_c, even without PyCapsules.
        Table::from_pyarrow_bound(&normalized.call_method0("to_reader")?)?
            .into_inner()
            .0
    };
    let dataframe = py.detach(|| {
        session
            .create_dataframe_from_arrow(batches, spark_schema)
            .to_pyerr()
    })?;
    let dataframe = match names {
        Some(names) => dataframe.to_df(names.iter().map(String::as_str).collect()),
        None => dataframe,
    };
    Ok(PyDataFrame::new(dataframe))
}
