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

//! Table export checks against the installed PyArrow. No Spark server is needed.

use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, StringArray, StructArray};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_python_arrow::bridge::record_batches_to_pyarrow;

fn evaluate<'py>(py: Python<'py>, expression: &CStr) -> PyResult<Bound<'py, PyAny>> {
    let locals = PyDict::new(py);
    locals.set_item("pa", py.import("pyarrow")?)?;
    py.eval(expression, Some(&locals), Some(&locals))
}

fn assert_schema_equal(actual: &Bound<'_, PyAny>, expected: &Bound<'_, PyAny>) -> PyResult<()> {
    assert!(actual
        .call_method1("equals", (expected, true))?
        .is_truthy()?);
    Ok(())
}

#[test]
fn table_export_shares_value_and_offset_buffers_after_rust_wrappers_drop() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let numbers = Int64Array::from(vec![Some(12), None, Some(34)]);
        let strings = StringArray::from(vec![Some("first"), None, Some("last")]);
        let addresses = [
            (0, 1, numbers.values().as_ptr() as usize),
            (1, 1, strings.value_offsets().as_ptr() as usize),
            (1, 2, strings.value_data().as_ptr() as usize),
        ];
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("number", DataType::Int64, true),
                Field::new("text", DataType::Utf8, true),
            ],
            HashMap::from([("buffer\0key".into(), "shared\0values".into())]),
        ));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(numbers), Arc::new(strings)]).unwrap();
        // Export consumes the batch; only PyArrow owns the buffers afterward.
        let table = record_batches_to_pyarrow(py, vec![batch])?;
        py.import("gc")?.call_method0("collect")?;
        for (column, buffer, address) in addresses {
            let chunk = table
                .bind(py)
                .call_method1("column", (column,))?
                .call_method1("chunk", (0,))?;
            assert_eq!(
                chunk
                    .call_method0("buffers")?
                    .get_item(buffer)?
                    .getattr("address")?
                    .extract::<usize>()?,
                address
            );
        }
        let expected = evaluate(
            py,
            c"pa.table({'number': pa.array([12, None, 34], type=pa.int64()),
                        'text': ['first', None, 'last']})",
        )?;
        assert!(table
            .bind(py)
            .call_method1("equals", (expected,))?
            .is_truthy()?);
        Ok(())
    })
    .unwrap();
}

#[test]
fn table_export_preserves_chunks_and_shares_each_buffer() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
        let arrays = [
            Arc::new(Int64Array::from(vec![Some(1), None])),
            Arc::new(Int64Array::from(vec![Some(3)])),
        ];
        let addresses: Vec<_> = arrays
            .iter()
            .map(|array| array.values().as_ptr() as usize)
            .collect();
        let batches = arrays
            .into_iter()
            .map(|array| RecordBatch::try_new(schema.clone(), vec![array]).unwrap())
            .collect();
        let table = record_batches_to_pyarrow(py, batches)?;
        let column = table.bind(py).call_method1("column", (0,))?;
        assert_eq!(column.getattr("num_chunks")?.extract::<usize>()?, 2);
        for (index, address) in addresses.iter().enumerate() {
            let chunk = column.call_method1("chunk", (index,))?;
            assert_eq!(
                chunk
                    .call_method0("buffers")?
                    .get_item(1)?
                    .getattr("address")?
                    .extract::<usize>()?,
                *address
            );
        }
        assert_eq!(
            column
                .call_method0("to_pylist")?
                .extract::<Vec<Option<i64>>>()?,
            vec![Some(1), None, Some(3)]
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn table_export_preserves_nested_values_and_nul_metadata() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let field = Arc::new(
            Field::new("value", DataType::Int64, true)
                .with_metadata(HashMap::from([("field\0key".into(), "a\0b".into())])),
        );
        let nested = StructArray::new(
            vec![field].into(),
            vec![Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]))],
            None,
        );
        let schema = Arc::new(Schema::new_with_metadata(
            vec![Field::new("nested", nested.data_type().clone(), true)],
            HashMap::from([("schema_key".into(), "c\0d".into())]),
        ));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(nested)]).unwrap();
        let table = record_batches_to_pyarrow(py, vec![batch])?;
        let expected_schema = evaluate(
            py,
            c"pa.schema([pa.field('nested', pa.struct([
                pa.field('value', pa.int64(), metadata={b'field\\x00key': b'a\\x00b'})
            ]))], metadata={b'schema_key': b'c\\x00d'})",
        )?;
        assert_schema_equal(&table.bind(py).getattr("schema")?, &expected_schema)?;
        let actual = table.bind(py).call_method0("to_pylist")?;
        let expected = evaluate(
            py,
            c"[{'nested': {'value': 1}}, {'nested': {'value': None}},
               {'nested': {'value': 3}}]",
        )?;
        assert!(actual.eq(expected)?);
        Ok(())
    })
    .unwrap();
}

#[test]
fn table_export_preserves_empty_schemas_and_zero_column_rows() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let empty = record_batches_to_pyarrow(py, vec![])?;
        assert_eq!(empty.bind(py).getattr("num_rows")?.extract::<usize>()?, 0);
        assert_eq!(
            empty.bind(py).getattr("num_columns")?.extract::<usize>()?,
            0
        );
        assert_schema_equal(
            &empty.bind(py).getattr("schema")?,
            &evaluate(py, c"pa.schema([])")?,
        )?;

        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
        let empty = record_batches_to_pyarrow(py, vec![RecordBatch::new_empty(schema)])?;
        assert_eq!(empty.bind(py).getattr("num_rows")?.extract::<usize>()?, 0);
        assert_schema_equal(
            &empty.bind(py).getattr("schema")?,
            &evaluate(py, c"pa.schema([('x', pa.int64())])")?,
        )?;

        for rows in [0, 4] {
            let schema =
                Schema::empty().with_metadata(HashMap::from([("empty".into(), "metadata".into())]));
            let options = RecordBatchOptions::new().with_row_count(Some(rows));
            let batch =
                RecordBatch::try_new_with_options(Arc::new(schema), vec![], &options).unwrap();
            let table = record_batches_to_pyarrow(py, vec![batch])?;
            assert_eq!(
                table.bind(py).getattr("num_rows")?.extract::<usize>()?,
                rows
            );
            assert_eq!(
                table.bind(py).getattr("num_columns")?.extract::<usize>()?,
                0
            );
            assert_schema_equal(
                &table.bind(py).getattr("schema")?,
                &evaluate(py, c"pa.schema([], metadata={b'empty': b'metadata'})")?,
            )?;
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn table_export_rejects_inconsistent_fields_and_metadata() {
    Python::initialize();
    Python::attach(|py| {
        let field = Field::new("x", DataType::Int64, true);
        let schema = Schema::new(vec![field.clone()]);
        for other in [
            Schema::new(vec![Field::new("other", DataType::Int64, true)]),
            schema
                .clone()
                .with_metadata(HashMap::from([("schema".into(), "other".into())])),
            Schema::new(vec![field
                .clone()
                .with_metadata(HashMap::from([("field".into(), "other".into())]))]),
        ] {
            let batches = vec![
                RecordBatch::new_empty(Arc::new(schema.clone())),
                RecordBatch::new_empty(Arc::new(other)),
            ];
            let error = record_batches_to_pyarrow(py, batches).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("inconsistent schemas"));
        }
    });
}

#[test]
fn table_export_falls_back_to_ipc_for_nul_names_and_timezones() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let cases: [(Field, &CStr); 3] = [
            (
                Field::new("bad\0name", DataType::Int64, true),
                c"pa.schema([('bad\\x00name', pa.int64())])",
            ),
            (
                Field::new(
                    "nested",
                    DataType::Struct(vec![Field::new("bad\0name", DataType::Int64, true)].into()),
                    true,
                ),
                c"pa.schema([('nested', pa.struct([('bad\\x00name', pa.int64())]))])",
            ),
            (
                Field::new(
                    "time",
                    DataType::Timestamp(TimeUnit::Microsecond, Some("UTC\0suffix".into())),
                    true,
                ),
                c"pa.schema([('time', pa.timestamp('us', tz='UTC\\x00suffix'))])",
            ),
        ];
        for (field, expected) in cases {
            let batch = RecordBatch::new_empty(Arc::new(Schema::new(vec![field])));
            let table = record_batches_to_pyarrow(py, vec![batch])?;
            assert_schema_equal(&table.bind(py).getattr("schema")?, &evaluate(py, expected)?)?;
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn nul_fallback_preserves_nonempty_batches() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "bad\0name",
            DataType::Int64,
            true,
        )]));
        let batches = [
            Int64Array::from(vec![Some(1), None]),
            Int64Array::from(vec![Some(3), Some(4)]),
        ]
        .into_iter()
        .map(|array| RecordBatch::try_new(schema.clone(), vec![Arc::new(array)]).unwrap())
        .collect();
        let table = record_batches_to_pyarrow(py, batches)?;
        let column = table.bind(py).call_method1("column", (0,))?;
        assert_eq!(column.getattr("num_chunks")?.extract::<usize>()?, 2);
        assert_eq!(
            column
                .call_method0("to_pylist")?
                .extract::<Vec<Option<i64>>>()?,
            vec![Some(1), None, Some(3), Some(4)]
        );
        assert_schema_equal(
            &table.bind(py).getattr("schema")?,
            &evaluate(py, c"pa.schema([('bad\\x00name', pa.int64())])")?,
        )?;
        Ok(())
    })
    .unwrap();
}

#[test]
fn nul_fallback_rejects_mismatched_schemas_first() {
    Python::initialize();
    Python::attach(|py| {
        let field = Field::new("bad\0name", DataType::Int64, true);
        let schema = Schema::new(vec![field]);
        let other = schema
            .clone()
            .with_metadata(HashMap::from([("changed".into(), "metadata".into())]));
        let batches = [schema, other]
            .into_iter()
            .map(|schema| {
                RecordBatch::try_new(Arc::new(schema), vec![Arc::new(Int64Array::from(vec![1]))])
                    .unwrap()
            })
            .collect();
        let error = record_batches_to_pyarrow(py, batches).unwrap_err();
        assert!(error.is_instance_of::<PyValueError>(py));
        assert!(error.to_string().contains("inconsistent schemas"));
    });
}
