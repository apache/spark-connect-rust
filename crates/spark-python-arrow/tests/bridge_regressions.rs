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

use std::ffi::CStr;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, DictionaryArray, Int64Array, Int8Array, ListArray, MapArray, StringArray,
    StructArray,
};
use arrow::buffer::{NullBuffer, OffsetBuffer};
use arrow::datatypes::{DataType, Field, Int8Type, Schema};
use arrow::record_batch::RecordBatch;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_python_arrow::bridge::record_batches_to_pyarrow;

fn assert_export_equal(py: Python<'_>, batch: RecordBatch, expected_code: &CStr) -> PyResult<()> {
    let locals = PyDict::new(py);
    locals.set_item("pa", py.import("pyarrow")?)?;
    py.run(expected_code, Some(&locals), Some(&locals))?;
    let expected = locals.get_item("expected")?.expect("expected table");
    let actual = record_batches_to_pyarrow(py, vec![batch])?;
    assert!(actual
        .bind(py)
        .call_method1("equals", (&expected,))?
        .is_truthy()?);
    assert!(actual
        .bind(py)
        .getattr("schema")?
        .call_method1("equals", (expected.getattr("schema")?, true))?
        .is_truthy()?);
    Ok(())
}

#[test]
fn table_export_preserves_sorted_map_flags_in_nested_fields() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let entries = StructArray::new(
            vec![
                Field::new("key", DataType::Utf8, false),
                Field::new("value", DataType::Int64, true),
            ]
            .into(),
            vec![
                Arc::new(StringArray::from(vec!["a", "b"])),
                Arc::new(Int64Array::from(vec![1, 2])),
            ],
            None,
        );
        let maps: ArrayRef = Arc::new(MapArray::new(
            Arc::new(Field::new("entries", entries.data_type().clone(), false)),
            OffsetBuffer::new(vec![0, 2, 2, 2].into()),
            entries,
            Some(NullBuffer::from(vec![true, false, true])),
            true,
        ));
        let lists: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new("item", maps.data_type().clone(), true)),
            OffsetBuffer::new(vec![0, 1, 2, 3].into()),
            maps.clone(),
            None,
        ));
        let nested: ArrayRef = Arc::new(StructArray::new(
            vec![Field::new("maps", lists.data_type().clone(), true)].into(),
            vec![lists],
            None,
        ));
        let schema = Arc::new(Schema::new(vec![
            Field::new("maps", maps.data_type().clone(), true),
            Field::new("nested", nested.data_type().clone(), true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![maps, nested]).unwrap();
        assert_export_equal(
            py,
            batch,
            c"map_type = pa.map_(pa.string(), pa.int64(), keys_sorted=True)
maps = pa.array([[('a', 1), ('b', 2)], None, []], type=map_type)
nested_type = pa.struct([pa.field('maps', pa.list_(map_type))])
nested = pa.array([
    {'maps': [[('a', 1), ('b', 2)]]}, {'maps': [None]}, {'maps': [[]]}
], type=nested_type)
expected = pa.Table.from_arrays([maps, nested], names=['maps', 'nested'])",
        )
    })
    .unwrap();
}

#[test]
fn table_export_preserves_ordered_dictionaries_in_nested_fields() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let dictionary: ArrayRef = Arc::new(DictionaryArray::<Int8Type>::new(
            Int8Array::from(vec![Some(1), None, Some(0)]),
            Arc::new(StringArray::from(vec!["a", "b"])),
        ));
        let field = Field::new("dictionary", dictionary.data_type().clone(), true)
            .with_dict_is_ordered(true);
        let nested: ArrayRef = Arc::new(StructArray::new(
            vec![field.clone()].into(),
            vec![dictionary.clone()],
            None,
        ));
        let lists: ArrayRef = Arc::new(ListArray::new(
            Arc::new(field.clone().with_name("item")),
            OffsetBuffer::new(vec![0, 1, 2, 3].into()),
            dictionary.clone(),
            None,
        ));
        let schema = Arc::new(Schema::new(vec![
            field,
            Field::new("nested", nested.data_type().clone(), true),
            Field::new("lists", lists.data_type().clone(), true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![dictionary, nested, lists]).unwrap();
        assert_export_equal(
            py,
            batch,
            c"dictionary = pa.DictionaryArray.from_arrays(
    pa.array([1, None, 0], type=pa.int8()), pa.array(['a', 'b']), ordered=True
)
nested = pa.StructArray.from_arrays([dictionary], names=['dictionary'])
lists = pa.ListArray.from_arrays(pa.array([0, 1, 2, 3], type=pa.int32()), dictionary)
expected = pa.Table.from_arrays([dictionary, nested, lists],
                                names=['dictionary', 'nested', 'lists'])",
        )
    })
    .unwrap();
}

#[test]
fn table_export_rejects_inconsistent_nested_dictionary_order() {
    Python::initialize();
    Python::attach(|py| {
        let fields = |ordered| {
            let dictionary = Field::new(
                "dictionary",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                true,
            )
            .with_dict_is_ordered(ordered);
            [
                dictionary.clone(),
                Field::new(
                    "nested",
                    DataType::Struct(vec![dictionary.clone()].into()),
                    true,
                ),
                Field::new("lists", DataType::List(Arc::new(dictionary)), true),
            ]
        };
        for (ordered, unordered) in fields(true).into_iter().zip(fields(false)) {
            let batches = vec![
                RecordBatch::new_empty(Arc::new(Schema::new(vec![ordered]))),
                RecordBatch::new_empty(Arc::new(Schema::new(vec![unordered]))),
            ];
            let error = record_batches_to_pyarrow(py, batches).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("inconsistent schemas"));
        }
    });
}

#[test]
fn sliced_validity_shares_values_without_requiring_bitmap_identity() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let original = Int64Array::from(vec![Some(1), Some(2), None, Some(4), None]);
        let slice = original.slice(1, 3);
        assert_eq!(slice.nulls().unwrap().offset(), 1);
        assert_eq!(slice.to_data().offset(), 0);
        let address = slice.values().as_ptr() as usize;
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(slice)]).unwrap();
        let table = record_batches_to_pyarrow(py, vec![batch])?;
        drop(original);
        py.import("gc")?.call_method0("collect")?;
        let array = table
            .bind(py)
            .call_method1("column", (0,))?
            .call_method1("chunk", (0,))?;
        assert_eq!(
            array
                .call_method0("buffers")?
                .get_item(1)?
                .getattr("address")?
                .extract::<usize>()?,
            address
        );
        // Arrow may realign validity to the value buffer's offset. The bitmap's
        // identity is not part of the value-buffer sharing promise.
        assert_eq!(array.getattr("offset")?.extract::<usize>()?, 0);
        assert_eq!(
            array
                .call_method0("to_pylist")?
                .extract::<Vec<Option<i64>>>()?,
            vec![Some(2), None, Some(4)]
        );
        Ok(())
    })
    .unwrap();
}
