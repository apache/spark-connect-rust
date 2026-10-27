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

use arrow::array::{Array, Int64Array};
use arrow::datatypes::DataType;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_python_arrow::bridge::{
    export_array, export_data_type, export_schema, import_array, import_batch, import_data_type,
    import_schema, record_batches_to_pyarrow,
};

fn setup<'py>(py: Python<'py>, code: &CStr) -> PyResult<Bound<'py, PyDict>> {
    let locals = PyDict::new(py);
    locals.set_item("pa", py.import("pyarrow")?)?;
    py.run(code, Some(&locals), Some(&locals))?;
    Ok(locals)
}

fn value<'py>(locals: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    Ok(locals.get_item(name)?.expect("test variable must exist"))
}

fn assert_equal(actual: &Bound<'_, PyAny>, expected: &Bound<'_, PyAny>) -> PyResult<()> {
    assert!(actual.call_method1("equals", (expected,))?.is_truthy()?);
    Ok(())
}

fn assert_schema_equal(actual: &Bound<'_, PyAny>, expected: &Bound<'_, PyAny>) -> PyResult<()> {
    assert!(actual
        .call_method1("equals", (expected, true))?
        .is_truthy()?);
    Ok(())
}

#[test]
fn sorted_maps_keep_type_flags_through_schema_array_and_batch_paths() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = setup(
            py,
            c"map_type = pa.map_(pa.string(), pa.int64(), keys_sorted=True)
maps = pa.array([[('a', 1), ('b', 2)], None, []], type=map_type)
nested_type = pa.struct([
    pa.field('maps', pa.list_(map_type), metadata={b'nested': b'yes'})
])
nested = pa.array([
    {'maps': [[('a', 1)]]}, None, {'maps': []}
], type=nested_type)
schema = pa.schema([
    pa.field('maps', map_type, metadata={b'field': b'yes'}),
    pa.field('nested', nested_type)
], metadata={b'schema': b'yes'})
batch = pa.RecordBatch.from_arrays([maps, nested], schema=schema)
expected = pa.Table.from_batches([batch])",
        )?;

        let schema = value(&locals, "schema")?;
        let imported_schema = import_schema(&schema)?;
        let exported_schema = export_schema(py, &imported_schema)?;
        assert_schema_equal(exported_schema.bind(py), &schema)?;
        for name in ["map_type", "nested_type"] {
            let data_type = value(&locals, name)?;
            let exported = export_data_type(py, &import_data_type(&data_type)?)?;
            assert_equal(exported.bind(py), &data_type)?;
        }
        for name in ["maps", "nested"] {
            let array = value(&locals, name)?;
            let exported = export_array(py, import_array(&array)?)?;
            assert_equal(exported.bind(py), &array)?;
            assert_equal(&exported.bind(py).getattr("type")?, &array.getattr("type")?)?;
        }
        let batch = value(&locals, "batch")?;
        let imported = import_batch(&batch)?;
        let exported = record_batches_to_pyarrow(py, vec![imported])?;
        assert_equal(exported.bind(py), &value(&locals, "expected")?)?;
        assert_schema_equal(&exported.bind(py).getattr("schema")?, &schema)
    })
    .unwrap();
}

#[test]
fn ordered_dictionaries_keep_field_flags_in_nested_types_and_batches() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = setup(
            py,
            c"dictionary = pa.DictionaryArray.from_arrays(
    pa.array([1, None, 0], type=pa.int8()), pa.array(['a', 'b']), ordered=True
)
nested = pa.StructArray.from_arrays([dictionary], names=['dictionary'])
lists = pa.ListArray.from_arrays(pa.array([0, 1, 2, 3], type=pa.int32()), dictionary)
schema = pa.schema([
    pa.field('dictionary', dictionary.type),
    pa.field('nested', nested.type),
    pa.field('lists', lists.type)
], metadata={b'schema': b'ordered'})
batch = pa.RecordBatch.from_arrays([dictionary, nested, lists], schema=schema)
expected = pa.Table.from_batches([batch])",
        )?;
        let schema = value(&locals, "schema")?;
        let imported_schema = import_schema(&schema)?;
        // Field equality deliberately ignores dictionary-order flags in Arrow Rust.
        // Check them explicitly, then also compare the exported Python schema.
        assert_eq!(imported_schema.field(0).dict_is_ordered(), Some(true));
        let DataType::Struct(fields) = imported_schema.field(1).data_type() else {
            panic!("expected a struct")
        };
        assert_eq!(fields[0].dict_is_ordered(), Some(true));
        let DataType::List(field) = imported_schema.field(2).data_type() else {
            panic!("expected a list")
        };
        assert_eq!(field.dict_is_ordered(), Some(true));
        let exported = export_schema(py, &imported_schema)?;
        assert_schema_equal(exported.bind(py), &schema)?;

        for name in ["nested", "lists"] {
            let array = value(&locals, name)?;
            let data_type = array.getattr("type")?;
            let exported_type = export_data_type(py, &import_data_type(&data_type)?)?;
            assert_equal(exported_type.bind(py), &data_type)?;
            let exported_array = export_array(py, import_array(&array)?)?;
            assert_equal(exported_array.bind(py), &array)?;
            assert_equal(&exported_array.bind(py).getattr("type")?, &data_type)?;
        }

        let imported = import_batch(&value(&locals, "batch")?)?;
        assert_eq!(imported.schema().field(0).dict_is_ordered(), Some(true));
        let DataType::Struct(fields) = imported.column(1).data_type() else {
            panic!("expected a struct column")
        };
        assert_eq!(fields[0].dict_is_ordered(), Some(true));
        let exported = record_batches_to_pyarrow(py, vec![imported])?;
        assert_equal(exported.bind(py), &value(&locals, "expected")?)?;
        assert_schema_equal(&exported.bind(py).getattr("schema")?, &schema)
    })
    .unwrap();
}

#[test]
fn bare_ordered_dictionaries_require_a_field_to_preserve_ordering() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = setup(
            py,
            c"ordered = pa.DictionaryArray.from_arrays(
    pa.array([0, 1], type=pa.int8()), pa.array(['a', 'b']), ordered=True
)
unordered = pa.DictionaryArray.from_arrays(
    pa.array([0, 1], type=pa.int8()), pa.array(['a', 'b'])
)",
        )?;
        let ordered = value(&locals, "ordered")?;
        for error in [
            import_array(&ordered).unwrap_err(),
            import_data_type(&ordered.getattr("type")?).unwrap_err(),
        ] {
            assert!(error.is_instance_of::<PyTypeError>(py));
            assert!(error.to_string().contains("preserve ordering"));
        }
        let unordered = value(&locals, "unordered")?;
        let exported = export_array(py, import_array(&unordered)?)?;
        assert_equal(exported.bind(py), &unordered)
    })
    .unwrap();
}

#[test]
fn batches_with_inconsistent_nested_dictionary_order_are_rejected() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = setup(
            py,
            c"ordered = pa.DictionaryArray.from_arrays(
    pa.array([1, None, 0], type=pa.int8()), pa.array(['a', 'b']), ordered=True
)
unordered = pa.DictionaryArray.from_arrays(ordered.indices, ordered.dictionary)
ordered_struct = pa.StructArray.from_arrays([ordered], names=['dictionary'])
unordered_struct = pa.StructArray.from_arrays([unordered], names=['dictionary'])
ordered_list = pa.ListArray.from_arrays(pa.array([0, 1, 2, 3], type=pa.int32()), ordered)
unordered_list = pa.ListArray.from_arrays(pa.array([0, 1, 2, 3], type=pa.int32()), unordered)
pairs = [
    (ordered, unordered),
    (ordered_struct, unordered_struct),
    (ordered_list, unordered_list)
]
batches = [
    (pa.record_batch([left], names=['x']), pa.record_batch([right], names=['x']))
    for left, right in pairs
]",
        )?;
        let batches = value(&locals, "batches")?;
        for index in 0..batches.len()? {
            let pair = batches.get_item(index)?;
            let left = import_batch(&pair.get_item(0)?)?;
            let right = import_batch(&pair.get_item(1)?)?;
            let error = record_batches_to_pyarrow(py, vec![left, right]).unwrap_err();
            assert!(error.is_instance_of::<PyValueError>(py));
            assert!(error.to_string().contains("inconsistent schemas"));
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn dictionary_value_extensions_are_rejected_instead_of_losing_identity() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = setup(
            py,
            c"class TestExtension(pa.ExtensionType):
    def __init__(self):
        super().__init__(pa.int64(), 'spark_python_arrow.dictionary_extension')

    def __arrow_ext_serialize__(self):
        return b'extension'

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()

extension = pa.ExtensionArray.from_storage(TestExtension(), pa.array([10, 20]))
dictionary = pa.DictionaryArray.from_arrays(pa.array([0, 1], type=pa.int8()), extension)
nested = pa.StructArray.from_arrays([dictionary], names=['dictionary'])
schema = pa.schema([pa.field('dictionary', dictionary.type)])
batch = pa.RecordBatch.from_arrays([dictionary], schema=schema)
nested_schema = pa.schema([pa.field('nested', nested.type)])
nested_batch = pa.RecordBatch.from_arrays([nested], schema=nested_schema)",
        )?;
        for name in ["dictionary", "nested"] {
            let array = value(&locals, name)?;
            for error in [
                import_array(&array).unwrap_err(),
                import_data_type(&array.getattr("type")?).unwrap_err(),
            ] {
                assert!(error.is_instance_of::<PyTypeError>(py));
                assert!(error.to_string().contains("dictionary value extension"));
            }
        }
        for name in ["schema", "nested_schema"] {
            let error = import_schema(&value(&locals, name)?).unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));
            assert!(error.to_string().contains("dictionary value extension"));
        }
        for name in ["batch", "nested_batch"] {
            let error = import_batch(&value(&locals, name)?).unwrap_err();
            assert!(error.is_instance_of::<PyTypeError>(py));
            assert!(error.to_string().contains("dictionary value extension"));
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn nullable_non_byte_aligned_slice_shares_values_without_requiring_bitmap_identity() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let original = Int64Array::from(vec![Some(1), Some(2), None, Some(4), None]);
        let slice = original.slice(1, 3);
        assert_eq!(slice.nulls().unwrap().offset(), 1);
        assert_eq!(slice.to_data().offset(), 0);
        let address = slice.values().as_ptr() as usize;
        let exported = export_array(py, Arc::new(slice))?;
        drop(original);
        py.import("gc")?.call_method0("collect")?;
        let array = exported.bind(py);
        assert_eq!(
            array
                .call_method0("buffers")?
                .get_item(1)?
                .getattr("address")?
                .extract::<usize>()?,
            address
        );
        // Arrow may realign the validity bitmap to the value buffer's offset.
        // Its identity is not part of the bridge's value-buffer sharing promise.
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
