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

//! Standalone conversion checks against the installed PyArrow. No Spark server is needed.
//! These exercise the shared interfaces, not the full client conformance suite.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;

use arrow::array::{Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use pyo3::exceptions::{
    PyOverflowError, PyTypeError, PyUnicodeDecodeError, PyUnicodeEncodeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use spark_python_arrow::bridge::{
    export_array, export_data_type, export_schema, import_array, import_batch, import_data_type,
    import_schema, record_batches_to_pyarrow,
};
use spark_python_arrow::scalar::{
    extract_binary, extract_bool, extract_f64, extract_i64, extract_string, utf8_into_python,
    Scalar,
};

fn context(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let locals = PyDict::new(py);
    locals.set_item("pa", py.import("pyarrow")?)?;
    locals.set_item("datetime", py.import("datetime")?)?;
    Ok(locals)
}

fn evaluate<'py>(
    py: Python<'py>,
    locals: &Bound<'py, PyDict>,
    expression: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let code = CString::new(expression).expect("test expressions contain no NUL");
    py.eval(&code, Some(locals), Some(locals))
}

fn extension_context<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyDict>> {
    let locals = context(py)?;
    locals.set_item("extension_name", name)?;
    py.run(
        c"class TestExtensionType(pa.ExtensionType):
    def __init__(self, storage_type=pa.string()):
        super().__init__(storage_type, extension_name)

    def __arrow_ext_serialize__(self):
        return b'extension metadata'

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        assert serialized == b'extension metadata'
        return cls(storage_type)

extension_type = TestExtensionType()
extension_array = pa.ExtensionArray.from_storage(
    extension_type, pa.array(['first', None, 'last'])
)",
        Some(&locals),
        Some(&locals),
    )?;
    Ok(locals)
}

fn assert_array_equal(actual: &Bound<'_, PyAny>, expected: &Bound<'_, PyAny>) -> PyResult<()> {
    assert!(
        actual
            .call_method1("equals", (expected,))?
            .extract::<bool>()?,
        "actual {actual}; expected {expected}"
    );
    Ok(())
}

#[test]
fn exported_array_keeps_rust_buffers_alive_without_copying() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let rust_array = Arc::new(Int64Array::from(vec![Some(12), None, Some(34)]));
        let address = rust_array.values().as_ptr() as usize;
        let array = export_array(py, rust_array.clone())?;
        drop(rust_array);
        py.import("gc")?.call_method0("collect")?;

        let array = array.bind(py);
        let actual_address: usize = array
            .call_method0("buffers")?
            .get_item(1)?
            .getattr("address")?
            .extract()?;
        assert_eq!(actual_address, address);
        assert_eq!(
            array
                .call_method0("to_pylist")?
                .extract::<Vec<Option<i64>>>()?,
            vec![Some(12), None, Some(34)]
        );
        Ok(())
    })
    .unwrap();
}

#[test]
fn imported_slice_keeps_python_buffers_alive_without_copying() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        let original = evaluate(py, &locals, "pa.array([10, 20, None, 40], type=pa.int64())")?;
        let address: usize = original
            .call_method0("buffers")?
            .get_item(1)?
            .getattr("address")?
            .extract()?;
        let slice = original.call_method1("slice", (1, 2))?;
        let imported = import_array(&slice)?;
        drop(slice);
        drop(original);
        py.import("gc")?.call_method0("collect")?;

        let values = imported.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(
            values.values().as_ptr() as usize,
            address + std::mem::size_of::<i64>()
        );
        assert_eq!(values.len(), 2);
        assert_eq!(values.value(0), 20);
        assert!(values.is_null(1));

        let exported = export_array(py, imported)?;
        let expected = evaluate(py, &locals, "pa.array([20, None], type=pa.int64())")?;
        assert_array_equal(exported.bind(py), &expected)
    })
    .unwrap();
}

#[test]
fn schema_and_nested_type_round_trip_preserves_metadata() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let field = Field::new("value", DataType::Int64, true)
            .with_metadata(HashMap::from([("field_key".to_owned(), "a\0b".to_owned())]));
        let schema = Schema::new_with_metadata(
            vec![field],
            HashMap::from([("schema_key".to_owned(), "c\0d".to_owned())]),
        );
        let exported = export_schema(py, &schema)?;
        assert_eq!(import_schema(exported.bind(py))?, schema);

        for data_type in [
            DataType::Int64,
            DataType::Utf8,
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            DataType::Struct(schema.fields().clone()),
        ] {
            let exported = export_data_type(py, &data_type)?;
            assert_eq!(import_data_type(exported.bind(py))?, data_type);
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn record_batch_import_preserves_schema_and_values() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        let batch = evaluate(
            py,
            &locals,
            "pa.RecordBatch.from_arrays([pa.array([1, None, 3], type=pa.int64())], \
             schema=pa.schema([pa.field('x', pa.int64(), metadata={b'f': b'v'})], \
             metadata={b's': b'v'}))",
        )?;
        let schema = import_schema(&batch.getattr("schema")?)?;
        let imported = import_batch(&batch)?;
        assert_eq!(imported.schema().as_ref(), &schema);
        assert_eq!(imported.num_rows(), 3);
        let exported = record_batches_to_pyarrow(py, vec![imported])?;
        let expected = py
            .import("pyarrow")?
            .getattr("Table")?
            .call_method1("from_batches", (PyList::new(py, [batch])?,))?;
        assert_array_equal(exported.bind(py), &expected)
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
            .map(|a| a.values().as_ptr() as usize)
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
fn empty_batches_and_zero_column_rows_round_trip() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let empty = record_batches_to_pyarrow(py, vec![])?;
        assert_eq!(empty.bind(py).getattr("num_rows")?.extract::<usize>()?, 0);
        assert_eq!(
            empty.bind(py).getattr("num_columns")?.extract::<usize>()?,
            0
        );

        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
        let empty = record_batches_to_pyarrow(py, vec![RecordBatch::new_empty(schema.clone())])?;
        assert_eq!(empty.bind(py).getattr("num_rows")?.extract::<usize>()?, 0);
        assert_eq!(import_schema(&empty.bind(py).getattr("schema")?)?, *schema);

        let options = RecordBatchOptions::new().with_row_count(Some(4));
        let batch =
            RecordBatch::try_new_with_options(Arc::new(Schema::empty()), vec![], &options).unwrap();
        let table = record_batches_to_pyarrow(py, vec![batch])?;
        assert_eq!(table.bind(py).getattr("num_rows")?.extract::<usize>()?, 4);
        assert_eq!(
            table.bind(py).getattr("num_columns")?.extract::<usize>()?,
            0
        );

        let locals = context(py)?;
        let batch = evaluate(
            py,
            &locals,
            "pa.record_batch([pa.array([1, 2, 3])], names=['x']).select([])",
        )?;
        let imported = import_batch(&batch)?;
        assert_eq!(imported.num_rows(), 3);
        assert_eq!(imported.num_columns(), 0);
        Ok(())
    })
    .unwrap();
}

#[test]
fn bridge_rejects_wrong_objects_without_calling_their_exporter() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        py.run(
            c"class Fake:
    def _export_to_c(self, *args):
        raise AssertionError('called')",
            Some(&locals),
            Some(&locals),
        )?;
        let fake = evaluate(py, &locals, "Fake()")?;
        for error in [
            import_schema(&fake).unwrap_err(),
            import_data_type(&fake).unwrap_err(),
            import_array(&fake).unwrap_err(),
            import_batch(&fake).unwrap_err(),
        ] {
            assert!(error.is_instance_of::<PyTypeError>(py), "{error}");
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn bare_extension_imports_fail_instead_of_losing_type_identity() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = extension_context(py, "spark_python_arrow.tests.bare_extension")?;
        let data_type = evaluate(py, &locals, "extension_type")?;
        let array = evaluate(py, &locals, "extension_array")?;
        for error in [
            import_data_type(&data_type).unwrap_err(),
            import_array(&array).unwrap_err(),
        ] {
            assert!(error.is_instance_of::<PyTypeError>(py), "{error}");
            assert!(error.to_string().contains("extension types"), "{error}");
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn field_metadata_preserves_extensions_in_schemas_and_record_batches() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let name = "spark_python_arrow.tests.field_extension";
        let locals = extension_context(py, name)?;
        let data_type = evaluate(py, &locals, "extension_type")?;
        let pa = py.import("pyarrow")?;
        pa.call_method1("register_extension_type", (&data_type,))?;
        let result = (|| -> PyResult<()> {
            let schema = evaluate(
                py,
                &locals,
                "pa.schema([pa.field('value', extension_type, metadata={b'f': b'v'})], \
                 metadata={b's': b'v'})",
            )?;
            let imported_schema = import_schema(&schema)?;
            assert_eq!(
                imported_schema
                    .field(0)
                    .metadata()
                    .get("ARROW:extension:name")
                    .map(String::as_str),
                Some(name)
            );
            let exported_schema = export_schema(py, &imported_schema)?;
            assert!(exported_schema
                .bind(py)
                .call_method1("equals", (&schema, true))?
                .is_truthy()?);

            // Nested extension fields have a metadata owner even when the outer
            // value is a bare DataType, so only root extension types are rejected.
            let nested = evaluate(
                py,
                &locals,
                "pa.struct([pa.field('value', extension_type)])",
            )?;
            let exported_type = export_data_type(py, &import_data_type(&nested)?)?;
            assert!(exported_type
                .bind(py)
                .call_method1("equals", (&nested,))?
                .is_truthy()?);

            let kwargs = PyDict::new(py);
            kwargs.set_item("schema", &schema)?;
            let batch = pa.getattr("RecordBatch")?.call_method(
                "from_arrays",
                (PyList::new(
                    py,
                    [evaluate(py, &locals, "extension_array")?],
                )?,),
                Some(&kwargs),
            )?;
            let imported = import_batch(&batch)?;
            assert_eq!(imported.schema().as_ref(), &imported_schema);
            let exported = record_batches_to_pyarrow(py, vec![imported])?;
            let expected = pa
                .getattr("Table")?
                .call_method1("from_batches", (PyList::new(py, [batch])?,))?;
            assert_array_equal(exported.bind(py), &expected)?;
            assert!(exported
                .bind(py)
                .getattr("schema")?
                .call_method1("equals", (&schema, true))?
                .is_truthy()?);
            Ok(())
        })();
        pa.call_method1("unregister_extension_type", (name,))?;
        result
    })
    .unwrap();
}

#[test]
fn table_export_rejects_inconsistent_schemas() {
    Python::initialize();
    Python::attach(|py| {
        let batches = ["a", "b"]
            .into_iter()
            .map(|name| {
                RecordBatch::new_empty(Arc::new(Schema::new(vec![Field::new(
                    name,
                    DataType::Int64,
                    true,
                )])))
            })
            .collect();
        let error = record_batches_to_pyarrow(py, batches).unwrap_err();
        assert!(error.is_instance_of::<PyValueError>(py));
    });
}

#[test]
fn scalar_materialization_preserves_values_and_python_types() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        let cases = [
            (Scalar::Null, "None"),
            (Scalar::Bool(true), "True"),
            (Scalar::Int(i64::MIN), "-(2**63)"),
            (Scalar::Int(i64::MAX), "2**63 - 1"),
            (Scalar::Float(1.25), "1.25"),
            (Scalar::String("a\0b"), "'a\\x00b'"),
            (Scalar::String("\u{03bb}"), "'\\u03bb'"),
            (Scalar::Binary(b"a\0b"), "b'a\\x00b'"),
            (Scalar::Date(0), "datetime.date(1970, 1, 1)"),
            (Scalar::Date(-719_162), "datetime.date.min"),
            (Scalar::Date(2_932_896), "datetime.date.max"),
        ];
        for (value, expression) in cases {
            let actual = value.into_python(py)?;
            let expected = evaluate(py, &locals, expression)?;
            assert!(actual.eq(&expected)?, "{value:?}");
            assert!(actual.get_type().is(expected.get_type()), "{value:?}");
        }
        let bytes = Scalar::Binary(b"value").into_python(py)?;
        assert!(bytes.is_exact_instance_of::<PyBytes>());
        for days in [-719_163, 2_932_897, i32::MIN, i32::MAX] {
            let error = Scalar::Date(days).into_python(py).unwrap_err();
            let expected = evaluate(
                py,
                &locals,
                &format!("datetime.date.fromordinal({})", i64::from(days) + 719_163),
            )
            .unwrap_err();
            assert!(error.get_type(py).is(expected.get_type(py)));
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn client_scalar_extraction_preserves_coercion_and_binary_copy_semantics() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        py.run(
            c"class IntSubclass(int): pass
class FloatSubclass(float): pass
class StrSubclass(str): pass
class BytesSubclass(bytes): pass
class ByteArraySubclass(bytearray): pass
class IndexValue:
    def __index__(self):
        return 42
class FloatValue:
    def __float__(self):
        return 1.5
class TruthValue:
    def __bool__(self):
        raise AssertionError('truthiness must not be invoked')
class StringValue:
    def __str__(self):
        raise AssertionError('stringification must not be invoked')
",
            Some(&locals),
            Some(&locals),
        )?;
        let truth = evaluate(py, &locals, "True")?;
        assert!(extract_bool(&truth)?);
        assert!(!extract_bool(&evaluate(py, &locals, "False")?)?);
        // The client tries bool before int because integer extraction also accepts bool.
        assert_eq!(extract_i64(&truth)?, 1);
        for expression in ["1", "None", "TruthValue()"] {
            let error = extract_bool(&evaluate(py, &locals, expression)?).unwrap_err();
            assert!(
                error.is_instance_of::<PyTypeError>(py),
                "{expression}: {error}"
            );
        }
        for (expression, expected) in [
            ("7", 7),
            ("IntSubclass(7)", 7),
            ("IndexValue()", 42),
            ("-(2**63)", i64::MIN),
            ("2**63 - 1", i64::MAX),
        ] {
            assert_eq!(extract_i64(&evaluate(py, &locals, expression)?)?, expected);
        }
        for expression in ["1.5", "'7'"] {
            let error = extract_i64(&evaluate(py, &locals, expression)?).unwrap_err();
            assert!(
                error.is_instance_of::<PyTypeError>(py),
                "{expression}: {error}"
            );
        }
        for expression in ["2**100", "-(2**100)"] {
            let error = extract_i64(&evaluate(py, &locals, expression)?).unwrap_err();
            assert!(
                error.is_instance_of::<PyOverflowError>(py),
                "{expression}: {error}"
            );
        }
        for (expression, expected) in [
            ("1.25", 1.25),
            ("FloatSubclass(1.25)", 1.25),
            ("7", 7.0),
            ("FloatValue()", 1.5),
            ("IndexValue()", 42.0),
        ] {
            assert_eq!(extract_f64(&evaluate(py, &locals, expression)?)?, expected);
        }
        let error = extract_f64(&evaluate(py, &locals, "'1.25'")?).unwrap_err();
        assert!(error.is_instance_of::<PyTypeError>(py));
        for expression in ["'text'", "StrSubclass('text')"] {
            assert_eq!(extract_string(&evaluate(py, &locals, expression)?)?, "text");
        }
        let text = evaluate(py, &locals, "'a\\x00b'")?;
        assert_eq!(extract_string(&text)?, "a\0b");
        for expression in ["7", "StringValue()", "b'text'"] {
            let error = extract_string(&evaluate(py, &locals, expression)?).unwrap_err();
            assert!(
                error.is_instance_of::<PyTypeError>(py),
                "{expression}: {error}"
            );
        }
        let error = extract_string(&evaluate(py, &locals, "'\\ud800'")?).unwrap_err();
        assert!(error.is_instance_of::<PyUnicodeEncodeError>(py));
        for expression in [
            "b'ab'",
            "BytesSubclass(b'ab')",
            "bytearray(b'ab')",
            "ByteArraySubclass(b'ab')",
        ] {
            assert_eq!(
                extract_binary(&evaluate(py, &locals, expression)?),
                Some(b"ab".to_vec())
            );
        }
        let mutable = evaluate(py, &locals, "bytearray(b'ab')")?;
        let snapshot = extract_binary(&mutable).expect("bytearray is supported");
        mutable.set_item(0, 99)?;
        assert_eq!(snapshot.as_slice(), b"ab");
        assert_eq!(extract_binary(&mutable), Some(b"cb".to_vec()));
        for expression in ["[97, 98]", "(97, 98)", "'ab'", "memoryview(b'ab')"] {
            assert_eq!(extract_binary(&evaluate(py, &locals, expression)?), None);
        }
        assert!(!PyErr::occurred(py));
        Ok(())
    })
    .unwrap();
}

#[test]
fn utf8_scalar_decoding_preserves_embedded_nul_and_rejects_invalid_bytes() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        for value in ["", "a\0b", "\u{03bb}"] {
            assert_eq!(
                utf8_into_python(py, value.as_bytes())?.extract::<String>()?,
                value
            );
        }
        for bytes in [&[0xff][..], &[0xc3][..]] {
            let error = utf8_into_python(py, bytes).unwrap_err();
            assert!(error.is_instance_of::<PyUnicodeDecodeError>(py));
        }
        Ok(())
    })
    .unwrap();
}
