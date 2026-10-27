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
//! These exercise the shared interfaces, not the full worker or client conformance suites.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;

use arrow::array::{Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use pyo3::exceptions::{
    PyImportError, PyIndexError, PyTypeError, PyUnicodeDecodeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyType};
use spark_python_arrow::bridge::{
    export_array, export_data_type, export_schema, import_array, import_batch, import_data_type,
    import_schema, record_batches_to_pyarrow,
};
use spark_python_arrow::scalar::{
    exact_bool, exact_bytes, exact_f64, exact_i64, exact_str, extract_binary, utf8_into_python,
    Scalar,
};
use spark_python_arrow::udf::{
    input_spec, result_spec, Finished, InputColumns, ResultColumn, ResultSpec,
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

fn variant_context(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let locals = context(py)?;
    py.run(
        c"class VariantType: pass

class VariantVal:
    def __init__(self, value, metadata):
        self.value = value
        self.metadata = metadata

class SqlTypes: pass
SqlTypes.VariantType = VariantType
spark_type = VariantType()
converter = lambda value: {'value': value.value, 'metadata': value.metadata}
arrow_type = pa.struct([
    pa.field('value', pa.binary(), nullable=False),
    pa.field('metadata', pa.binary(), nullable=False, metadata={b'variant': b'true'}),
])",
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

fn primitive_spec<'py>(
    py: Python<'py>,
    arrow_type: &Bound<'py, PyAny>,
    safe: bool,
) -> PyResult<ResultSpec> {
    let none = py.None().into_bound(py);
    Ok(
        result_spec(py, arrow_type, &none, &none, safe, &none, || Ok(None))?
            .expect("the primitive type must be supported"),
    )
}

fn build_results<'py>(
    py: Python<'py>,
    arrow_type: &Bound<'py, PyAny>,
    values: &Bound<'py, PyList>,
    safe: bool,
) -> PyResult<Finished<'py>> {
    let spec = primitive_spec(py, arrow_type, safe)?;
    let mut column = ResultColumn::new(&spec, values.len());
    column.extend(py, values);
    column.finish(py)
}

// The documented worker output sequence, including its ArrowInvalid-only retry.
fn reference_results<'py>(
    py: Python<'py>,
    arrow_type: &Bound<'py, PyAny>,
    values: &Bound<'py, PyList>,
    safe: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let pa = py.import("pyarrow")?;
    match pa.call_method1("array", (values, arrow_type)) {
        Ok(array) => Ok(array),
        Err(error) if error.is_instance(py, &pa.getattr("ArrowInvalid")?) => {
            let kwargs = PyDict::new(py);
            kwargs.set_item("safe", safe)?;
            pa.call_method1("array", (values,))?
                .call_method("cast", (arrow_type,), Some(&kwargs))
        }
        Err(error) => Err(error),
    }
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
            spark_python_arrow::udf::import_data_type(&data_type).unwrap_err(),
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
fn extension_results_select_python_fallback_without_lowering_the_type() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = extension_context(py, "spark_python_arrow.tests.result_extension")?;
        let data_type = evaluate(py, &locals, "extension_type")?;
        let none = py.None().into_bound(py);
        let unsupported = result_spec(py, &data_type, &none, &none, true, &none, || {
            panic!("an unsupported extension must not resolve the Variant class")
        })?;
        let reason = match unsupported {
            Ok(_) => panic!("an extension result must not use its primitive storage builder"),
            Err(reason) => reason,
        };
        assert!(reason.contains("extension types"), "{reason}");
        let wrong_type = evaluate(py, &locals, "object()")?;
        let failure = result_spec(py, &wrong_type, &none, &none, true, &none, || {
            panic!("an invalid Arrow type must not resolve the Variant class")
        });
        assert!(matches!(failure, Err(error) if error.is_instance_of::<PyTypeError>(py)));
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
fn non_variant_and_unsupported_policies_do_not_resolve_the_variant_class() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = variant_context(py)?;
        let none = py.None().into_bound(py);
        for (expression, supported) in [("pa.int64()", true), ("pa.list_(pa.int64())", false)] {
            let arrow_type = evaluate(py, &locals, expression)?;
            let selected = result_spec(py, &arrow_type, &none, &none, true, &none, || {
                panic!("non-Variant and unsupported Arrow types must not resolve VariantVal")
            })?;
            assert_eq!(selected.is_ok(), supported);
        }
        let arrow_type = evaluate(py, &locals, "arrow_type")?;
        let spark_type = evaluate(py, &locals, "spark_type")?;
        let converter = evaluate(py, &locals, "converter")?;
        let sql_types = evaluate(py, &locals, "SqlTypes")?;
        for (spark_type, converter) in [(&spark_type, &none), (&none, &converter)] {
            let selected = result_spec(
                py,
                &arrow_type,
                spark_type,
                converter,
                true,
                &sql_types,
                || panic!("an unsupported converter must not resolve VariantVal"),
            )?;
            assert!(selected.is_err());
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn supported_variant_policy_resolves_once_and_propagates_resolver_errors() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = variant_context(py)?;
        let arrow_type = evaluate(py, &locals, "arrow_type")?;
        let spark_type = evaluate(py, &locals, "spark_type")?;
        let converter = evaluate(py, &locals, "converter")?;
        let sql_types = evaluate(py, &locals, "SqlTypes")?;
        let mut calls = 0;
        let failure = result_spec(
            py,
            &arrow_type,
            &spark_type,
            &converter,
            true,
            &sql_types,
            || {
                calls += 1;
                Err(PyImportError::new_err("VariantVal resolver failed"))
            },
        );
        assert_eq!(calls, 1);
        let error = match failure {
            Err(error) => error,
            Ok(_) => panic!("a resolver failure must remain a Python error"),
        };
        assert!(error.is_instance_of::<PyImportError>(py));
        assert!(error.to_string().contains("VariantVal resolver failed"));

        let missing = result_spec(
            py,
            &arrow_type,
            &spark_type,
            &converter,
            true,
            &sql_types,
            || Ok(None),
        )?;
        assert!(matches!(missing, Err(reason) if reason.contains("VariantVal class")));

        let selected = result_spec(
            py,
            &arrow_type,
            &spark_type,
            &converter,
            true,
            &sql_types,
            || {
                calls += 1;
                Ok(Some(
                    evaluate(py, &locals, "VariantVal")?.cast_into::<PyType>()?,
                ))
            },
        )?;
        assert!(selected.is_ok());
        assert_eq!(calls, 2);
        Ok(())
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
fn exact_scalar_extraction_rejects_coercion_and_clears_rejected_errors() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        py.run(
            c"class IntSubclass(int): pass\nclass StrSubclass(str): pass",
            Some(&locals),
            Some(&locals),
        )?;
        let truth = evaluate(py, &locals, "True")?;
        assert_eq!(exact_bool(&truth), Some(true));
        assert_eq!(exact_i64(&truth), None);
        let integer = evaluate(py, &locals, "7")?;
        assert_eq!(exact_i64(&integer), Some(7));
        assert_eq!(exact_bool(&integer), None);
        assert_eq!(exact_f64(&integer), None);
        assert_eq!(exact_f64(&evaluate(py, &locals, "1.25")?), Some(1.25));
        for expression in ["IntSubclass(7)", "2**100", "-(2**100)"] {
            assert_eq!(exact_i64(&evaluate(py, &locals, expression)?), None);
            assert!(!PyErr::occurred(py));
        }
        for expression in ["StrSubclass('text')", "'\\ud800'"] {
            assert_eq!(exact_str(&evaluate(py, &locals, expression)?), None);
            assert!(!PyErr::occurred(py));
        }
        let text = evaluate(py, &locals, "'a\\x00b'")?;
        assert_eq!(exact_str(&text).as_deref(), Some("a\0b"));
        let bytes = evaluate(py, &locals, "b'ab'")?;
        assert_eq!(exact_bytes(&bytes), Some(b"ab".as_slice()));
        assert_eq!(extract_binary(&bytes), Some(b"ab".to_vec()));
        let mutable = evaluate(py, &locals, "bytearray(b'ab')")?;
        assert_eq!(exact_bytes(&mutable), None);
        assert_eq!(extract_binary(&mutable), Some(b"ab".to_vec()));
        for expression in ["[97, 98]", "(97, 98)", "'ab'"] {
            assert_eq!(extract_binary(&evaluate(py, &locals, expression)?), None);
        }
        let error = utf8_into_python(py, &[0xff]).unwrap_err();
        assert!(error.is_instance_of::<PyUnicodeDecodeError>(py));
        Ok(())
    })
    .unwrap();
}

#[test]
fn exact_primitive_results_match_pyarrow_without_python_value_fallback() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        for (data_type, values) in [
            ("pa.int8()", "[-128, None, 127]"),
            ("pa.int16()", "[-32768, None, 32767]"),
            ("pa.int32()", "[-(2**31), None, 2**31 - 1]"),
            ("pa.int64()", "[-(2**63), None, 2**63 - 1]"),
            ("pa.float32()", "[-1.25, None, 2**24, 0, -0.0]"),
            ("pa.float64()", "[-1.25, None, 2**53, 0, -0.0]"),
            ("pa.bool_()", "[True, None, False]"),
            ("pa.string()", "['hello', None, 'a\\x00b', '']"),
            ("pa.large_string()", "['hello', None, 'a\\x00b', '']"),
            ("pa.binary()", "[b'hello', None, b'a\\x00b', b'']"),
            ("pa.large_binary()", "[b'hello', None, b'a\\x00b', b'']"),
        ] {
            let arrow_type = evaluate(py, &locals, data_type)?;
            let values = evaluate(py, &locals, values)?.cast_into::<PyList>()?;
            let expected = reference_results(py, &arrow_type, &values, true)?;
            let actual = build_results(py, &arrow_type, &values, true)?;
            assert_eq!(actual.pyarrow_values, 0, "{data_type}");
            assert!(!actual.retried, "{data_type}");
            assert_array_equal(&actual.array, &expected)?;
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn subclass_results_use_fallback_and_match_pyarrow() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        py.run(
            c"class IntSubclass(int): pass\nclass StrSubclass(str): pass",
            Some(&locals),
            Some(&locals),
        )?;
        // For variable-width columns, the capacity check cannot prove that a subclass fits.
        // It sends the prefix and suffix to PyArrow together, rather than concatenating them.
        for (data_type, values, pyarrow_values) in [
            ("pa.int64()", "[1, IntSubclass(2), None, 3]", 3),
            ("pa.string()", "['a', StrSubclass('b'), None, 'c']", 4),
        ] {
            let arrow_type = evaluate(py, &locals, data_type)?;
            let values = evaluate(py, &locals, values)?.cast_into::<PyList>()?;
            let expected = reference_results(py, &arrow_type, &values, true)?;
            let actual = build_results(py, &arrow_type, &values, true)?;
            assert_eq!(actual.pyarrow_values, pyarrow_values, "{data_type}");
            assert!(!actual.retried);
            assert_array_equal(&actual.array, &expected)?;
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn overflowing_results_follow_pyarrow_safe_and_unsafe_retry() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        for (data_type, values) in [
            ("pa.int8()", "[1, 128]"),
            ("pa.int64()", "[1, 2**63]"),
            ("pa.float32()", "[1, 2**24 + 1]"),
            ("pa.float64()", "[1, 2**53 + 1]"),
        ] {
            let arrow_type = evaluate(py, &locals, data_type)?;
            let values = evaluate(py, &locals, values)?.cast_into::<PyList>()?;
            for safe in [true, false] {
                let expected = reference_results(py, &arrow_type, &values, safe);
                let actual = build_results(py, &arrow_type, &values, safe);
                match (actual, expected) {
                    (Ok(actual), Ok(expected)) => {
                        assert!(actual.pyarrow_values > 0, "{data_type}");
                        assert_array_equal(&actual.array, &expected)?;
                    }
                    (Err(actual), Err(expected)) => {
                        assert!(
                            actual.get_type(py).is(expected.get_type(py)),
                            "{data_type}, safe={safe}: {actual}; expected {expected}"
                        );
                    }
                    (actual, expected) => panic!(
                        "{data_type}, safe={safe}: actual success {}, expected success {}",
                        actual.is_ok(),
                        expected.is_ok()
                    ),
                }
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn held_mutable_results_are_read_at_finish_and_extend_copies_the_list() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        let arrow_type = evaluate(py, &locals, "pa.binary()")?;
        let values = evaluate(py, &locals, "[b'prefix', bytearray(b'b'), b'tail']")?
            .cast_into::<PyList>()?;
        let spec = primitive_spec(py, &arrow_type, true)?;
        let mut column = ResultColumn::new(&spec, 4);
        column.extend(py, &values);
        values.get_item(1)?.set_item(0, 99)?;
        values.set_item(2, PyBytes::new(py, b"replacement"))?;
        column.push(py, py.None().into_bound(py));
        let actual = column.finish(py)?;
        let expected = evaluate(py, &locals, "pa.array([b'prefix', b'c', b'tail', None])")?;
        assert_array_equal(&actual.array, &expected)
    })
    .unwrap();
}

#[test]
fn lowered_variable_width_capacity_falls_back_for_the_whole_column() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        let arrow_type = evaluate(py, &locals, "pa.string()")?;
        let values = evaluate(py, &locals, "['ab', 'cd', None]")?.cast_into::<PyList>()?;
        let spec = primitive_spec(py, &arrow_type, true)?.with_value_capacity(3);
        let mut column = ResultColumn::new(&spec, values.len());
        column.extend(py, &values);
        let actual = column.finish(py)?;
        assert_eq!(actual.pyarrow_values, 3);
        assert!(!actual.retried);
        let expected = reference_results(py, &arrow_type, &values, true)?;
        assert_array_equal(&actual.array, &expected)
    })
    .unwrap();
}

#[test]
fn input_columns_match_pyarrow_cells_for_supported_primitive_types() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        let none = py.None().into_bound(py);
        let convert_input = evaluate(py, &locals, "lambda column, converter: column.to_pylist()")?;
        for (data_type, values) in [
            ("pa.int8()", "[-128, None, 127]"),
            ("pa.int16()", "[-32768, None, 32767]"),
            ("pa.int32()", "[-(2**31), None, 2**31 - 1]"),
            ("pa.int64()", "[-(2**63), None, 2**63 - 1]"),
            ("pa.float32()", "[-1.25, None, 2.5]"),
            ("pa.float64()", "[-1.25, None, 2.5]"),
            ("pa.bool_()", "[True, None, False]"),
            ("pa.string()", "['hello', None, 'a\\x00b']"),
            ("pa.large_string()", "['hello', None, 'a\\x00b']"),
            (
                "pa.date32()",
                "[datetime.date.min, None, datetime.date.max]",
            ),
        ] {
            let array = evaluate(
                py,
                &locals,
                &format!("pa.array({values}, type={data_type})"),
            )?;
            let expected = array.call_method0("to_pylist")?;
            let rust_type = import_data_type(&array.getattr("type")?)?;
            let spec = input_spec(&rust_type, &none, &none, &none)?
                .expect("the primitive type must be supported");
            let batch = py
                .import("pyarrow")?
                .call_method1("record_batch", (PyList::new(py, [array])?, vec!["x"]))?;
            let input = InputColumns::import(&batch, &[spec], &convert_input)?;
            assert_eq!(input.num_rows(), 3);
            drop(batch);
            for row in 0..input.num_rows() {
                let actual = input.value(py, 0, row)?;
                let expected = expected.get_item(row)?;
                assert!(actual.eq(&expected)?, "{data_type}, row={row}");
                assert!(
                    actual.get_type().is(expected.get_type()),
                    "{data_type}, row={row}"
                );
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn binary_input_policy_selects_bytes_or_bytearray() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        let none = py.None().into_bound(py);
        // Binary policy belongs to the caller: provide its type identity and converter.
        let binary_class = evaluate(py, &locals, "type('BinaryType', (), {})")?;
        let binary_type = binary_class.call0()?;
        let sql_types = py.import("types")?.getattr("SimpleNamespace")?.call0()?;
        sql_types.setattr("BinaryType", binary_class)?;
        let convert_input = evaluate(
            py,
            &locals,
            "lambda column, converter: [converter(v) for v in column.to_pylist()]",
        )?;
        for data_type in ["pa.binary()", "pa.large_binary()"] {
            let array = evaluate(
                py,
                &locals,
                &format!("pa.array([b'hello', None, b'a\\x00b'], type={data_type})"),
            )?;
            let expected = array.call_method0("to_pylist")?;
            let rust_type = import_data_type(&array.getattr("type")?)?;
            assert!(input_spec(&rust_type, &none, &none, &none)?.is_err());
            let batch = py
                .import("pyarrow")?
                .call_method1("record_batch", (PyList::new(py, [array])?, vec!["x"]))?;
            for policy in [
                "lambda v: v",
                "lambda v: None if v is None else bytearray(v)",
            ] {
                let converter = evaluate(py, &locals, policy)?;
                let spec = input_spec(&rust_type, &binary_type, &converter, &sql_types)?
                    .expect("both configured binary policies are supported");
                let input = InputColumns::import(&batch, &[spec], &convert_input)?;
                for row in 0..input.num_rows() {
                    let actual = input.value(py, 0, row)?;
                    let expected = converter.call1((expected.get_item(row)?,))?;
                    assert!(actual.eq(&expected)?, "{data_type}, {policy}, row={row}");
                    assert!(
                        actual.get_type().is(expected.get_type()),
                        "{data_type}, {policy}, row={row}"
                    );
                }
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn input_columns_reject_mismatched_specs_and_out_of_range_cells() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        let none = py.None().into_bound(py);
        let convert_input = evaluate(py, &locals, "lambda column, converter: column.to_pylist()")?;
        let batch = evaluate(
            py,
            &locals,
            "pa.record_batch([pa.array([1, 2], type=pa.int64())], names=['x'])",
        )?;
        let wrong = input_spec(&DataType::Int32, &none, &none, &none)?.expect("int32 is supported");
        assert!(InputColumns::import(&batch, &[wrong], &convert_input).is_err());
        assert!(InputColumns::import(&batch, &[], &convert_input).is_err());
        let right = input_spec(&DataType::Int64, &none, &none, &none)?.expect("int64 is supported");
        let columns = InputColumns::import(&batch, &[right], &convert_input)?;
        for (column, row) in [(1, 0), (0, 2), (usize::MAX, 0), (0, usize::MAX)] {
            let error = columns.value(py, column, row).unwrap_err();
            assert!(error.is_instance_of::<PyIndexError>(py));
        }
        assert_eq!(columns.value(py, 0, 1)?.extract::<i64>()?, 2);
        Ok(())
    })
    .unwrap();
}
