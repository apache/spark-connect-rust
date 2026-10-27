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

//! UDF input conversion and deferred Python errors, using valid Arrow buffer layouts.

use arrow::datatypes::DataType;
use pyo3::exceptions::{PyRuntimeError, PyUnicodeDecodeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use spark_python_arrow::bridge::import_data_type;
use spark_python_arrow::udf::{input_spec, InputColumns};

fn context(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let locals = PyDict::new(py);
    locals.set_item("pa", py.import("pyarrow")?)?;
    // This is the worker's input sequence for the primitive types tested here:
    // materialize the whole column, then apply its optional caller converter.
    py.run(
        c"calls = []
def convert_input(column, converter):
    calls.append((column, converter))
    values = column.to_pylist()
    return values if converter is None else [converter(value) for value in values]
",
        Some(&locals),
        Some(&locals),
    )?;
    Ok(locals)
}

fn get<'py>(locals: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    Ok(locals.get_item(name)?.expect("fixture name is defined"))
}

fn primitive_input<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    convert_input: &Bound<'py, PyAny>,
) -> PyResult<InputColumns> {
    let none = py.None().into_bound(py);
    let data_type = import_data_type(&array.getattr("type")?)?;
    let first = input_spec(&DataType::Int64, &none, &none, &none)?
        .expect("the leading integer column is supported");
    let second = input_spec(&data_type, &none, &none, &none)?
        .expect("the tested primitive column is supported");
    let pa = py.import("pyarrow")?;
    let leading = pa.call_method1("array", (vec![1_i64; array.len()?],))?;
    let batch = pa.call_method1(
        "record_batch",
        (
            PyList::new(py, [leading, array.clone()])?,
            vec!["first", "second"],
        ),
    )?;
    InputColumns::import(&batch, &[first, second], convert_input)
}

fn assert_same_error(py: Python<'_>, actual: &PyErr, expected: &PyErr) -> PyResult<()> {
    assert!(actual.get_type(py).is(expected.get_type(py)));
    assert_eq!(
        actual.value(py).str()?.extract::<String>()?,
        expected.value(py).str()?.extract::<String>()?
    );
    Ok(())
}

fn assert_one_callback(locals: &Bound<'_, PyDict>, array: &Bound<'_, PyAny>) -> PyResult<()> {
    let calls = get(locals, "calls")?;
    assert_eq!(calls.len()?, 1);
    let call = calls.get_item(0)?;
    assert!(call
        .get_item(0)?
        .call_method1("equals", (array,))?
        .extract::<bool>()?);
    assert!(call.get_item(1)?.is_none());
    Ok(())
}

#[test]
fn out_of_python_range_date_uses_the_original_column_callback() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        let convert_input = get(&locals, "convert_input")?;
        let none = py.None().into_bound(py);
        for days in [-719_163_i32, 2_932_897, i32::MIN, i32::MAX] {
            locals.set_item("days", days)?;
            // Every day value fits Arrow date32, even when Python cannot represent it.
            let array = py.eval(
                c"pa.array([0, days, None], type=pa.date32())",
                Some(&locals),
                None,
            )?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("full", true)?;
            array.call_method("validate", (), Some(&kwargs))?;
            let expected = convert_input.call1((&array, &none)).unwrap_err();
            get(&locals, "calls")?.call_method0("clear")?;
            let input = primitive_input(py, &array, &convert_input)?;
            assert_eq!(input.value(py, 0, 0)?.extract::<i64>()?, 1);
            assert_eq!(
                input.value(py, 1, 0)?.str()?.extract::<String>()?,
                "1970-01-01"
            );
            assert!(input.value(py, 1, 2)?.is_none());
            assert_eq!(get(&locals, "calls")?.len()?, 0);
            let actual = input.value(py, 1, 1).unwrap_err();
            assert_same_error(py, &actual, &expected)?;
            assert_one_callback(&locals, &array)?;
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn invalid_utf8_payload_defers_to_the_python_column_error() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        locals.set_item("struct", py.import("struct")?)?;
        let convert_input = get(&locals, "convert_input")?;
        let none = py.None().into_bound(py);
        for (type_name, offset_format) in [("string", "<4i"), ("large_string", "<4q")] {
            locals.set_item(
                "data_type",
                py.import("pyarrow")?.getattr(type_name)?.call0()?,
            )?;
            locals.set_item("offset_format", offset_format)?;
            // All offsets are monotonic and within this seven-byte payload. Only the text
            // in the second cell is invalid UTF-8; no malformed offset or pointer is used.
            let array = py.eval(
                cr"pa.Array.from_buffers(data_type, 3, [
    None,
    pa.py_buffer(struct.pack(offset_format, 0, 2, 3, 7)),
    pa.py_buffer(b'ok\xfflast'),
])",
                Some(&locals),
                Some(&locals),
            )?;
            array.call_method0("validate")?;
            let expected = convert_input.call1((&array, &none)).unwrap_err();
            assert!(expected.is_instance_of::<PyUnicodeDecodeError>(py));
            get(&locals, "calls")?.call_method0("clear")?;
            let input = primitive_input(py, &array, &convert_input)?;
            assert_eq!(input.value(py, 1, 0)?.extract::<String>()?, "ok");
            assert_eq!(input.value(py, 1, 2)?.extract::<String>()?, "last");
            assert_eq!(get(&locals, "calls")?.len()?, 0);
            let actual = input.value(py, 1, 1).unwrap_err();
            assert_same_error(py, &actual, &expected)?;
            assert_one_callback(&locals, &array)?;
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn callback_exception_is_propagated_without_replacement() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        py.run(
            c"sentinel = LookupError('caller conversion failed')
def convert_input(column, converter):
    calls.append((column, converter))
    raise sentinel
array = pa.array([2932897], type=pa.date32())
",
            Some(&locals),
            Some(&locals),
        )?;
        let array = get(&locals, "array")?;
        let input = primitive_input(py, &array, &get(&locals, "convert_input")?)?;
        let error = input.value(py, 1, 0).unwrap_err();
        assert!(error.value(py).is(get(&locals, "sentinel")?));
        assert_one_callback(&locals, &array)
    })
    .unwrap();
}

#[test]
fn unexpected_callback_success_is_not_silently_accepted() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        py.run(
            c"def convert_input(column, converter):
    calls.append((column, converter))
    return ['unexpected success']
array = pa.array([2932897], type=pa.date32())
",
            Some(&locals),
            Some(&locals),
        )?;
        let array = get(&locals, "array")?;
        let input = primitive_input(py, &array, &get(&locals, "convert_input")?)?;
        let error = input.value(py, 1, 0).unwrap_err();
        assert!(error.is_instance_of::<PyRuntimeError>(py));
        assert_eq!(
            error.value(py).str()?.extract::<String>()?,
            "PySpark converted input 1, which the native converter cannot read"
        );
        assert_one_callback(&locals, &array)
    })
    .unwrap();
}

#[test]
fn variant_input_preserves_class_values_and_masked_null_parts() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        py.run(
            c"class VariantType: pass
class VariantVal:
    def __init__(self, value, metadata):
        self.value = value
        self.metadata = metadata
    def __eq__(self, other):
        return type(self) is type(other) and (self.value, self.metadata) == (
            other.value, other.metadata
        )
class SqlTypes: pass
SqlTypes.VariantType = VariantType
spark_type = VariantType()
converter = lambda value: None if value is None else VariantVal(
    value['value'], value['metadata']
)
",
            Some(&locals),
            Some(&locals),
        )?;
        for width in ["binary", "large_binary"] {
            locals.set_item("part_type", py.import("pyarrow")?.getattr(width)?.call0()?)?;
            py.run(
                cr"fields = [
    pa.field('value', part_type, nullable=False),
    pa.field('metadata', part_type, nullable=False, metadata={b'variant': b'true'}),
]
array = pa.StructArray.from_arrays([
    pa.array([b'a', None, b'\x00'], type=part_type),
    pa.array([b'm', None, b'\xff'], type=part_type),
], fields=fields, mask=pa.array([False, True, False]))
array.validate(full=True)
expected = convert_input(array, converter)
calls.clear()
batch = pa.record_batch([array], names=['variant'])
",
                Some(&locals),
                Some(&locals),
            )?;
            let data_type = import_data_type(&get(&locals, "array")?.getattr("type")?)?;
            let spec = input_spec(
                &data_type,
                &get(&locals, "spark_type")?,
                &get(&locals, "converter")?,
                &get(&locals, "SqlTypes")?,
            )?
            .expect("tagged Variant inputs are supported");
            let input = InputColumns::import(
                &get(&locals, "batch")?,
                &[spec],
                &get(&locals, "convert_input")?,
            )?;
            for row in 0..input.num_rows() {
                let actual = input.value(py, 0, row)?;
                let expected = get(&locals, "expected")?.get_item(row)?;
                assert!(actual.eq(&expected)?);
                assert!(actual.get_type().is(expected.get_type()));
            }
            // The parent-null row is valid even though its non-nullable parts are null:
            // those child nulls are masked, so no fallback or child access is required.
            assert!(input.value(py, 0, 1)?.is_none());
            assert_eq!(get(&locals, "calls")?.len()?, 0);
        }
        Ok(())
    })
    .unwrap();
}
