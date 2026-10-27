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

//! UDF result-policy checks, without a Spark server. The fixtures model PySpark's nullable
//! top-level String/Binary/Variant converters. Logging and injected failures make scheduling
//! observable; they are test instrumentation, not support for arbitrary stateful converters.
//! Use --include-ignored with official PySpark installed to run the optional comparison.

use std::ffi::CString;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyType};
use spark_python_arrow::udf::{result_spec, Finished, ResultColumn, ResultSpec};

#[derive(Clone, Copy, Debug)]
enum Mode {
    Push,
    Extend,
}

const MODES: [Mode; 2] = [Mode::Push, Mode::Extend];

fn context(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let locals = PyDict::new(py);
    py.run(
        c"import pyarrow as pa
from types import SimpleNamespace

class StringType: pass
class BinaryType: pass
class VariantType: pass
sql_types = SimpleNamespace(
    StringType=StringType, BinaryType=BinaryType, VariantType=VariantType
)

class VariantVal:
    def __init__(self, value, metadata):
        self.value = value
        self.metadata = metadata

class VariantSubclass(VariantVal): pass
class BytesSubclass(bytes): pass

class Text:
    def __init__(self, text):
        self.text = text
    def __str__(self):
        return self.text

class BadText:
    def __str__(self):
        raise ValueError('text conversion failed')

variant_type = pa.struct([
    pa.field('value', pa.binary(), nullable=False),
    pa.field('metadata', pa.binary(), nullable=False, metadata={b'variant': b'true'}),
])

def convert_string(value):
    if value is None:
        return None
    return str(value).lower() if isinstance(value, bool) else str(value)

def convert_binary(value):
    if value is None:
        return None
    assert isinstance(value, (bytes, bytearray))
    return bytes(value)

def convert_variant(value):
    if value is None:
        return None
    if isinstance(value, VariantVal):
        return {'value': value.value, 'metadata': value.metadata}
    raise ValueError('malformed variant')

events = []

def tracked_string(value):
    events.append(value)
    return convert_string(value)

def tracked_binary(value):
    events.append(value)
    return convert_binary(value)

def tracked_variant(value):
    events.append(value)
    return convert_variant(value)

def faulty_string(value):
    events.append(value)
    if value == 7:
        raise ValueError('injected converter failure')
    return convert_string(value)

def later_udf_error(value):
    if value == 2:
        raise RuntimeError('later UDF failure')
    return 7

def reference_results(values, converter, arrow_type):
    converted = [converter(value) for value in values]
    try:
        return pa.array(converted, type=arrow_type)
    except pa.ArrowInvalid:
        return pa.array(converted).cast(arrow_type, safe=True)
",
        Some(&locals),
        Some(&locals),
    )?;
    Ok(locals)
}

fn evaluate<'py>(
    py: Python<'py>,
    locals: &Bound<'py, PyDict>,
    expression: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let code = CString::new(expression).expect("test expression contains no NUL");
    py.eval(&code, Some(locals), Some(locals))
}

fn policy<'py>(
    py: Python<'py>,
    locals: &Bound<'py, PyDict>,
    arrow_type: &Bound<'py, PyAny>,
    spark_type: &Bound<'py, PyAny>,
    converter: &Bound<'py, PyAny>,
) -> PyResult<ResultSpec> {
    let sql_types = evaluate(py, locals, "sql_types")?;
    let spec = result_spec(
        py,
        arrow_type,
        spark_type,
        converter,
        true,
        &sql_types,
        || {
            Ok(Some(
                evaluate(py, locals, "VariantVal")?.cast_into::<PyType>()?,
            ))
        },
    )?;
    Ok(spec.expect("fixture type is supported"))
}

fn fixture_policy<'py>(
    py: Python<'py>,
    locals: &Bound<'py, PyDict>,
    arrow_type: &str,
    spark_type: &str,
    converter: &str,
) -> PyResult<ResultSpec> {
    policy(
        py,
        locals,
        &evaluate(py, locals, arrow_type)?,
        &evaluate(py, locals, spark_type)?,
        &evaluate(py, locals, converter)?,
    )
}

fn add_values<'py>(
    py: Python<'py>,
    column: &mut ResultColumn<'_>,
    values: &Bound<'py, PyList>,
    mode: Mode,
) {
    match mode {
        Mode::Push => {
            for value in values.iter() {
                column.push(py, value);
            }
        }
        Mode::Extend => column.extend(py, values),
    }
}

fn reference<'py>(
    py: Python<'py>,
    locals: &Bound<'py, PyDict>,
    values: &Bound<'py, PyList>,
    converter: &Bound<'py, PyAny>,
    arrow_type: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    evaluate(py, locals, "reference_results")?.call1((values, converter, arrow_type))
}

fn assert_same_result(
    py: Python<'_>,
    actual: PyResult<Finished<'_>>,
    expected: PyResult<Bound<'_, PyAny>>,
) -> PyResult<()> {
    match (actual, expected) {
        (Ok(actual), Ok(expected)) => assert!(
            actual
                .array
                .call_method1("equals", (&expected,))?
                .extract::<bool>()?,
            "actual {}; expected {expected}",
            actual.array
        ),
        (Err(actual), Err(expected)) => {
            assert!(actual.get_type(py).is(expected.get_type(py)));
            assert_eq!(actual.to_string(), expected.to_string());
        }
        (actual, expected) => panic!(
            "actual success {}, expected success {}",
            actual.is_ok(),
            expected.is_ok()
        ),
    }
    Ok(())
}

#[test]
fn string_conversion_matches_nullable_pyspark_policy_in_both_ingestion_modes() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        for mode in MODES {
            let locals = context(py)?;
            let values = evaluate(py, &locals, "[None, 'plain', True, False, 1, -7, 1.25]")?
                .cast_into::<PyList>()?;
            for arrow_expression in ["pa.string()", "pa.large_string()"] {
                evaluate(py, &locals, "events.clear()")?;
                let spec = fixture_policy(
                    py,
                    &locals,
                    arrow_expression,
                    "StringType()",
                    "tracked_string",
                )?;
                assert!(evaluate(py, &locals, "not events")?.extract::<bool>()?);
                let mut column = ResultColumn::new(&spec, values.len());
                add_values(py, &mut column, &values, mode);
                let actual = column.finish(py)?;
                assert_eq!(actual.pyarrow_values, 0);
                assert!(
                    evaluate(py, &locals, "events == [True, False, 1, -7, 1.25]")?
                        .extract::<bool>()?
                );
                let expected = reference(
                    py,
                    &locals,
                    &values,
                    &evaluate(py, &locals, "convert_string")?,
                    &evaluate(py, &locals, arrow_expression)?,
                );
                assert_same_result(py, Ok(actual), expected)?;
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn capacity_fallback_retains_converted_value_and_calls_converter_once() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        for mode in MODES {
            for capacity in [0, 1] {
                let locals = context(py)?;
                let values = evaluate(py, &locals, "[1, 2, 3]")?.cast_into::<PyList>()?;
                let spec =
                    fixture_policy(py, &locals, "pa.string()", "StringType()", "tracked_string")?
                        .with_value_capacity(capacity);
                let mut column = ResultColumn::new(&spec, values.len());
                add_values(py, &mut column, &values, mode);
                let actual = column.finish(py)?;
                assert_eq!(actual.pyarrow_values, 3);
                assert!(evaluate(py, &locals, "events == [1, 2, 3]")?.extract::<bool>()?);
                let expected = reference(
                    py,
                    &locals,
                    &values,
                    &evaluate(py, &locals, "convert_string")?,
                    &evaluate(py, &locals, "pa.string()")?,
                );
                assert_same_result(py, Ok(actual), expected)?;
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn held_string_and_binary_callbacks_observe_final_mutable_values() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        for mode in MODES {
            for (arrow_type, spark_type, converter, values, mutation) in [
                (
                    "pa.string()",
                    "StringType()",
                    "tracked_string",
                    "['prefix', Text('before'), None, True]",
                    "values[1].text = 'after'",
                ),
                (
                    "pa.binary()",
                    "BinaryType()",
                    "tracked_binary",
                    "[b'prefix', bytearray(b'before'), None, BytesSubclass(b'tail')]",
                    "values[1][:] = b'after'",
                ),
            ] {
                let locals = context(py)?;
                let values = evaluate(py, &locals, values)?.cast_into::<PyList>()?;
                locals.set_item("values", &values)?;
                let spec = fixture_policy(py, &locals, arrow_type, spark_type, converter)?;
                let mut column = ResultColumn::new(&spec, values.len());
                add_values(py, &mut column, &values, mode);
                assert!(evaluate(py, &locals, "not events")?.extract::<bool>()?);
                py.run(
                    &CString::new(mutation).unwrap(),
                    Some(&locals),
                    Some(&locals),
                )?;
                let actual = column.finish(py)?;
                assert!(actual.pyarrow_values > 0);
                assert!(
                    evaluate(py, &locals, "len(events) == 2 and events[0] is values[1]")?
                        .extract::<bool>()?
                );
                let pure_converter = converter.replacen("tracked_", "convert_", 1);
                let expected = reference(
                    py,
                    &locals,
                    &values,
                    &evaluate(py, &locals, &pure_converter)?,
                    &evaluate(py, &locals, arrow_type)?,
                );
                assert_same_result(py, Ok(actual), expected)?;
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn converter_failures_are_delayed_and_match_reference_error_order() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        for mode in MODES {
            for (arrow_type, spark_type, converter, values) in [
                (
                    "pa.string()",
                    "StringType()",
                    "faulty_string",
                    "[1, 7, BadText()]",
                ),
                (
                    "pa.string()",
                    "StringType()",
                    "convert_string",
                    "['prefix', BadText(), 7]",
                ),
                (
                    "pa.binary()",
                    "BinaryType()",
                    "convert_binary",
                    "[b'prefix', object(), None]",
                ),
            ] {
                let locals = context(py)?;
                let values = evaluate(py, &locals, values)?.cast_into::<PyList>()?;
                let spec = fixture_policy(py, &locals, arrow_type, spark_type, converter)?;
                let mut column = ResultColumn::new(&spec, values.len());
                add_values(py, &mut column, &values, mode);
                // Neither ingestion API raises. Failed-state ingestion also preserves the
                // first error rather than running a later callback.
                column.push(py, py.None().into_bound(py));
                column.extend(py, &PyList::empty(py));
                let actual = column.finish(py);
                assert!(actual.is_err());
                let expected = reference(
                    py,
                    &locals,
                    &values,
                    &evaluate(py, &locals, converter)?,
                    &evaluate(py, &locals, arrow_type)?,
                );
                assert_same_result(py, actual, expected)?;
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn later_udf_error_wins_when_caller_finishes_only_after_all_udf_calls() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        for mode in MODES {
            let locals = context(py)?;
            let spec = fixture_policy(py, &locals, "pa.string()", "StringType()", "faulty_string")?;
            let udf = evaluate(py, &locals, "later_udf_error")?;
            let outcome = (|| -> PyResult<Finished<'_>> {
                let mut column = ResultColumn::new(&spec, 3);
                let collected = PyList::empty(py);
                for row in 0..3 {
                    let value = udf.call1((row,))?;
                    match mode {
                        Mode::Push => column.push(py, value),
                        Mode::Extend => collected.append(value)?,
                    }
                }
                if matches!(mode, Mode::Extend) {
                    column.extend(py, &collected);
                }
                column.finish(py)
            })();
            let error = outcome.err().expect("the later UDF call must fail");
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            assert!(error.to_string().contains("later UDF failure"));
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn large_integer_string_conversion_is_deferred_until_finish() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        for mode in MODES {
            for expression in ["[2**80]", "[2**16000]"] {
                let locals = context(py)?;
                let values = evaluate(py, &locals, expression)?.cast_into::<PyList>()?;
                let spec =
                    fixture_policy(py, &locals, "pa.string()", "StringType()", "tracked_string")?;
                let mut column = ResultColumn::new(&spec, values.len());
                add_values(py, &mut column, &values, mode);
                assert!(evaluate(py, &locals, "not events")?.extract::<bool>()?);
                let actual = column.finish(py);
                assert!(evaluate(py, &locals, "len(events) == 1")?.extract::<bool>()?);
                // Some Python versions reject the larger value under their digit limit.
                // Either outcome must match the same nullable converter at finish time.
                let expected = reference(
                    py,
                    &locals,
                    &values,
                    &evaluate(py, &locals, "convert_string")?,
                    &evaluate(py, &locals, "pa.string()")?,
                );
                assert_same_result(py, actual, expected)?;
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn native_variant_results_include_nulls_and_observe_mutation_before_finish() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        for mode in MODES {
            let locals = context(py)?;
            let value = evaluate(py, &locals, "VariantVal(b'old', b'metadata')")?;
            let values = PyList::new(py, [value.clone(), py.None().into_bound(py), value.clone()])?;
            let spec = fixture_policy(
                py,
                &locals,
                "variant_type",
                "VariantType()",
                "tracked_variant",
            )?;
            let mut column = ResultColumn::new(&spec, values.len());
            add_values(py, &mut column, &values, mode);
            value.setattr("value", evaluate(py, &locals, "b'new'")?)?;
            let actual = column.finish(py)?;
            assert_eq!(actual.pyarrow_values, 0);
            assert!(!actual.retried);
            assert!(evaluate(py, &locals, "not events")?.extract::<bool>()?);
            let expected = reference(
                py,
                &locals,
                &values,
                &evaluate(py, &locals, "convert_variant")?,
                &evaluate(py, &locals, "variant_type")?,
            );
            assert_same_result(py, Ok(actual), expected)?;
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn variant_subclass_non_bytes_and_capacity_use_fallback() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        for mode in MODES {
            for (expression, capacity) in [
                ("[VariantSubclass(b'v', b'm'), None]", usize::MAX),
                (
                    "[VariantVal(bytearray(b'v'), bytearray(b'm')), None]",
                    usize::MAX,
                ),
                ("[VariantVal(b'vv', b'mm'), None]", 1),
            ] {
                let locals = context(py)?;
                let values = evaluate(py, &locals, expression)?.cast_into::<PyList>()?;
                let spec = fixture_policy(
                    py,
                    &locals,
                    "variant_type",
                    "VariantType()",
                    "tracked_variant",
                )?
                .with_value_capacity(capacity);
                let mut column = ResultColumn::new(&spec, values.len());
                add_values(py, &mut column, &values, mode);
                let actual = column.finish(py)?;
                assert_eq!(actual.pyarrow_values, values.len());
                assert!(evaluate(py, &locals, "len(events) == 1")?.extract::<bool>()?);
                let expected = reference(
                    py,
                    &locals,
                    &values,
                    &evaluate(py, &locals, "convert_variant")?,
                    &evaluate(py, &locals, "variant_type")?,
                );
                assert_same_result(py, Ok(actual), expected)?;
            }
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn variant_converter_failure_precedes_arrow_rejection() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        for mode in MODES {
            let locals = context(py)?;
            // The first result reaches PyArrow only if every converter succeeds. Its invalid
            // binary part must not mask the second result's conversion failure.
            let values = evaluate(py, &locals, "[VariantVal(object(), b'm'), object()]")?
                .cast_into::<PyList>()?;
            let spec = fixture_policy(
                py,
                &locals,
                "variant_type",
                "VariantType()",
                "convert_variant",
            )?;
            let mut column = ResultColumn::new(&spec, values.len());
            add_values(py, &mut column, &values, mode);
            let actual = column.finish(py);
            assert!(matches!(&actual, Err(error) if error.is_instance_of::<PyValueError>(py)));
            let expected = reference(
                py,
                &locals,
                &values,
                &evaluate(py, &locals, "convert_variant")?,
                &evaluate(py, &locals, "variant_type")?,
            );
            assert_same_result(py, actual, expected)?;
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn dictionary_results_select_python_fallback_including_ordered_dictionaries() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        let none = py.None().into_bound(py);
        for ordered in ["False", "True"] {
            let arrow_type = evaluate(
                py,
                &locals,
                &format!("pa.dictionary(pa.int8(), pa.string(), ordered={ordered})"),
            )?;
            let selected = result_spec(py, &arrow_type, &none, &none, true, &none, || {
                panic!("dictionary fallback must not resolve VariantVal")
            })?;
            assert!(matches!(selected, Err(reason) if reason.contains("dictionary")));
        }
        Ok(())
    })
    .unwrap();
}

#[test]
fn nested_dictionary_extension_results_select_python_fallback_before_import() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        py.run(
            c"class BinaryExtension(pa.ExtensionType):
    def __init__(self):
        super().__init__(pa.binary(), 'spark.test.udf_result_binary')
    def __arrow_ext_serialize__(self):
        return b''
    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return cls()

dictionary_extension = pa.dictionary(pa.int8(), BinaryExtension())
types = [
    pa.list_(dictionary_extension),
    pa.struct([pa.field('nested', dictionary_extension)]),
    pa.map_(pa.string(), dictionary_extension),
    pa.struct([
        pa.field('value', dictionary_extension, nullable=False),
        pa.field('metadata', pa.binary(), nullable=False, metadata={b'variant': b'true'}),
    ]),
]
",
            Some(&locals),
            Some(&locals),
        )?;
        let none = py.None().into_bound(py);
        for arrow_type in evaluate(py, &locals, "types")?.try_iter()? {
            let arrow_type = arrow_type?;
            let selected = result_spec(py, &arrow_type, &none, &none, true, &none, || {
                panic!("unsupported nested types must not resolve VariantVal")
            })?;
            assert!(matches!(selected, Err(reason) if reason.contains("not supported")));
        }
        Ok(())
    })
    .unwrap();
}

#[test]
#[ignore = "requires official pyspark-client==4.2.0 and PyArrow"]
fn official_pyspark_result_converters_match_native_and_fallback_paths() {
    Python::initialize();
    Python::attach(|py| -> PyResult<()> {
        let locals = context(py)?;
        py.run(
            c"from pyspark.sql import types as sql_types
from pyspark.sql.conversion import LocalDataToArrowConversion
from pyspark.sql.pandas.types import to_arrow_type
VariantVal = sql_types.VariantVal
class OfficialVariantSubclass(VariantVal): pass
cases = [
    (sql_types.StringType(), [None, 'plain', True, False, 1, -7, 1.25, Text('text')]),
    (sql_types.StringType(), ['prefix', BadText()]),
    (sql_types.StringType(), [2**80]),
    (sql_types.StringType(), [2**16000]),
    (sql_types.BinaryType(), [b'prefix', bytearray(b'value'), None, BytesSubclass(b'tail')]),
    (sql_types.BinaryType(), [b'prefix', object()]),
    (sql_types.VariantType(), [VariantVal(b'v', b'm'), None]),
    (sql_types.VariantType(), [OfficialVariantSubclass(b'v', b'm'), None]),
    (sql_types.VariantType(), [VariantVal(bytearray(b'v'), bytearray(b'm')), None]),
    (sql_types.VariantType(), [VariantVal(object(), b'm'), object()]),
]
",
            Some(&locals),
            Some(&locals),
        )?;
        for case in evaluate(py, &locals, "cases")?.try_iter()? {
            let case = case?;
            let spark_type = case.get_item(0)?;
            let values = case.get_item(1)?.cast_into::<PyList>()?;
            let converter = evaluate(py, &locals, "LocalDataToArrowConversion._create_converter")?
                .call1((&spark_type,))?;
            let arrow_type = evaluate(py, &locals, "to_arrow_type")?.call1((&spark_type,))?;
            for mode in MODES {
                for capacity in [1, usize::MAX] {
                    let spec = policy(py, &locals, &arrow_type, &spark_type, &converter)?
                        .with_value_capacity(capacity);
                    let mut column = ResultColumn::new(&spec, values.len());
                    add_values(py, &mut column, &values, mode);
                    assert_same_result(
                        py,
                        column.finish(py),
                        reference(py, &locals, &values, &converter, &arrow_type),
                    )?;
                }
            }
        }
        Ok(())
    })
    .unwrap();
}
