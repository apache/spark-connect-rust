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

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BinaryBuilder, BooleanBufferBuilder, BooleanBuilder, Float32Array,
    Float64Builder, Int16Builder, Int32Builder, Int64Builder, Int8Builder, LargeBinaryBuilder,
    LargeStringBuilder, NullBufferBuilder, PrimitiveBuilder, StringBuilder, StructArray,
};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::{ArrowPrimitiveType, DataType, Float64Type};
use pyo3::exceptions::PyRuntimeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyType};

use crate::bridge::arrow_error;
use crate::scalar::{exact_bool, exact_bytes, exact_f64, exact_i64, exact_str, into_exact_bytes};
use crate::udf::input::{Column, Read};
use crate::udf::{export_array, import_data_type, is_spark_type, variant_fields};

/// The largest integer magnitudes PyArrow converts to float64 and float32, which it refuses
/// above 2^53 and 2^24.
const MAX_EXACT_F64_INT: u64 = 1 << 53;
const MAX_EXACT_F32_INT: u64 = 1 << 24;

/// Accepts a UDF's results for [`ResultColumn`], or returns why its native builder is unsupported.
/// `arrow_type` is the `pa.DataType` PySpark builds for `spark_return_type`, `converter` is
/// PySpark's result converter or `None`, `safecheck` is
/// `spark.sql.execution.pandas.convertToArrowArraySafely`, which PySpark's retry casts with, and
/// `sql_types` is the backend package's `sql.types` module, whose classes `spark_return_type`
/// comes from. For supported Variant results, `variant_class` resolves the class UDF code
/// constructs, which need not be the backend module's class. The resolver runs only after the
/// Arrow type and converter are accepted; its errors propagate. No package or class is inferred.
pub fn result_spec<'py>(
    py: Python<'py>,
    arrow_type: &Bound<'_, PyAny>,
    spark_return_type: &Bound<'_, PyAny>,
    converter: &Bound<'_, PyAny>,
    safecheck: bool,
    sql_types: &Bound<'_, PyAny>,
    variant_class: impl FnOnce() -> PyResult<Option<Bound<'py, PyType>>>,
) -> PyResult<Result<ResultSpec, String>> {
    // Lowering an extension type to its storage type would select the primitive
    // builder and export an array without the requested extension identity.
    if crate::bridge::is_extension_type(arrow_type)? {
        return Ok(Err(
            "Arrow extension types require the caller's Python converter".to_owned(),
        ));
    }
    let data_type = import_data_type(arrow_type)?;
    let Some(kind) = OutputKind::of(&data_type) else {
        return Ok(Err(format!(
            "{data_type} is not supported by the native UDF result converter"
        )));
    };
    // Build a variant only as PySpark's VariantType converter makes it.
    let converter = match (converter.is_none(), kind.converter_type()) {
        (true, _) if kind != OutputKind::Variant => None,
        (false, Some(name)) if is_spark_type(spark_return_type, sql_types, name)? => {
            Some(converter.clone().unbind())
        }
        _ => {
            return Ok(Err(format!(
                "{spark_return_type} needs PySpark's converter"
            )))
        }
    };
    let variant = match kind {
        // The caller knows the class UDF code constructs; it need not belong to `sql_types`.
        OutputKind::Variant => match variant_class()? {
            Some(class) => Some(class.into_any().unbind()),
            None => {
                return Ok(Err(
                    "VariantType requires the caller's VariantVal class".to_owned()
                ))
            }
        },
        _ => None,
    };
    let pyarrow = py.import("pyarrow")?;
    Ok(Ok(ResultSpec {
        kind,
        arrow_type: arrow_type.clone().unbind(),
        converter,
        variant,
        safecheck,
        value_capacity: i32::MAX as usize - 1,
        pyarrow_array: pyarrow.getattr("array")?.unbind(),
        concat_arrays: pyarrow.getattr("concat_arrays")?.unbind(),
        array_type: pyarrow.getattr("Array")?.unbind(),
        arrow_invalid: pyarrow.getattr("lib")?.getattr("ArrowInvalid")?.unbind(),
    }))
}

/// A UDF's accepted result type, and what PySpark's output step needs for it.
pub struct ResultSpec {
    kind: OutputKind,
    arrow_type: Py<PyAny>,
    /// PySpark's `StringType`, `BinaryType` or `VariantType` converter; every other result type
    /// handled by this native builder has none.
    converter: Option<Py<PyAny>>,
    /// PySpark's `VariantVal`, for a variant result.
    variant: Option<Py<PyAny>>,
    safecheck: bool,
    /// The most bytes the values of a string, binary or variant-part array can take, as its
    /// offsets are 32-bit: PyArrow's limit, one under `i32::MAX`, past which `pa.array` chunks.
    value_capacity: usize,
    pyarrow_array: Py<PyAny>,
    concat_arrays: Py<PyAny>,
    array_type: Py<PyAny>,
    arrow_invalid: Py<PyAny>,
}

impl ResultSpec {
    /// Lowers the short-offset buffer threshold before selecting the PyArrow fallback.
    ///
    /// This also permits testing chunk/fallback behavior without allocating 2 GiB. The threshold
    /// cannot be raised above its current value or the maximum supported by 32-bit offsets.
    pub fn with_value_capacity(self, value_capacity: usize) -> ResultSpec {
        ResultSpec {
            value_capacity: self.value_capacity.min(value_capacity),
            ..self
        }
    }

    /// `value` through PySpark's result converter. Each returns `None` unchanged, and the
    /// `StringType` and `BinaryType` converters return exact strs and bytes unchanged, so it is not
    /// called for them.
    fn convert<'py>(
        &self,
        py: Python<'py>,
        value: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.converter {
            Some(converter) if !(value.is_none() || self.kind.converts_unchanged(&value)) => {
                converter.bind(py).call1((value,))
            }
            _ => Ok(value),
        }
    }

    /// The `except pa.lib.ArrowInvalid` branch of PySpark's output step:
    /// `pa.array(values).cast(target_type=T, safe=safecheck)`.
    fn retry<'py>(
        &self,
        py: Python<'py>,
        values: Bound<'py, PyList>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "target_type"), self.arrow_type.bind(py))?;
        kwargs.set_item(intern!(py, "safe"), self.safecheck)?;
        let inferred = self.pyarrow_array.bind(py).call1((values,))?;
        inferred.call_method(intern!(py, "cast"), (), Some(&kwargs))
    }
}

/// The result types this UDF policy builds natively. Date results use the caller's Python path;
/// they are not part of this native result policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputKind {
    Int8,
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Boolean,
    Utf8,
    LargeUtf8,
    Binary,
    LargeBinary,
    Variant,
}

impl OutputKind {
    fn of(data_type: &DataType) -> Option<OutputKind> {
        Some(match data_type {
            DataType::Int8 => OutputKind::Int8,
            DataType::Int16 => OutputKind::Int16,
            DataType::Int32 => OutputKind::Int32,
            DataType::Int64 => OutputKind::Int64,
            DataType::Float32 => OutputKind::Float32,
            DataType::Float64 => OutputKind::Float64,
            DataType::Boolean => OutputKind::Boolean,
            DataType::Utf8 => OutputKind::Utf8,
            DataType::LargeUtf8 => OutputKind::LargeUtf8,
            DataType::Binary => OutputKind::Binary,
            DataType::LargeBinary => OutputKind::LargeBinary,
            DataType::Struct(fields) if *fields == variant_fields(DataType::Binary) => {
                OutputKind::Variant
            }
            _ => return None,
        })
    }

    fn is_string(self) -> bool {
        matches!(self, OutputKind::Utf8 | OutputKind::LargeUtf8)
    }

    /// The Spark type whose PySpark converter results of this kind may go through.
    fn converter_type(self) -> Option<&'static str> {
        match self {
            OutputKind::Utf8 | OutputKind::LargeUtf8 => Some("StringType"),
            OutputKind::Binary | OutputKind::LargeBinary => Some("BinaryType"),
            OutputKind::Variant => Some("VariantType"),
            _ => None,
        }
    }

    /// Whether this kind's converter returns `value` unchanged.
    fn converts_unchanged(self, value: &Bound<'_, PyAny>) -> bool {
        match self {
            OutputKind::Utf8 | OutputKind::LargeUtf8 => value.is_exact_instance_of::<PyString>(),
            OutputKind::Binary | OutputKind::LargeBinary => value.is_exact_instance_of::<PyBytes>(),
            _ => false,
        }
    }
}

enum Builder {
    Int8(Int8Builder),
    Int16(Int16Builder),
    Int32(Int32Builder),
    Int64(Int64Builder),
    /// Each result as the float64 PySpark's retry has to be given, which `finish` narrows as
    /// PyArrow does, with the rows that held an exact `int`.
    Float32(Float64Builder, BooleanBufferBuilder),
    /// With the rows that held an exact `int`, which PySpark's retry has to be given back as one.
    Float64(Float64Builder, BooleanBufferBuilder),
    Boolean(BooleanBuilder),
    Utf8(StringBuilder),
    LargeUtf8(LargeStringBuilder),
    Binary(BinaryBuilder),
    LargeBinary(LargeBinaryBuilder),
    /// The variants' two parts and which rows hold one. It takes no result when the UDF returns
    /// it, as a VariantVal can change afterwards, only every result once the UDF has run on
    /// every row: see [`Builder::append_variants`].
    Variant(BinaryBuilder, BinaryBuilder, NullBufferBuilder),
}

/// What a [`Builder`] appended.
struct Appended {
    /// As an array of the spec's type.
    array: ArrayRef,
    /// The same rows as the objects PySpark's retry is given: for float32, the float64 each was,
    /// and for a string column, what the converter made of a built-in.
    returned: ArrayRef,
    /// For floats, the rows that held an exact `int`.
    int_rows: Option<BooleanBuffer>,
}

impl Builder {
    fn new(kind: OutputKind, rows: usize) -> Builder {
        match kind {
            OutputKind::Int8 => Builder::Int8(Int8Builder::with_capacity(rows)),
            OutputKind::Int16 => Builder::Int16(Int16Builder::with_capacity(rows)),
            OutputKind::Int32 => Builder::Int32(Int32Builder::with_capacity(rows)),
            OutputKind::Int64 => Builder::Int64(Int64Builder::with_capacity(rows)),
            OutputKind::Float32 => Builder::Float32(
                Float64Builder::with_capacity(rows),
                BooleanBufferBuilder::new(rows),
            ),
            OutputKind::Float64 => Builder::Float64(
                Float64Builder::with_capacity(rows),
                BooleanBufferBuilder::new(rows),
            ),
            OutputKind::Boolean => Builder::Boolean(BooleanBuilder::with_capacity(rows)),
            OutputKind::Utf8 => Builder::Utf8(StringBuilder::with_capacity(rows, rows * 8)),
            OutputKind::LargeUtf8 => {
                Builder::LargeUtf8(LargeStringBuilder::with_capacity(rows, rows * 8))
            }
            OutputKind::Binary => Builder::Binary(BinaryBuilder::with_capacity(rows, rows * 8)),
            OutputKind::LargeBinary => {
                Builder::LargeBinary(LargeBinaryBuilder::with_capacity(rows, rows * 8))
            }
            OutputKind::Variant => Builder::Variant(
                BinaryBuilder::with_capacity(rows, rows * 8),
                BinaryBuilder::with_capacity(rows, rows * 8),
                NullBufferBuilder::new(rows),
            ),
        }
    }

    /// Appends `value` if it is `None` or an exact `int`, `float`, `bool`, `str` or `bytes` that
    /// PyArrow is known to convert to this type unchanged; appends nothing and returns false
    /// otherwise, or if a `Utf8` or `Binary` builder would then hold more than `capacity` bytes. A
    /// variant builder takes no value, not even `None`: see [`Builder::append_variants`].
    fn append_exact(&mut self, value: &Bound<'_, PyAny>, capacity: usize) -> bool {
        if value.is_none() {
            match self {
                Builder::Int8(builder) => builder.append_null(),
                Builder::Int16(builder) => builder.append_null(),
                Builder::Int32(builder) => builder.append_null(),
                Builder::Int64(builder) => builder.append_null(),
                Builder::Float32(builder, ints) | Builder::Float64(builder, ints) => {
                    builder.append_null();
                    ints.append(false);
                }
                Builder::Boolean(builder) => builder.append_null(),
                Builder::Utf8(builder) => builder.append_null(),
                Builder::LargeUtf8(builder) => builder.append_null(),
                Builder::Binary(builder) => builder.append_null(),
                Builder::LargeBinary(builder) => builder.append_null(),
                Builder::Variant(..) => return false,
            }
            return true;
        }
        match self {
            Builder::Int8(builder) => append_int(builder, value),
            Builder::Int16(builder) => append_int(builder, value),
            Builder::Int32(builder) => append_int(builder, value),
            Builder::Int64(builder) => append_int(builder, value),
            Builder::Float32(builder, ints) => {
                append_float(builder, ints, value, MAX_EXACT_F32_INT)
            }
            Builder::Float64(builder, ints) => {
                append_float(builder, ints, value, MAX_EXACT_F64_INT)
            }
            Builder::Boolean(builder) => {
                if let Some(value) = exact_bool(value) {
                    builder.append_value(value);
                    true
                } else {
                    false
                }
            }
            Builder::Utf8(builder) => match exact_str(value) {
                Some(text) if builder.values_slice().len() + text.len() <= capacity => {
                    builder.append_value(text);
                    true
                }
                _ => false,
            },
            Builder::LargeUtf8(builder) => match exact_str(value) {
                Some(text) => {
                    builder.append_value(text);
                    true
                }
                None => false,
            },
            Builder::Binary(builder) => match exact_bytes(value) {
                Some(bytes) if builder.values_slice().len() + bytes.len() <= capacity => {
                    builder.append_value(bytes);
                    true
                }
                _ => false,
            },
            Builder::LargeBinary(builder) => match exact_bytes(value) {
                Some(bytes) => {
                    builder.append_value(bytes);
                    true
                }
                None => false,
            },
            Builder::Variant(..) => false,
        }
    }

    /// Appends `values`, all of a variant column's results, given once the UDF has run on every
    /// row, if each is `None` or a VariantVal of exactly `class` whose two parts are exact
    /// `bytes`: PySpark's converter then gives PyArrow those two, which it stores unchanged.
    /// Appends nothing and returns false otherwise, if either part's values would take more than
    /// `capacity` bytes, and for any other column.
    fn append_variants(
        &mut self,
        values: &Bound<'_, PyList>,
        class: &Bound<'_, PyAny>,
        capacity: usize,
    ) -> bool {
        let Builder::Variant(value_parts, metadata_parts, nulls) = self else {
            return false;
        };
        let py = values.py();
        let mut parts = Vec::with_capacity(values.len());
        for value in values.iter() {
            if value.is_none() {
                parts.push(None);
                continue;
            }
            if !value.get_type().is(class) {
                return false;
            }
            // In the order PySpark's converter reads them.
            let part = value
                .getattr(intern!(py, "value"))
                .ok()
                .and_then(into_exact_bytes);
            let metadata = value
                .getattr(intern!(py, "metadata"))
                .ok()
                .and_then(into_exact_bytes);
            let (Some(part), Some(metadata)) = (part, metadata) else {
                return false;
            };
            parts.push(Some((part, metadata)));
        }
        let (value_bytes, metadata_bytes) = parts.iter().flatten().fold(
            (0, 0),
            |(value_bytes, metadata_bytes), (part, metadata)| {
                (
                    value_bytes + part.as_bytes().len(),
                    metadata_bytes + metadata.as_bytes().len(),
                )
            },
        );
        if value_bytes > capacity || metadata_bytes > capacity {
            return false;
        }
        for part in &parts {
            match part {
                // PyArrow gives a null variant's two parts empty values, not nulls.
                None => {
                    value_parts.append_value(b"");
                    metadata_parts.append_value(b"");
                    nulls.append_null();
                }
                Some((part, metadata)) => {
                    value_parts.append_value(part.as_bytes());
                    metadata_parts.append_value(metadata.as_bytes());
                    nulls.append_non_null();
                }
            }
        }
        true
    }

    /// The bytes of values a `Utf8` or `Binary` builder holds; `None` for every other builder.
    fn value_bytes(&self) -> Option<usize> {
        match self {
            Builder::Utf8(builder) => Some(builder.values_slice().len()),
            Builder::Binary(builder) => Some(builder.values_slice().len()),
            _ => None,
        }
    }

    /// The values so far.
    fn finish(&mut self) -> PyResult<Appended> {
        fn as_returned(array: ArrayRef) -> Appended {
            Appended {
                returned: array.clone(),
                array,
                int_rows: None,
            }
        }
        Ok(match self {
            Builder::Int8(builder) => as_returned(Arc::new(builder.finish())),
            Builder::Int16(builder) => as_returned(Arc::new(builder.finish())),
            Builder::Int32(builder) => as_returned(Arc::new(builder.finish())),
            Builder::Int64(builder) => as_returned(Arc::new(builder.finish())),
            Builder::Float32(builder, ints) => {
                let returned = builder.finish();
                // As PyArrow's static_cast<float> does.
                let narrowed: Float32Array = returned.unary(|value| value as f32);
                Appended {
                    array: Arc::new(narrowed),
                    returned: Arc::new(returned),
                    int_rows: Some(ints.finish()),
                }
            }
            Builder::Float64(builder, ints) => Appended {
                int_rows: Some(ints.finish()),
                ..as_returned(Arc::new(builder.finish()))
            },
            Builder::Boolean(builder) => as_returned(Arc::new(builder.finish())),
            Builder::Utf8(builder) => as_returned(Arc::new(builder.finish())),
            Builder::LargeUtf8(builder) => as_returned(Arc::new(builder.finish())),
            Builder::Binary(builder) => as_returned(Arc::new(builder.finish())),
            Builder::LargeBinary(builder) => as_returned(Arc::new(builder.finish())),
            Builder::Variant(value_parts, metadata_parts, nulls) => as_returned(Arc::new(
                StructArray::try_new(
                    variant_fields(DataType::Binary),
                    vec![
                        Arc::new(value_parts.finish()),
                        Arc::new(metadata_parts.finish()),
                    ],
                    nulls.finish(),
                )
                .map_err(arrow_error)?,
            )),
        })
    }
}

/// Appends an exact `float`, or an exact `int` of at most `max_int` in magnitude, which PyArrow
/// converts without rounding.
fn append_float(
    builder: &mut Float64Builder,
    ints: &mut BooleanBufferBuilder,
    value: &Bound<'_, PyAny>,
    max_int: u64,
) -> bool {
    if let Some(value) = exact_f64(value) {
        builder.append_value(value);
        ints.append(false);
        true
    } else if let Some(int) = exact_i64(value).filter(|int| int.unsigned_abs() <= max_int) {
        builder.append_value(int as f64);
        ints.append(true);
        true
    } else {
        false
    }
}

fn append_int<T>(builder: &mut PrimitiveBuilder<T>, value: &Bound<'_, PyAny>) -> bool
where
    T: ArrowPrimitiveType,
    T::Native: TryFrom<i64>,
{
    match exact_i64(value).and_then(|int| T::Native::try_from(int).ok()) {
        Some(int) => {
            builder.append_value(int);
            true
        }
        None => false,
    }
}

/// Whether `value` is `None` or an exact `int`, `float`, `bool` or `str`, none of which can change
/// after the UDF returns it.
fn is_builtin(value: &Bound<'_, PyAny>) -> bool {
    value.is_none()
        || value.is_exact_instance_of::<PyInt>()
        || value.is_exact_instance_of::<PyFloat>()
        || value.is_exact_instance_of::<PyBool>()
        || value.is_exact_instance_of::<PyString>()
}

/// One UDF's results for one batch, taken through PySpark's steps in order: every result through
/// the converter, then all of them through `pa.array(values, type=T)`, retried as
/// `pa.array(values).cast(T, safe=safecheck)` on `ArrowInvalid`.
///
/// PySpark runs the UDF on every row of the batch before it converts any result, so a result can
/// change after the UDF returns it, and a UDF's error on a later row wins over a result that
/// cannot be converted. [`ResultColumn::push`] therefore converts only built-ins, and the caller
/// must run the UDF on every row, giving each result to `push` or [`ResultColumn::extend`] in row
/// order, before it calls [`ResultColumn::finish`].
pub struct ResultColumn<'s> {
    spec: &'s ResultSpec,
    builder: Builder,
    state: State,
}

enum State {
    Building,
    /// The results from the first one the builder could not take, unconverted, in row order:
    /// those in `listed`, then those in `pushed`. `listed` has the held part of each list given
    /// to `extend` and the results pushed before the last such call; `pushed` has the results
    /// pushed after it.
    Holding {
        listed: Option<Py<PyList>>,
        pushed: Vec<Py<PyAny>>,
    },
    /// The converter raised, which PySpark does before it gives PyArrow any value, or a result
    /// could not be held.
    Failed(PyErr),
}

/// A finished [`ResultColumn`].
pub struct Finished<'py> {
    /// The results as a `pa.Array` of the spec's type, or past 2 GiB the `pa.ChunkedArray`
    /// PyArrow makes of them.
    pub array: Bound<'py, PyAny>,
    /// How many results the first `pa.array` call was given.
    pub pyarrow_values: usize,
    /// Whether PySpark's retry built the array.
    pub retried: bool,
}

impl<'s> ResultColumn<'s> {
    /// A column for a batch of `rows` rows.
    pub fn new(spec: &'s ResultSpec, rows: usize) -> ResultColumn<'s> {
        ResultColumn {
            spec,
            builder: Builder::new(spec.kind, rows),
            state: State::Building,
        }
    }

    /// Adds the next row's result. A built-in the builder takes is converted and appended now, as
    /// it cannot change; from the first result that is not, every result is held for
    /// [`ResultColumn::finish`]. Never raises: a converter error is raised by `finish`.
    pub fn push<'py>(&mut self, py: Python<'py>, value: Bound<'py, PyAny>) {
        match &mut self.state {
            State::Building => {}
            State::Holding { pushed, .. } => return pushed.push(value.unbind()),
            State::Failed(_) => return,
        }
        if self.builder.append_exact(&value, self.spec.value_capacity)
            || self.append_converted(py, &value)
        {
            return;
        }
        self.state = State::Holding {
            listed: None,
            pushed: vec![value.unbind()],
        };
    }

    /// Adds the next rows' results, as pushing each of `values` in turn would, for a caller that
    /// collected them into a list. From the first result the builder cannot take, the column keeps
    /// a copy of the rest of `values` as one slice, so the caller may change `values` afterwards.
    pub fn extend<'py>(&mut self, py: Python<'py>, values: &Bound<'py, PyList>) {
        if let State::Building = self.state {
            if let Err(error) = self.extend_building(py, values) {
                self.state = State::Failed(error);
            }
        } else if let State::Holding { listed, pushed } = &mut self.state {
            if let Err(error) = extend_held(py, listed, pushed, values) {
                self.state = State::Failed(error);
            }
        }
    }

    fn extend_building<'py>(
        &mut self,
        py: Python<'py>,
        values: &Bound<'py, PyList>,
    ) -> PyResult<()> {
        let mut appended = 0;
        // The iterator protocol reads a list with one call per item, and the abi3 list API with
        // three.
        for value in values.try_iter()? {
            let value = value?;
            if !(self.builder.append_exact(&value, self.spec.value_capacity)
                || self.append_converted(py, &value))
            {
                break;
            }
            if let State::Failed(_) = self.state {
                return Ok(());
            }
            appended += 1;
        }
        if appended < values.len() {
            self.state = State::Holding {
                listed: Some(values.get_slice(appended, values.len()).unbind()),
                pushed: Vec::new(),
            };
        }
        Ok(())
    }

    /// Appends `value`, which the builder did not take, if it is a float, a bool or an int that
    /// fits in 64 bits, which the `StringType` converter turns into a str it takes. Returns false
    /// if `value` has to be held instead. A converter error fails the column.
    fn append_converted<'py>(&mut self, py: Python<'py>, value: &Bound<'py, PyAny>) -> bool {
        // A built-in cannot change after the UDF returns it, so converting it now matches PySpark,
        // which converts once the UDF has run on every row. For a string column the builder takes
        // only None and exact strs, which the StringType converter returns unchanged.
        if !self.spec.kind.is_string() || self.spec.converter.is_none() || !is_builtin(value) {
            return false;
        }
        // A later row may call sys.set_int_max_str_digits, which decides whether str() takes an
        // int of more than 640 digits.
        // Keep large integers until every UDF call has completed so that limit changes are seen.
        if value.is_exact_instance_of::<PyInt>() && exact_i64(value).is_none() {
            return false;
        }
        match self.spec.convert(py, value.clone()) {
            Ok(converted) => self
                .builder
                .append_exact(&converted, self.spec.value_capacity),
            Err(error) => {
                self.state = State::Failed(error);
                true
            }
        }
    }

    /// The results as a `pa.Array`. Held results go through the converter and one
    /// `pa.array(held, type=T)` call, and are joined to the appended ones; if PyArrow rejects one
    /// with `ArrowInvalid`, PySpark's retry builds the whole batch instead. If the two would not
    /// fit in one array, every result goes into that call, as in PySpark, where past 2 GiB it
    /// returns a `pa.ChunkedArray` that fails the batch once every UDF has run. A converter error
    /// is raised first, as PySpark converts every result before PyArrow sees one. A variant
    /// column's results are all held, and are appended now if the builder takes every one.
    pub fn finish<'py>(mut self, py: Python<'py>) -> PyResult<Finished<'py>> {
        let mut held = match self.state {
            State::Building => None,
            State::Holding { listed, pushed } => Some(match listed {
                None => PyList::new(py, pushed)?,
                Some(listed) => {
                    let listed = listed.into_bound(py);
                    for value in pushed {
                        listed.append(value)?;
                    }
                    listed
                }
            }),
            State::Failed(error) => return Err(error),
        };
        if let (Some(values), Some(class)) = (&held, &self.spec.variant) {
            if self
                .builder
                .append_variants(values, class.bind(py), self.spec.value_capacity)
            {
                held = None;
            }
        }
        let head_bytes = self.builder.value_bytes();
        let appended = self.builder.finish()?;
        let array_type = self.spec.array_type.bind(py);
        let Some(held) = held else {
            let array = export_array(array_type, appended.array)?;
            return Ok(Finished {
                array,
                pyarrow_values: 0,
                retried: false,
            });
        };
        let converted = if self.spec.converter.is_some() {
            let converted = held
                .iter()
                .map(|value| self.spec.convert(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, converted)?
        } else {
            held
        };
        let pyarrow_array = self.spec.pyarrow_array.bind(py);
        let arrow_invalid = self.spec.arrow_invalid.bind(py);
        if appended.array.is_empty()
            || head_bytes.is_some_and(|bytes| !fits(bytes, &converted, self.spec.value_capacity))
        {
            let values = appended_values(py, &appended)?;
            values.call_method1(intern!(py, "extend"), (&converted,))?;
            let pyarrow_values = values.len();
            return match pyarrow_array.call1((&values, self.spec.arrow_type.bind(py))) {
                Ok(array) => Ok(Finished {
                    array,
                    pyarrow_values,
                    retried: false,
                }),
                Err(error) if error.is_instance(py, arrow_invalid) => Ok(Finished {
                    array: self.spec.retry(py, values)?,
                    pyarrow_values,
                    retried: true,
                }),
                Err(error) => Err(error),
            };
        }
        let pyarrow_values = converted.len();
        match pyarrow_array.call1((&converted, self.spec.arrow_type.bind(py))) {
            Ok(tail) => {
                let head = export_array(array_type, appended.array)?;
                let array = self
                    .spec
                    .concat_arrays
                    .bind(py)
                    .call1((PyList::new(py, [head, tail])?,))?;
                Ok(Finished {
                    array,
                    pyarrow_values,
                    retried: false,
                })
            }
            Err(error) if error.is_instance(py, arrow_invalid) => {
                let values = appended_values(py, &appended)?;
                for value in converted.iter() {
                    values.append(value)?;
                }
                Ok(Finished {
                    array: self.spec.retry(py, values)?,
                    pyarrow_values,
                    retried: true,
                })
            }
            Err(error) => Err(error),
        }
    }
}

/// Whether `values`, results through the converter, fit after `used` of `capacity` bytes in a
/// string or binary array. A value that is not `None`, an exact `str` or exact `bytes` is taken
/// not to.
fn fits(mut used: usize, values: &Bound<'_, PyList>, capacity: usize) -> bool {
    for value in values.iter() {
        used += if value.is_none() {
            0
        } else if let Some(text) = exact_str(&value) {
            text.len()
        } else if let Some(bytes) = exact_bytes(&value) {
            bytes.len()
        } else {
            return false;
        };
        if used > capacity {
            return false;
        }
    }
    true
}

/// Appends `values` to the held results, after the pushed ones.
fn extend_held<'py>(
    py: Python<'py>,
    listed: &mut Option<Py<PyList>>,
    pushed: &mut Vec<Py<PyAny>>,
    values: &Bound<'py, PyList>,
) -> PyResult<()> {
    let list = listed
        .get_or_insert_with(|| PyList::empty(py).unbind())
        .bind(py);
    for value in pushed.drain(..) {
        list.append(value)?;
    }
    list.call_method1(intern!(py, "extend"), (values,))?;
    Ok(())
}

/// The appended results as the objects PySpark's retry is given, with the float rows that held
/// an exact `int` given back as one.
fn appended_values<'py>(py: Python<'py>, appended: &Appended) -> PyResult<Bound<'py, PyList>> {
    let values = PyList::empty(py);
    let returned = &appended.returned;
    // A variant column appends nothing before it holds every result.
    if returned.is_empty() {
        return Ok(values);
    }
    let column = Column::new(py, returned, &Read::Plain)?.ok_or_else(|| {
        PyRuntimeError::new_err(format!("no reader for {}", returned.data_type()))
    })?;
    for row in 0..returned.len() {
        if appended
            .int_rows
            .as_ref()
            .is_some_and(|ints| ints.value(row))
        {
            values.append(returned.as_primitive::<Float64Type>().value(row) as i64)?;
        } else {
            values.append(column.value(py, row, || {
                PyRuntimeError::new_err(format!("appended result {row} cannot be read back"))
            })?)?;
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{Field, Fields, TimeUnit};

    use super::*;

    #[test]
    fn builds_a_variant_only_as_pyspark_types_it() {
        assert_eq!(
            OutputKind::of(&DataType::Struct(variant_fields(DataType::Binary))),
            Some(OutputKind::Variant)
        );
        let untagged = Fields::from(vec![
            Field::new("value", DataType::Binary, false),
            Field::new("metadata", DataType::Binary, false),
        ]);
        for data_type in [
            DataType::Struct(untagged),
            DataType::Decimal128(10, 2),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Date64,
        ] {
            assert_eq!(OutputKind::of(&data_type), None, "{data_type}");
        }
    }
}
