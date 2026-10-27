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

use arrow::array::{
    Array, ArrayRef, AsArray, BinaryArray, BooleanArray, Date32Array, Float32Array, Float64Array,
    GenericBinaryArray, GenericStringArray, Int16Array, Int32Array, Int64Array, Int8Array,
    LargeBinaryArray, LargeStringArray, OffsetSizeTrait, StringArray, StructArray,
};
use arrow::datatypes::DataType;
use pyo3::exceptions::{PyIndexError, PyRuntimeError, PyUnicodeDecodeError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDate, PyDict};
use pyo3::{intern, PyTypeInfo};

use crate::scalar::{date_into_python, is_date_in_range, utf8_into_python, Scalar};
use crate::udf::{import_batch, is_spark_type, variant_fields};

/// Accepts an input column of `data_type`, the Arrow type of `spark_type`, given PySpark's input
/// converter for it or `None` and the backend package's `sql.types` module; or returns the reason
/// the native UDF input converter does not support it.
pub fn input_spec(
    data_type: &DataType,
    spark_type: &Bound<'_, PyAny>,
    converter: &Bound<'_, PyAny>,
    sql_types: &Bound<'_, PyAny>,
) -> PyResult<Result<InputSpec, String>> {
    let py = spark_type.py();
    let read = if converter.is_none() {
        if !Column::reads(data_type) {
            return Ok(Err(format!(
                "{data_type} is not supported by the native UDF input converter"
            )));
        }
        Read::Plain
    } else if matches!(data_type, DataType::Binary | DataType::LargeBinary)
        && is_spark_type(spark_type, sql_types, "BinaryType")?
    {
        // PySpark's BinaryType converter returns the bytes PyArrow gives, or a bytearray copy
        // when spark.sql.execution.pyspark.binaryAsBytes is off.
        let sample = converter.call1((PyBytes::new(py, b""),))?;
        if sample.is_exact_instance_of::<PyBytes>() {
            Read::Plain
        } else if sample.is_exact_instance_of::<PyByteArray>() {
            Read::ByteArray
        } else {
            return Ok(Err(format!(
                "{spark_type}'s converter gives {}",
                sample.get_type()
            )));
        }
    } else if is_variant_struct(data_type) && is_spark_type(spark_type, sql_types, "VariantType")? {
        // The converter can return a VariantVal whose class differs from the backend module's.
        // Take the class from its output to preserve that identity.
        let sample = PyDict::new(py);
        sample.set_item("value", PyBytes::new(py, b""))?;
        sample.set_item("metadata", PyBytes::new(py, b""))?;
        Read::Variant {
            class: converter.call1((sample,))?.get_type().into_any().unbind(),
        }
    } else {
        return Ok(Err(format!("{spark_type} needs PySpark's converter")));
    };
    Ok(Ok(InputSpec {
        data_type: data_type.clone(),
        read,
        converter: converter.clone().unbind(),
    }))
}

/// Whether `data_type` is the struct Spark gives `VariantType`, with parts of either width.
fn is_variant_struct(data_type: &DataType) -> bool {
    [DataType::Binary, DataType::LargeBinary]
        .into_iter()
        .any(|part| *data_type == DataType::Struct(variant_fields(part)))
}

/// How [`InputColumns`] reads a column [`input_spec`] accepted.
pub struct InputSpec {
    data_type: DataType,
    read: Read,
    /// PySpark's input converter for the column, or `None`.
    converter: Py<PyAny>,
}

impl InputSpec {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    pub fn clone_ref(&self, py: Python<'_>) -> InputSpec {
        InputSpec {
            data_type: self.data_type.clone(),
            read: match &self.read {
                Read::Plain => Read::Plain,
                Read::ByteArray => Read::ByteArray,
                Read::Variant { class } => Read::Variant {
                    class: class.clone_ref(py),
                },
            },
            converter: self.converter.clone_ref(py),
        }
    }
}

/// What a column's values become, as PySpark makes them.
pub(crate) enum Read {
    /// What PyArrow's `to_pylist` gives, which PySpark keeps.
    Plain,
    /// A `bytearray` copy of each binary value.
    ByteArray,
    /// A VariantVal of `class` for each struct of the variant's two binaries.
    Variant { class: Py<PyAny> },
}

/// A batch's input columns, read one cell at a time.
pub struct InputColumns {
    batch: Py<PyAny>,
    columns: Vec<Column>,
    converters: Vec<Py<PyAny>>,
    convert_input: Py<PyAny>,
    rows: usize,
}

impl InputColumns {
    /// Imports `batch`, a `pa.RecordBatch`, without copying it. Its columns must have the types
    /// of `specs`. `convert_input` is PySpark's input step, `_convert_arrow_udf_input`, which
    /// [`InputColumns::value`] runs to raise PySpark's error.
    pub fn import(
        batch: &Bound<'_, PyAny>,
        specs: &[InputSpec],
        convert_input: &Bound<'_, PyAny>,
    ) -> PyResult<InputColumns> {
        let py = batch.py();
        let input = import_batch(batch)?;
        let actual: Vec<&DataType> = input.columns().iter().map(|c| c.data_type()).collect();
        if actual.len() != specs.len()
            || actual
                .iter()
                .zip(specs)
                .any(|(actual, spec)| **actual != spec.data_type)
        {
            let expected: Vec<&DataType> = specs.iter().map(InputSpec::data_type).collect();
            return Err(PyRuntimeError::new_err(format!(
                "the batch has types {actual:?}, but {expected:?} were expected"
            )));
        }
        let columns = input
            .columns()
            .iter()
            .zip(specs)
            .map(|(array, spec)| {
                Column::new(py, array, &spec.read)?.ok_or_else(|| {
                    PyRuntimeError::new_err(format!("no reader for {}", array.data_type()))
                })
            })
            .collect::<PyResult<Vec<Column>>>()?;
        Ok(InputColumns {
            batch: batch.clone().unbind(),
            columns,
            converters: specs
                .iter()
                .map(|spec| spec.converter.clone_ref(py))
                .collect(),
            convert_input: convert_input.clone().unbind(),
            rows: input.len(),
        })
    }

    pub fn num_rows(&self) -> usize {
        self.rows
    }

    /// What PySpark gives a UDF for `row` of `column`, as a new object. `column` indexes the
    /// imported types, and `row` is below [`InputColumns::num_rows`].
    ///
    /// A cell PySpark fails to convert, such as a string that is not UTF-8, raises what PySpark's
    /// input step raises for its column. PySpark converts every input column, in order, before it
    /// calls a UDF, so to fail as it does, read every cell, column by column in order, before
    /// calling any UDF.
    pub fn value<'py>(
        &self,
        py: Python<'py>,
        column: usize,
        row: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let Some(input) = self.columns.get(column) else {
            return Err(PyIndexError::new_err(format!(
                "column index {column} is out of bounds for {} columns",
                self.columns.len()
            )));
        };
        if row >= self.rows {
            return Err(PyIndexError::new_err(format!(
                "row index {row} is out of bounds for {} rows",
                self.rows
            )));
        }
        input.value(py, row, || self.pyspark_error(py, column))
    }

    /// What PySpark's input step raises for `index`, a column with a cell it fails to convert.
    fn pyspark_error(&self, py: Python<'_>, index: usize) -> PyErr {
        let converted = (|| -> PyResult<_> {
            let column = self
                .batch
                .bind(py)
                .call_method1(intern!(py, "column"), (index,))?;
            self.convert_input
                .bind(py)
                .call1((column, self.converters[index].bind(py)))
        })();
        match converted {
            Err(error) => error,
            Ok(_) => PyRuntimeError::new_err(format!(
                "PySpark converted input {index}, which the native converter cannot read"
            )),
        }
    }
}

/// A column, read as PySpark reads it.
pub(crate) enum Column {
    Int8(Int8Array),
    Int16(Int16Array),
    Int32(Int32Array),
    Int64(Int64Array),
    Float32(Float32Array),
    Float64(Float64Array),
    Boolean(BooleanArray),
    Utf8(StringArray),
    LargeUtf8(LargeStringArray),
    /// With `datetime.date.fromordinal`.
    Date32(Date32Array, Py<PyAny>),
    /// With whether each value is read as a `bytearray`.
    Binary(BinaryArray, bool),
    LargeBinary(LargeBinaryArray, bool),
    Variant(Box<VariantColumn<i32>>),
    LargeVariant(Box<VariantColumn<i64>>),
}

/// A variant column, with its two parts and the class of VariantVal its values are made of.
pub(crate) struct VariantColumn<O: OffsetSizeTrait> {
    array: StructArray,
    value: GenericBinaryArray<O>,
    metadata: GenericBinaryArray<O>,
    class: Py<PyAny>,
}

impl<O: OffsetSizeTrait> VariantColumn<O> {
    fn new(py: Python<'_>, array: &ArrayRef, class: &Py<PyAny>) -> Self {
        let array = array.as_struct().clone();
        VariantColumn {
            value: array.column(0).as_binary().clone(),
            metadata: array.column(1).as_binary().clone(),
            array,
            class: class.clone_ref(py),
        }
    }

    fn value<'py>(
        &self,
        py: Python<'py>,
        row: usize,
        unconvertible: impl FnOnce() -> PyErr,
    ) -> PyResult<Bound<'py, PyAny>> {
        if self.array.is_null(row) {
            return Ok(py.None().into_bound(py));
        }
        if self.value.is_null(row) || self.metadata.is_null(row) {
            return Err(unconvertible());
        }
        self.class.bind(py).call1((
            PyBytes::new(py, self.value.value(row)),
            PyBytes::new(py, self.metadata.value(row)),
        ))
    }
}

impl Column {
    /// The types this native reader supports without an additional PySpark converter.
    fn reads(data_type: &DataType) -> bool {
        matches!(
            data_type,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Float32
                | DataType::Float64
                | DataType::Boolean
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Date32
        )
    }

    pub(crate) fn new(py: Python<'_>, array: &ArrayRef, read: &Read) -> PyResult<Option<Column>> {
        Ok(Some(match (array.data_type(), read) {
            (DataType::Int8, Read::Plain) => Column::Int8(array.as_primitive().clone()),
            (DataType::Int16, Read::Plain) => Column::Int16(array.as_primitive().clone()),
            (DataType::Int32, Read::Plain) => Column::Int32(array.as_primitive().clone()),
            (DataType::Int64, Read::Plain) => Column::Int64(array.as_primitive().clone()),
            (DataType::Float32, Read::Plain) => Column::Float32(array.as_primitive().clone()),
            (DataType::Float64, Read::Plain) => Column::Float64(array.as_primitive().clone()),
            (DataType::Boolean, Read::Plain) => Column::Boolean(array.as_boolean().clone()),
            (DataType::Utf8, Read::Plain) => Column::Utf8(array.as_string().clone()),
            (DataType::LargeUtf8, Read::Plain) => Column::LargeUtf8(array.as_string().clone()),
            (DataType::Date32, Read::Plain) => Column::Date32(
                array.as_primitive().clone(),
                PyDate::type_object(py).getattr("fromordinal")?.unbind(),
            ),
            (DataType::Binary, Read::Plain | Read::ByteArray) => {
                Column::Binary(array.as_binary().clone(), matches!(read, Read::ByteArray))
            }
            (DataType::LargeBinary, Read::Plain | Read::ByteArray) => {
                Column::LargeBinary(array.as_binary().clone(), matches!(read, Read::ByteArray))
            }
            (DataType::Struct(fields), Read::Variant { class })
                if *fields == variant_fields(DataType::Binary) =>
            {
                Column::Variant(Box::new(VariantColumn::new(py, array, class)))
            }
            (DataType::Struct(fields), Read::Variant { class })
                if *fields == variant_fields(DataType::LargeBinary) =>
            {
                Column::LargeVariant(Box::new(VariantColumn::new(py, array, class)))
            }
            _ => return Ok(None),
        }))
    }

    /// A cell PySpark fails to convert raises `unconvertible()`: a string that is not UTF-8, a date
    /// outside `datetime.date`'s range, or a variant with a null part.
    pub(crate) fn value<'py>(
        &self,
        py: Python<'py>,
        row: usize,
        unconvertible: impl FnOnce() -> PyErr,
    ) -> PyResult<Bound<'py, PyAny>> {
        fn read<'a, 'py, A: Array>(
            py: Python<'py>,
            array: &A,
            row: usize,
            value: impl FnOnce() -> Scalar<'a>,
        ) -> PyResult<Bound<'py, PyAny>> {
            if array.is_null(row) {
                Scalar::Null.into_python(py)
            } else {
                value().into_python(py)
            }
        }
        /// Decodes the slot's bytes rather than reading a `&str`: arrow-rs imports strings
        /// through the C Data Interface without checking them, so CPython's decode is the check.
        fn string<'py, O: OffsetSizeTrait>(
            py: Python<'py>,
            array: &GenericStringArray<O>,
            row: usize,
            unconvertible: impl FnOnce() -> PyErr,
        ) -> PyResult<Bound<'py, PyAny>> {
            if array.is_null(row) {
                return Ok(py.None().into_bound(py));
            }
            let offsets = array.value_offsets();
            let bytes = &array.value_data()[offsets[row].as_usize()..offsets[row + 1].as_usize()];
            match utf8_into_python(py, bytes) {
                Ok(string) => Ok(string),
                Err(error) if error.is_instance_of::<PyUnicodeDecodeError>(py) => {
                    Err(unconvertible())
                }
                Err(error) => Err(error),
            }
        }
        fn binary<'py, O: OffsetSizeTrait>(
            py: Python<'py>,
            array: &GenericBinaryArray<O>,
            row: usize,
            bytearray: bool,
        ) -> PyResult<Bound<'py, PyAny>> {
            if array.is_null(row) {
                return Scalar::Null.into_python(py);
            }
            let bytes = array.value(row);
            if bytearray {
                Ok(PyByteArray::new(py, bytes).into_any())
            } else {
                Scalar::Binary(bytes).into_python(py)
            }
        }
        match self {
            Column::Int8(array) => {
                read(py, array, row, || Scalar::Int(i64::from(array.value(row))))
            }
            Column::Int16(array) => {
                read(py, array, row, || Scalar::Int(i64::from(array.value(row))))
            }
            Column::Int32(array) => {
                read(py, array, row, || Scalar::Int(i64::from(array.value(row))))
            }
            Column::Int64(array) => read(py, array, row, || Scalar::Int(array.value(row))),
            Column::Float32(array) => read(py, array, row, || {
                Scalar::Float(f64::from(array.value(row)))
            }),
            Column::Float64(array) => read(py, array, row, || Scalar::Float(array.value(row))),
            Column::Boolean(array) => read(py, array, row, || Scalar::Bool(array.value(row))),
            Column::Utf8(array) => string(py, array, row, unconvertible),
            Column::LargeUtf8(array) => string(py, array, row, unconvertible),
            Column::Date32(array, fromordinal) => {
                if array.is_null(row) {
                    return Ok(py.None().into_bound(py));
                }
                let days = array.value(row);
                if !is_date_in_range(days) {
                    return Err(unconvertible());
                }
                date_into_python(days, fromordinal.bind(py))
            }
            Column::Binary(array, bytearray) => binary(py, array, row, *bytearray),
            Column::LargeBinary(array, bytearray) => binary(py, array, row, *bytearray),
            Column::Variant(variant) => variant.value(py, row, unconvertible),
            Column::LargeVariant(variant) => variant.value(py, row, unconvertible),
        }
    }
}
