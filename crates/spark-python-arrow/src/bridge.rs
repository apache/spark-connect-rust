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

//! Share Arrow buffers with PyArrow through the Arrow C Data Interface.
//!
//! Imports accept real PyArrow objects, not arbitrary objects implementing an
//! export method. PyArrow remains responsible for the validity of its C data.
//! These functions perform limited structural checks, not full validation of all
//! offsets, dictionary indices, or payloads. Valid offsets and buffer bounds are
//! producer preconditions. String decoding errors remain observable at conversion.

use std::sync::Arc;

use arrow::array::{make_array, Array, ArrayData, ArrayRef, StructArray};
use arrow::datatypes::{DataType, Field, Schema, UnionFields};
use arrow::error::ArrowError;
use arrow::ffi::{from_ffi_and_data_type, FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::{RecordBatch, RecordBatchOptions};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

/// Import a PyArrow schema, preserving schema and field metadata.
pub fn import_schema(value: &Bound<'_, PyAny>) -> PyResult<Schema> {
    let class = require_pyarrow_class(value, "Schema")?;
    check_python_schema(value)?;
    let mut schema = FFI_ArrowSchema::empty();
    let address = &mut schema as *mut FFI_ArrowSchema as usize;
    class.call_method1("_export_to_c", (value, address))?;
    import_c_schema(&schema).map_err(arrow_error)
}

/// Import a PyArrow data type.
///
/// Root extension types and ordered dictionaries require a schema or record batch:
/// a bare Rust `DataType` has nowhere to retain these field properties.
pub fn import_data_type(value: &Bound<'_, PyAny>) -> PyResult<DataType> {
    let class = require_pyarrow_class(value, "DataType")?;
    reject_root_extension_type(value)?;
    reject_root_dictionary_order(value)?;
    check_python_data_type(value)?;
    let mut schema = FFI_ArrowSchema::empty();
    let address = &mut schema as *mut FFI_ArrowSchema as usize;
    class.call_method1("_export_to_c", (value, address))?;
    import_c_data_type(&schema).map_err(arrow_error)
}

/// Import a PyArrow array without copying its data buffers.
///
/// Root extensions and ordered dictionaries are rejected because an `ArrayRef`
/// cannot retain their field properties. Import a record batch to preserve them.
pub fn import_array(value: &Bound<'_, PyAny>) -> PyResult<ArrayRef> {
    let class = require_pyarrow_class(value, "Array")?;
    let data_type = value.getattr("type")?;
    reject_root_extension_type(&data_type)?;
    reject_root_dictionary_order(&data_type)?;
    check_python_data_type(&data_type)?;
    let (array, schema) = export_python_array(value, &class)?;
    let data_type = import_c_data_type(&schema).map_err(arrow_error)?;
    // SAFETY: the base PyArrow exporter, not an arbitrary Python method, has
    // initialized both descriptors and transferred ownership of valid C data.
    let data = unsafe { from_ffi_and_data_type(array, data_type) }.map_err(arrow_error)?;
    validate_structure(&data)?;
    Ok(make_array(data))
}

/// Import a PyArrow record batch, preserving metadata and zero-column rows.
pub fn import_batch(value: &Bound<'_, PyAny>) -> PyResult<RecordBatch> {
    let class = require_pyarrow_class(value, "RecordBatch")?;
    check_python_schema(&value.getattr("schema")?)?;
    let (array, ffi_schema) = export_python_array(value, &class)?;
    let schema = import_c_schema(&ffi_schema).map_err(arrow_error)?;
    let data_type = DataType::Struct(schema.fields().clone());
    // SAFETY: only the real PyArrow RecordBatch exporter writes the descriptors.
    let data = unsafe { from_ffi_and_data_type(array, data_type) }.map_err(arrow_error)?;
    validate_structure(&data)?;
    if !matches!(data.data_type(), DataType::Struct(_)) || data.null_count() != 0 {
        return Err(PyValueError::new_err(
            "A record batch must be a struct array without top-level nulls",
        ));
    }
    let array = make_array(data);
    let array = array
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| PyValueError::new_err("A record batch must be a struct array"))?;
    let options = RecordBatchOptions::new().with_row_count(Some(array.len()));
    // Unlike RecordBatch::from(StructArray), this validates non-nullable columns
    // and retains the schema-level metadata from the exported C schema.
    RecordBatch::try_new_with_options(Arc::new(schema), array.columns().to_vec(), &options)
        .map_err(arrow_error)
}

/// Export an Arrow array sharing its value and offset buffers.
///
/// Arrow may copy a sliced validity bitmap to align its offset with the data.
pub fn export_array(py: Python<'_>, array: ArrayRef) -> PyResult<Py<PyAny>> {
    let array_type = py.import("pyarrow")?.getattr("Array")?;
    Ok(export_array_with_type(&array_type, array)?.unbind())
}

pub(crate) fn export_array_with_type<'py>(
    array_type: &Bound<'py, PyAny>,
    array: ArrayRef,
) -> PyResult<Bound<'py, PyAny>> {
    check_c_data_type(array.data_type())?;
    let mut schema = export_c_data_type(array.data_type()).map_err(arrow_error)?;
    let schema_address = &mut schema as *mut FFI_ArrowSchema as usize;
    let mut array = FFI_ArrowArray::new(&array.to_data());
    let array_address = &mut array as *mut FFI_ArrowArray as usize;
    // Both descriptors are owned here until the importer consumes them. Their
    // release callbacks also cover importer errors before either transfer.
    array_type.call_method1("_import_from_c", (array_address, schema_address))
}

/// Export an Arrow schema, preserving schema and field metadata.
///
/// C schema names and timezone strings cannot contain NUL. This low-level API
/// returns an error for such schemas; table export provides an IPC fallback.
pub fn export_schema(py: Python<'_>, schema: &Schema) -> PyResult<Py<PyAny>> {
    check_c_schema(schema)?;
    let data_type = DataType::Struct(schema.fields().clone());
    let mut schema = export_c_data_type(&data_type)
        .and_then(|exported| exported.with_metadata(schema.metadata()))
        .map_err(arrow_error)?;
    let address = &mut schema as *mut FFI_ArrowSchema as usize;
    Ok(py
        .import("pyarrow")?
        .getattr("Schema")?
        .call_method1("_import_from_c", (address,))?
        .unbind())
}

/// Export an Arrow data type, rejecting strings not representable in C schemas.
pub fn export_data_type(py: Python<'_>, data_type: &DataType) -> PyResult<Py<PyAny>> {
    check_c_data_type(data_type)?;
    let mut schema = export_c_data_type(data_type).map_err(arrow_error)?;
    let address = &mut schema as *mut FFI_ArrowSchema as usize;
    Ok(py
        .import("pyarrow")?
        .getattr("DataType")?
        .call_method1("_import_from_c", (address,))?
        .unbind())
}

/// Build a PyArrow table, sharing data buffers for C-compatible schemas.
///
/// Batch boundaries, schema metadata, and zero-column row counts are preserved.
/// An empty batch vector produces an empty table with an empty schema.
/// Value and offset buffers are shared; sliced validity bitmaps may be realigned.
pub fn record_batches_to_pyarrow(py: Python<'_>, batches: Vec<RecordBatch>) -> PyResult<Py<PyAny>> {
    let schema = batches
        .first()
        .map(|batch| batch.schema())
        .unwrap_or_else(|| Arc::new(Schema::empty()));
    if batches.iter().any(|batch| {
        !Arc::ptr_eq(batch.schema_ref(), &schema) && !schemas_equal(batch.schema_ref(), &schema)
    }) {
        return Err(PyValueError::new_err(
            "Arrow record batches have inconsistent schemas",
        ));
    }
    let pa = py.import("pyarrow")?;
    if schema.fields().iter().any(|field| field_needs_ipc(field)) {
        // C schema strings cannot contain NUL. Use the already collected
        // batches for this rare compatibility path without repeating any RPC.
        let bytes = py
            .detach(|| -> arrow::error::Result<Vec<u8>> {
                let mut bytes = Vec::new();
                let mut writer = FileWriter::try_new(&mut bytes, &schema)?;
                for batch in &batches {
                    writer.write(batch)?;
                }
                writer.finish()?;
                drop(writer);
                Ok(bytes)
            })
            .map_err(arrow_error)?;
        return Ok(pa
            .getattr("ipc")?
            .call_method1("open_file", (PyBytes::new(py, &bytes),))?
            .call_method0("read_all")?
            .unbind());
    }

    // Import once before any arrays. This also lets older PyArrow versions
    // reject an unsupported schema without consuming any data descriptors.
    let py_schema = export_schema(py, &schema)?;
    let import_batch = pa.getattr("RecordBatch")?.getattr("_import_from_c")?;
    let py_batches = PyList::empty(py);
    for batch in batches {
        let data = StructArray::from(batch).to_data();
        let mut array = FFI_ArrowArray::new(&data);
        let address = &mut array as *mut FFI_ArrowArray as usize;
        // Import consumes the array and clears its release callback. If import
        // fails before consumption, Rust releases the remaining ownership.
        py_batches.append(import_batch.call1((address, py_schema.bind(py)))?)?;
    }
    Ok(pa
        .getattr("Table")?
        .call_method1("from_batches", (py_batches, py_schema.bind(py)))?
        .unbind())
}

pub(crate) fn arrow_error(error: ArrowError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

pub(crate) fn is_extension_type(data_type: &Bound<'_, PyAny>) -> PyResult<bool> {
    let class = data_type
        .py()
        .import("pyarrow.lib")?
        .getattr("BaseExtensionType")?;
    data_type.is_instance(&class)
}

fn reject_root_extension_type(data_type: &Bound<'_, PyAny>) -> PyResult<()> {
    if is_extension_type(data_type)? {
        return Err(PyTypeError::new_err(
            "Arrow extension types require a schema or record batch to preserve metadata",
        ));
    }
    Ok(())
}

fn reject_root_dictionary_order(data_type: &Bound<'_, PyAny>) -> PyResult<()> {
    if let Ok(ordered) = data_type.getattr("ordered") {
        if ordered.is_truthy()? {
            return Err(PyTypeError::new_err(
                "Ordered Arrow dictionaries require a schema or record batch to preserve ordering",
            ));
        }
    }
    Ok(())
}

// Arrow 58's Field conversion overwrites a map's type flags with its field flags.
// Rebuild the descriptor tree using public constructors, retaining both. Only
// descriptors are rebuilt: array buffers are not involved in schema conversion.
fn export_c_data_type(data_type: &DataType) -> Result<FFI_ArrowSchema, ArrowError> {
    fn has_sorted_map(data_type: &DataType) -> bool {
        matches!(data_type, DataType::Map(_, true))
            || child_fields(data_type)
                .iter()
                .any(|field| has_sorted_map(field.data_type()))
            || matches!(data_type, DataType::Dictionary(_, values) if has_sorted_map(values))
    }

    fn rebuild(
        source: &FFI_ArrowSchema,
        data_type: &DataType,
    ) -> Result<FFI_ArrowSchema, ArrowError> {
        let children = child_fields(data_type)
            .into_iter()
            .enumerate()
            .map(|(index, field)| rebuild(source.child(index), field.data_type()))
            .collect::<Result<Vec<_>, _>>()?;
        let dictionary = match (source.dictionary(), data_type) {
            (Some(schema), DataType::Dictionary(_, values)) => Some(rebuild(schema, values)?),
            _ => None,
        };
        let mut schema = FFI_ArrowSchema::try_new(source.format(), children, dictionary)?
            .with_metadata(source.metadata()?)?;
        if let Some(name) = source.name() {
            schema = schema.with_name(name)?;
        }
        schema.flags = source.flags;
        if matches!(data_type, DataType::Map(_, true)) {
            // ARROW_FLAG_MAP_KEYS_SORTED in the public Arrow C Data Interface.
            schema.flags |= 0b100;
        }
        Ok(schema)
    }

    let schema = FFI_ArrowSchema::try_from(data_type)?;
    if has_sorted_map(data_type) {
        rebuild(&schema, data_type)
    } else {
        Ok(schema)
    }
}

fn child_fields(data_type: &DataType) -> Vec<&Field> {
    match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => vec![field],
        DataType::Struct(fields) => fields.iter().map(AsRef::as_ref).collect(),
        DataType::Union(fields, _) => fields.iter().map(|(_, field)| field.as_ref()).collect(),
        DataType::RunEndEncoded(run_ends, values) => vec![run_ends, values],
        _ => vec![],
    }
}

fn schemas_equal(left: &Schema, right: &Schema) -> bool {
    fn same_order(left: &DataType, right: &DataType) -> bool {
        let fields_match =
            child_fields(left)
                .into_iter()
                .zip(child_fields(right))
                .all(|(left, right)| {
                    left.dict_is_ordered() == right.dict_is_ordered()
                        && same_order(left.data_type(), right.data_type())
                });
        fields_match
            && match (left, right) {
                (DataType::Dictionary(_, left), DataType::Dictionary(_, right)) => {
                    same_order(left, right)
                }
                _ => true,
            }
    }

    // Arrow's structural equality intentionally excludes dictionary-order flags.
    // PyArrow includes them, so a table must not relabel later batches' ordering.
    left == right
        && same_order(
            &DataType::Struct(left.fields().clone()),
            &DataType::Struct(right.fields().clone()),
        )
}

fn import_c_schema(schema: &FFI_ArrowSchema) -> Result<Schema, ArrowError> {
    let DataType::Struct(fields) = import_c_data_type(schema)? else {
        return Err(ArrowError::CDataInterface(
            "A schema must be a struct type".into(),
        ));
    };
    Ok(Schema::new(fields).with_metadata(schema.metadata()?))
}

fn import_c_data_type(schema: &FFI_ArrowSchema) -> Result<DataType, ArrowError> {
    fn has_ordered_dictionary(schema: &FFI_ArrowSchema) -> bool {
        schema.dictionary_ordered()
            || schema.children().any(has_ordered_dictionary)
            || schema.dictionary().is_some_and(has_ordered_dictionary)
    }

    // Arrow 58 drops DICTIONARY_ORDERED when importing Fields. Restore it at
    // every field, including those nested in array types and dictionary values.
    fn restore(data_type: DataType, schema: &FFI_ArrowSchema) -> Result<DataType, ArrowError> {
        let field = |field: &Field, index: usize| -> Result<Arc<Field>, ArrowError> {
            let child = schema.child(index);
            Ok(Arc::new(
                field
                    .clone()
                    .with_data_type(restore(field.data_type().clone(), child)?)
                    .with_dict_is_ordered(child.dictionary_ordered()),
            ))
        };
        Ok(match data_type {
            DataType::List(child) => DataType::List(field(&child, 0)?),
            DataType::LargeList(child) => DataType::LargeList(field(&child, 0)?),
            DataType::ListView(child) => DataType::ListView(field(&child, 0)?),
            DataType::LargeListView(child) => DataType::LargeListView(field(&child, 0)?),
            DataType::FixedSizeList(child, size) => {
                DataType::FixedSizeList(field(&child, 0)?, size)
            }
            DataType::Map(child, sorted) => DataType::Map(field(&child, 0)?, sorted),
            DataType::Struct(fields) => DataType::Struct(
                fields
                    .iter()
                    .enumerate()
                    .map(|(index, child)| field(child, index))
                    .collect::<Result<Vec<_>, _>>()?
                    .into(),
            ),
            DataType::Union(fields, mode) => DataType::Union(
                UnionFields::try_new(
                    fields.iter().map(|(id, _)| id),
                    fields
                        .iter()
                        .enumerate()
                        .map(|(index, (_, child))| field(child, index))
                        .collect::<Result<Vec<_>, _>>()?,
                )?,
                mode,
            ),
            DataType::RunEndEncoded(run_ends, values) => {
                DataType::RunEndEncoded(field(&run_ends, 0)?, field(&values, 1)?)
            }
            DataType::Dictionary(keys, values) => DataType::Dictionary(
                keys,
                Box::new(restore(
                    *values,
                    schema.dictionary().ok_or_else(|| {
                        ArrowError::CDataInterface("A dictionary must have a value type".into())
                    })?,
                )?),
            ),
            other => other,
        })
    }

    let data_type = DataType::try_from(schema)?;
    if has_ordered_dictionary(schema) {
        restore(data_type, schema)
    } else {
        Ok(data_type)
    }
}

fn require_pyarrow_class<'py>(
    value: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    // Use the extension's classes, not replaceable pyarrow package aliases.
    // Calling their descriptors also bypasses a subclass's Python override.
    let class = value.py().import("pyarrow.lib")?.getattr(name)?;
    if !value.is_instance(&class)? {
        return Err(PyTypeError::new_err(format!("Expected a pyarrow.{name}")));
    }
    Ok(class)
}

fn export_python_array(
    value: &Bound<'_, PyAny>,
    class: &Bound<'_, PyAny>,
) -> PyResult<(FFI_ArrowArray, FFI_ArrowSchema)> {
    let mut array = FFI_ArrowArray::empty();
    let mut schema = FFI_ArrowSchema::empty();
    let array_address = &mut array as *mut FFI_ArrowArray as usize;
    let schema_address = &mut schema as *mut FFI_ArrowSchema as usize;
    class.call_method1("_export_to_c", (value, array_address, schema_address))?;
    Ok((array, schema))
}

fn validate_structure(data: &ArrayData) -> PyResult<()> {
    // Arrow's inexpensive validation recurses through child descriptors. It
    // checks sizes and offset endpoints, not every offset or dictionary index;
    // those remain producer preconditions. UTF-8 is decoded when a value is read.
    data.validate().map_err(arrow_error)
}

fn nul_schema_error() -> PyErr {
    PyValueError::new_err("Arrow C schema names and timezones cannot contain NUL")
}

fn check_c_schema(schema: &Schema) -> PyResult<()> {
    if schema.fields().iter().any(|field| field_needs_ipc(field)) {
        return Err(nul_schema_error());
    }
    Ok(())
}

fn check_c_data_type(data_type: &DataType) -> PyResult<()> {
    if data_type_needs_ipc(data_type) {
        return Err(nul_schema_error());
    }
    Ok(())
}

fn field_needs_ipc(field: &Field) -> bool {
    field.name().contains('\0') || data_type_needs_ipc(field.data_type())
}

fn data_type_needs_ipc(data_type: &DataType) -> bool {
    match data_type {
        DataType::Timestamp(_, Some(timezone)) => timezone.contains('\0'),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => field_needs_ipc(field),
        DataType::Struct(fields) => fields.iter().any(|field| field_needs_ipc(field)),
        DataType::Union(fields, _) => fields.iter().any(|(_, field)| field_needs_ipc(field)),
        DataType::RunEndEncoded(run_ends, values) => {
            field_needs_ipc(run_ends) || field_needs_ipc(values)
        }
        DataType::Dictionary(keys, values) => {
            data_type_needs_ipc(keys) || data_type_needs_ipc(values)
        }
        _ => false,
    }
}

fn check_python_schema(schema: &Bound<'_, PyAny>) -> PyResult<()> {
    for index in 0..schema.len()? {
        check_python_field(&schema.call_method1("field", (index,))?)?;
    }
    Ok(())
}

fn check_python_field(field: &Bound<'_, PyAny>) -> PyResult<()> {
    if field.getattr("name")?.extract::<String>()?.contains('\0') {
        return Err(nul_schema_error());
    }
    check_python_data_type(&field.getattr("type")?)
}

fn check_python_data_type(data_type: &Bound<'_, PyAny>) -> PyResult<()> {
    // Check the original Python strings before C export can truncate at NUL.
    // Metadata uses length-prefixed strings and does not need this restriction.
    if let Ok(timezone) = data_type.getattr("tz") {
        if !timezone.is_none() && timezone.extract::<String>()?.contains('\0') {
            return Err(nul_schema_error());
        }
    }
    let num_fields = data_type.getattr("num_fields")?.extract::<usize>()?;
    for index in 0..num_fields {
        check_python_field(&data_type.call_method1("field", (index,))?)?;
    }
    // Dictionary and extension value/storage types are not child fields.
    // Lists expose both a child field and value_type; do not visit those twice.
    if num_fields == 0 {
        for name in ["value_type", "storage_type"] {
            if let Ok(child) = data_type.getattr(name) {
                if name == "value_type" && is_extension_type(&child)? {
                    return Err(PyTypeError::new_err(
                        "Arrow dictionary value extension types cannot preserve their metadata",
                    ));
                }
                if name == "value_type" {
                    reject_root_dictionary_order(&child)?;
                }
                check_python_data_type(&child)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ffi::CStr;

    use arrow::array::Int64Array;
    use arrow::datatypes::TimeUnit;
    use pyo3::types::PyDict;

    use super::*;

    fn eval<'py>(py: Python<'py>, code: &CStr) -> Bound<'py, PyAny> {
        let globals = PyDict::new(py);
        globals
            .set_item("pa", py.import("pyarrow").unwrap())
            .unwrap();
        py.eval(code, Some(&globals), None).unwrap()
    }

    #[test]
    fn schema_roundtrip_preserves_metadata() {
        Python::initialize();
        Python::attach(|py| {
            let field = Field::new("value", DataType::Int64, true)
                .with_metadata(HashMap::from([("field\0key".into(), "value\0".into())]));
            let schema = Schema::new(vec![field])
                .with_metadata(HashMap::from([("schema".into(), "metadata".into())]));
            let exported = export_schema(py, &schema).unwrap();
            assert_eq!(import_schema(exported.bind(py)).unwrap(), schema);
        });
    }

    #[test]
    fn array_roundtrip_shares_buffers_and_preserves_lifetime() {
        Python::initialize();
        Python::attach(|py| {
            let values: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)]));
            let address = values.to_data().buffers()[0].as_ptr();
            let exported = export_array(py, values).unwrap();
            let imported = import_array(exported.bind(py)).unwrap();
            drop(exported);
            assert_eq!(imported.to_data().buffers()[0].as_ptr(), address);
            assert_eq!(imported.len(), 3);
            assert!(imported.is_null(1));
            assert_eq!(
                imported
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(2),
                3
            );
        });
    }

    #[test]
    fn imports_preserve_slices_and_batch_metadata() {
        Python::initialize();
        Python::attach(|py| {
            let batch = eval(
                py,
                c"pa.RecordBatch.from_arrays(
                    [pa.array([1, None, 3])],
                    schema=pa.schema(
                        [pa.field('v', pa.int64(), metadata={b'f': b'v'})],
                        metadata={b's': b'v'}
                    )
                ).slice(1, 2)",
            );
            let imported = import_batch(&batch).unwrap();
            assert_eq!(imported.num_rows(), 2);
            assert!(imported.column(0).is_null(0));
            assert_eq!(imported.schema().metadata().get("s").unwrap(), "v");
            assert_eq!(imported.schema().field(0).metadata().get("f").unwrap(), "v");
            let table = record_batches_to_pyarrow(py, vec![imported]).unwrap();
            assert!(table
                .bind(py)
                .getattr("schema")
                .unwrap()
                .call_method1("equals", (batch.getattr("schema").unwrap(), true))
                .unwrap()
                .is_truthy()
                .unwrap());
        });
    }

    #[test]
    fn imports_preserve_zero_column_rows_and_empty_schema_metadata() {
        Python::initialize();
        Python::attach(|py| {
            for length in [0, 3] {
                let array = StructArray::new_empty_fields(length, None);
                let batch = RecordBatch::from(array);
                let schema = batch
                    .schema()
                    .as_ref()
                    .clone()
                    .with_metadata(HashMap::from([("schema".into(), "empty".into())]));
                let batch = batch.with_schema(Arc::new(schema)).unwrap();
                let table = record_batches_to_pyarrow(py, vec![batch.clone()]).unwrap();
                let batches = table.bind(py).call_method0("to_batches").unwrap();
                if length == 0 {
                    assert_eq!(
                        table
                            .bind(py)
                            .getattr("num_rows")
                            .unwrap()
                            .extract::<usize>()
                            .unwrap(),
                        0
                    );
                    assert_eq!(
                        import_schema(&table.bind(py).getattr("schema").unwrap()).unwrap(),
                        *batch.schema()
                    );
                } else {
                    let imported = import_batch(&batches.get_item(0).unwrap()).unwrap();
                    assert_eq!(imported.num_rows(), length);
                    assert_eq!(imported.num_columns(), 0);
                    assert_eq!(imported.schema(), batch.schema());
                }
            }
        });
    }

    #[test]
    fn imports_reject_wrong_pyarrow_classes() {
        Python::initialize();
        Python::attach(|py| {
            let array = eval(py, c"pa.array([1, 2])");
            assert!(import_schema(&array)
                .unwrap_err()
                .is_instance_of::<PyTypeError>(py));
            assert!(import_data_type(&array)
                .unwrap_err()
                .is_instance_of::<PyTypeError>(py));
            assert!(import_batch(&array)
                .unwrap_err()
                .is_instance_of::<PyTypeError>(py));
            let schema = eval(py, c"pa.schema([('v', pa.int64())])");
            assert!(import_array(&schema)
                .unwrap_err()
                .is_instance_of::<PyTypeError>(py));
            let fake = eval(
                py,
                c"type('Fake', (), {'_export_to_c': lambda *args: 1 / 0})()",
            );
            assert!(import_schema(&fake)
                .unwrap_err()
                .is_instance_of::<PyTypeError>(py));
            assert!(import_data_type(&fake)
                .unwrap_err()
                .is_instance_of::<PyTypeError>(py));
            assert!(import_array(&fake)
                .unwrap_err()
                .is_instance_of::<PyTypeError>(py));
            assert!(import_batch(&fake)
                .unwrap_err()
                .is_instance_of::<PyTypeError>(py));
        });
    }

    #[test]
    fn batch_import_rejects_nulls_in_non_nullable_columns() {
        Python::initialize();
        Python::attach(|py| {
            let batch = eval(
                py,
                c"pa.RecordBatch.from_arrays(
                    [pa.array([None], type=pa.int64())],
                    schema=pa.schema([pa.field('v', pa.int64(), nullable=False)])
                )",
            );
            assert!(import_batch(&batch)
                .unwrap_err()
                .to_string()
                .contains("non-nullable"));
        });
    }

    #[test]
    fn nul_schema_strings_are_rejected_without_panicking() {
        let field = Field::new("bad\0name", DataType::Int64, true);
        assert!(check_c_schema(&Schema::new(vec![field.clone()])).is_err());
        assert!(check_c_data_type(&DataType::List(Arc::new(field))).is_err());
        let timestamp = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC\0suffix".into()));
        assert!(check_c_data_type(&timestamp).is_err());
        assert!(check_c_data_type(&DataType::Dictionary(
            Box::new(DataType::Int8),
            Box::new(timestamp)
        ))
        .is_err());
        Python::initialize();
        Python::attach(|py| {
            assert!(export_data_type(py, &DataType::Time32(TimeUnit::Nanosecond)).is_err());
            let schema = eval(py, c"pa.schema([('bad\\x00name', pa.int64())])");
            assert!(import_schema(&schema)
                .unwrap_err()
                .is_instance_of::<PyValueError>(py));
            let data_type = eval(py, c"pa.timestamp('us', tz='UTC\\x00suffix')");
            assert!(import_data_type(&data_type)
                .unwrap_err()
                .is_instance_of::<PyValueError>(py));
        });
    }
}
