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

//! Synthetic producer fixtures for the actual zero-copy importer. The buffers
//! model an external immutable allocation and use a separate release callback.
use super::zero_copy::import_batch_with_sizes;
use arrow::array::builder::{Int64Builder, StringBuilder};
use arrow::array::{
    Array, ArrayData, ArrayRef, BooleanArray, Decimal128Array, Int32Array, Int64Array, ListArray,
    MapBuilder, NullArray, StringArray, StructArray, TimestampMicrosecondArray,
};
use arrow::buffer::{Buffer, NullBuffer};
use arrow::datatypes::{DataType, Field, Fields, Int32Type, Schema, SchemaRef};
use arrow::ffi::FFI_ArrowArray;
use arrow::record_batch::RecordBatch;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

struct Probe {
    _original: FFI_ArrowArray,
    releases: Arc<AtomicUsize>,
}

unsafe extern "C" fn release_probe(array: *mut FFI_ArrowArray) {
    let array = unsafe { &mut *array };
    if array.release.is_none() {
        return;
    }
    array.release = None;
    let probe = unsafe { Box::from_raw(array.private_data.cast::<Probe>()) };
    array.private_data = std::ptr::null_mut();
    probe.releases.fetch_add(1, Ordering::SeqCst);
    // Dropping original releases every producer-owned child and buffer.
    drop(probe);
}

struct Fixture {
    array: FFI_ArrowArray,
    sizes: HashMap<usize, usize>,
    releases: Arc<AtomicUsize>,
}

impl Fixture {
    fn new(data: &ArrayData) -> Self {
        let original = FFI_ArrowArray::new(data);
        let releases = Arc::new(AtomicUsize::new(0));
        let array = FFI_ArrowArray {
            length: original.length,
            null_count: original.null_count,
            offset: original.offset,
            n_buffers: original.n_buffers,
            n_children: original.n_children,
            buffers: original.buffers,
            children: original.children,
            dictionary: original.dictionary,
            release: Some(release_probe),
            private_data: std::ptr::null_mut(),
        };
        let mut array = array;
        array.private_data = Box::into_raw(Box::new(Probe {
            _original: original,
            releases: releases.clone(),
        }))
        .cast::<c_void>();
        let mut sizes = HashMap::new();
        collect_sizes(data, &mut sizes);
        Self {
            array,
            sizes,
            releases,
        }
    }

    fn import(self, schema: SchemaRef) -> Result<RecordBatch, tonic::Status> {
        let sizes = self.sizes;
        // All descriptor tables and their referenced buffers belong to original
        // inside Probe and remain immutable until release_probe runs. Resolver
        // lengths are actual producer buffer lengths, including validity bytes.
        unsafe {
            import_batch_with_sizes(self.array, schema, move |node, index| {
                if node.buffers.is_null() || index >= node.n_buffers.max(0) as usize {
                    return None;
                }
                let pointer = *node.buffers.add(index);
                if pointer.is_null() {
                    Some(0)
                } else {
                    sizes.get(&(pointer as usize)).copied()
                }
            })
        }
    }
}

fn collect_sizes(data: &ArrayData, sizes: &mut HashMap<usize, usize>) {
    for buffer in data.buffers() {
        sizes
            .entry(buffer.as_ptr() as usize)
            .and_modify(|n| *n = (*n).max(buffer.len()))
            .or_insert(buffer.len());
    }
    if let Some(nulls) = data.nulls() {
        let buffer = nulls.buffer();
        sizes
            .entry(buffer.as_ptr() as usize)
            .and_modify(|n| *n = (*n).max(buffer.len()))
            .or_insert(buffer.len());
    }
    for child in data.child_data() {
        collect_sizes(child, sizes);
    }
}

fn batch_fixture(columns: Vec<(&str, ArrayRef)>) -> (RecordBatch, Fixture) {
    let fields: Fields = columns
        .iter()
        .map(|(name, array)| Arc::new(Field::new(*name, array.data_type().clone(), true)))
        .collect();
    let arrays = columns
        .into_iter()
        .map(|(_, array)| array)
        .collect::<Vec<_>>();
    let root = StructArray::new(fields.clone(), arrays.clone(), None);
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
    let fixture = Fixture::new(&root.to_data());
    (batch, fixture)
}

fn assert_same_buffers(expected: &ArrayData, actual: &ArrayData) {
    assert_eq!(expected.data_type(), actual.data_type());
    assert_eq!(expected.buffers().len(), actual.buffers().len());
    for (left, right) in expected.buffers().iter().zip(actual.buffers()) {
        if !left.is_empty() {
            assert_eq!(
                left.as_ptr(),
                right.as_ptr(),
                "value/offset buffer was copied"
            );
        }
    }
    if let (Some(left), Some(right)) = (expected.nulls(), actual.nulls()) {
        if !left.buffer().is_empty() {
            assert_eq!(
                left.buffer().as_ptr(),
                right.buffer().as_ptr(),
                "validity buffer was copied"
            );
        }
    }
    assert_eq!(expected.child_data().len(), actual.child_data().len());
    for (left, right) in expected.child_data().iter().zip(actual.child_data()) {
        assert_same_buffers(left, right);
    }
}

#[test]
fn imports_nested_arrow_without_copying_values_offsets_or_validity() {
    let longs: ArrayRef = Arc::new(Int64Array::from(vec![Some(1), None, Some(-3)]));
    let strings: ArrayRef = Arc::new(StringArray::from(vec![Some("a"), None, Some("π😀")]));
    let lists: ArrayRef = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
        Some(vec![Some(1), None]),
        None,
        Some(vec![]),
    ]));
    let mut map = MapBuilder::new(None, StringBuilder::new(), Int64Builder::new());
    map.keys().append_value("first");
    map.values().append_value(7);
    map.append(true).unwrap();
    map.append(false).unwrap();
    map.keys().append_value("π");
    map.values().append_null();
    map.append(true).unwrap();
    let map: ArrayRef = Arc::new(map.finish());
    let decimal: ArrayRef = Arc::new(
        Decimal128Array::from(vec![Some(12345), None, Some(-987)])
            .with_precision_and_scale(12, 2)
            .unwrap(),
    );
    let timestamp: ArrayRef = Arc::new(
        TimestampMicrosecondArray::from(vec![Some(0), None, Some(1234567)]).with_timezone("UTC"),
    );
    let nested: ArrayRef = Arc::new(StructArray::new(
        vec![
            Arc::new(Field::new("number", DataType::Int64, true)),
            Arc::new(Field::new("text", DataType::Utf8, true)),
        ]
        .into(),
        vec![longs.clone(), strings.clone()],
        Some(NullBuffer::from(vec![true, false, true])),
    ));
    let boolean: ArrayRef = Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)]));
    let (expected, fixture) = batch_fixture(vec![
        ("longs", longs),
        ("strings", strings),
        ("lists", lists),
        ("maps", map),
        ("decimal", decimal),
        ("timestamp", timestamp),
        ("nested", nested),
        ("boolean", boolean),
    ]);
    let releases = fixture.releases.clone();
    let actual = fixture.import(expected.schema()).unwrap();
    assert_eq!(expected, actual);
    for (left, right) in expected.columns().iter().zip(actual.columns()) {
        assert_same_buffers(&left.to_data(), &right.to_data());
    }
    assert_eq!(releases.load(Ordering::SeqCst), 0);
    drop(actual);
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[test]
fn slices_and_column_clones_retain_external_owner_until_last_drop_on_other_thread() {
    let (expected, fixture) = batch_fixture(vec![(
        "text",
        Arc::new(StringArray::from(vec![Some("first"), None, Some("last")])),
    )]);
    let releases = fixture.releases.clone();
    let batch = fixture.import(expected.schema()).unwrap();
    let column = batch.column(0).clone();
    let slice = column.slice(1, 2);
    drop(batch);
    drop(column);
    drop(expected);
    assert_eq!(releases.load(Ordering::SeqCst), 0);
    let count = releases.clone();
    std::thread::spawn(move || {
        assert_eq!(slice.len(), 2);
        assert!(slice.is_null(0));
        let values = slice.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(values.value(1), "last");
        assert_eq!(count.load(Ordering::SeqCst), 0);
        drop(slice);
        assert_eq!(count.load(Ordering::SeqCst), 1);
    })
    .join()
    .unwrap();
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[test]
fn null_only_data_releases_external_owner_early_without_losing_values() {
    let (expected, fixture) = batch_fixture(vec![("nulls", Arc::new(NullArray::new(3)))]);
    let releases = fixture.releases.clone();
    let actual = fixture.import(expected.schema()).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.num_rows(), 3);
    assert_eq!(actual.column(0).logical_null_count(), 3);
    assert_eq!(releases.load(Ordering::SeqCst), 1);
    drop(actual);
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[test]
fn empty_variable_width_batch_remains_valid_and_releases_once() {
    let (expected, fixture) = batch_fixture(vec![(
        "text",
        Arc::new(StringArray::from(Vec::<Option<&str>>::new())),
    )]);
    let releases = fixture.releases.clone();
    let actual = fixture.import(expected.schema()).unwrap();
    assert_eq!(actual, expected);
    drop(actual);
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[test]
fn empty_string_values_do_not_require_an_allocated_value_buffer() {
    let (expected, fixture) =
        batch_fixture(vec![("text", Arc::new(StringArray::from(vec!["", ""])))]);
    let releases = fixture.releases.clone();
    let actual = fixture.import(expected.schema()).unwrap();
    assert_eq!(actual, expected);
    for (left, right) in expected.columns().iter().zip(actual.columns()) {
        assert_same_buffers(&left.to_data(), &right.to_data());
    }
    drop(actual);
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[test]
fn zero_column_batch_preserves_row_count_without_retaining_external_storage() {
    let values = StructArray::new_empty_fields(3, None);
    let fixture = Fixture::new(&values.to_data());
    let releases = fixture.releases.clone();
    let actual = fixture.import(Arc::new(Schema::empty())).unwrap();
    assert_eq!(actual.num_columns(), 0);
    assert_eq!(actual.num_rows(), 3);
    assert_eq!(releases.load(Ordering::SeqCst), 1);
    drop(actual);
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

fn primitive_fixture() -> (SchemaRef, Fixture) {
    let (batch, fixture) = batch_fixture(vec![(
        "number",
        Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])),
    )]);
    (batch.schema(), fixture)
}

fn fails_and_releases(fixture: Fixture, schema: SchemaRef) {
    let releases = fixture.releases.clone();
    assert!(fixture.import(schema).is_err());
    assert_eq!(releases.load(Ordering::SeqCst), 1);
}

#[test]
fn rejects_negative_offsets_lengths_and_child_counts_before_import() {
    for kind in 0..4 {
        let (schema, mut fixture) = primitive_fixture();
        match kind {
            0 => fixture.array.offset = -1,
            1 => fixture.array.length = -1,
            2 => fixture.array.n_children = 0,
            _ => fixture.array.null_count = fixture.array.length + 1,
        }
        fails_and_releases(fixture, schema);
    }
}

#[test]
fn rejects_missing_buffers_invalid_null_masks_and_short_capacities() {
    for kind in 0..6 {
        let (schema, fixture) = primitive_fixture();
        let mut fixture = fixture;
        let child = unsafe { &mut **fixture.array.children };
        match kind {
            0 => child.n_buffers = 1,
            1 => child.buffers = std::ptr::null_mut(),
            2 => {
                child.null_count = 1;
                unsafe {
                    *child.buffers = std::ptr::null();
                }
            }
            3 => {
                let pointer = unsafe { *child.buffers.add(1) };
                fixture.sizes.insert(pointer as usize, 3);
            }
            4 => {
                let pointer = unsafe { *child.buffers };
                fixture.sizes.insert(pointer as usize, 0);
            }
            _ => {
                let pointer = unsafe { *child.buffers.add(1) };
                fixture.sizes.remove(&(pointer as usize));
            }
        }
        fails_and_releases(fixture, schema);
    }
}

#[test]
fn rejects_invalid_child_shapes_and_releases_producer_once() {
    for kind in 0..6 {
        let (schema, mut fixture) = primitive_fixture();
        match kind {
            0 => fixture.array.n_children = 2,
            1 => fixture.array.n_buffers = -1,
            2 => fixture.array.children = std::ptr::null_mut(),
            3 => unsafe { (**fixture.array.children).length = 2 },
            4 => unsafe { (**fixture.array.children).offset = 1 },
            _ => unsafe { (**fixture.array.children).n_children = 1 },
        }
        fails_and_releases(fixture, schema);
    }
}

#[test]
fn rejects_unsupported_dictionary_arrays_without_copying_or_leaking() {
    let (_, fixture) = primitive_fixture();
    let schema = Arc::new(Schema::new(vec![Field::new(
        "dictionary",
        DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
        true,
    )]));
    fails_and_releases(fixture, schema);
}

#[test]
fn rejects_unaligned_external_values_without_copying_to_repair_them() {
    let data = ArrayData::builder(DataType::Int32)
        .len(1)
        .add_buffer(Buffer::from_slice_ref([1_i32, 2, 3, 4]))
        .build()
        .unwrap();
    let root = ArrayData::builder(DataType::Struct(
        vec![Arc::new(Field::new("n", DataType::Int32, true))].into(),
    ))
    .len(1)
    .add_child_data(data)
    .build()
    .unwrap();
    let mut fixture = Fixture::new(&root);
    let child = unsafe { &mut **fixture.array.children };
    let pointer = unsafe { (*child.buffers.add(1)).cast::<u8>().add(1).cast::<c_void>() };
    unsafe {
        *child.buffers.add(1) = pointer;
    }
    fixture.sizes.insert(pointer as usize, 15);
    fails_and_releases(
        fixture,
        Arc::new(Schema::new(vec![Field::new("n", DataType::Int32, true)])),
    );
}

fn malformed_strings(offsets: &[i32], values: &[u8]) -> (SchemaRef, Fixture) {
    let string = unsafe {
        ArrayData::builder(DataType::Utf8)
            .len(offsets.len() - 1)
            .add_buffer(Buffer::from_slice_ref(offsets))
            .add_buffer(Buffer::from(values))
            .build_unchecked()
    };
    let field = Field::new("text", DataType::Utf8, true);
    let root = unsafe {
        ArrayData::builder(DataType::Struct(vec![Arc::new(field.clone())].into()))
            .len(offsets.len() - 1)
            .add_child_data(string)
            .build_unchecked()
    };
    (Arc::new(Schema::new(vec![field])), Fixture::new(&root))
}

#[test]
fn rejects_negative_nonmonotonic_and_out_of_bounds_variable_offsets() {
    for (offsets, bytes) in [
        (&[0, -1][..], &b"abc"[..]),
        (&[0, 2, 1][..], &b"abc"[..]),
        (&[0, 10][..], &b"abc"[..]),
    ] {
        let (schema, fixture) = malformed_strings(offsets, bytes);
        fails_and_releases(fixture, schema);
    }
}

#[test]
fn rejects_invalid_utf8_before_exposing_arrow_values() {
    let (schema, fixture) = malformed_strings(&[0, 1], &[0xff]);
    fails_and_releases(fixture, schema);
}
