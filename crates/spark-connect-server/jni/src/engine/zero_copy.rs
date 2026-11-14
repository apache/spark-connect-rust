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

//! Borrow immutable Java-owned Arrow buffers through the Arrow C Data Interface.
//! Descriptor metadata is checked before Arrow's unsafe importer can read offsets.
use super::MAX_BYTES;
use arrow::{
    array::{
        ffi::{from_ffi_and_data_type, FFI_ArrowArray},
        layout, Array, BufferSpec, StructArray,
    },
    datatypes::{DataType, SchemaRef},
    record_batch::{RecordBatch, RecordBatchOptions},
};
use tonic::Status;

extern "C" {
    fn spark_connect_engine_arrow_buffer_size(array: *const FFI_ArrowArray, index: i64) -> i64;
}

/// Import the trusted JNI producer's C descriptors without copying value buffers.
///
/// # Safety
/// Array descriptor tables and every buffer must remain immutable, valid memory
/// for the capacity reported by the producer until its release callback runs.
pub(super) unsafe fn import_batch(
    array: FFI_ArrowArray,
    schema: SchemaRef,
) -> Result<RecordBatch, Status> {
    unsafe {
        import_batch_with_sizes(array, schema, |node, index| {
            usize::try_from(spark_connect_engine_arrow_buffer_size(node, index as i64)).ok()
        })
    }
}

/// Injectable capacity lookup permits real importer/lifetime tests without a JVM.
///
/// # Safety
/// The descriptor tables must be valid, and sizes must report actual accessible
/// capacities of immutable buffers, not lengths guessed from an untrusted schema.
pub(super) unsafe fn import_batch_with_sizes<F>(
    array: FFI_ArrowArray,
    schema: SchemaRef,
    sizes: F,
) -> Result<RecordBatch, Status>
where
    F: Fn(&FFI_ArrowArray, usize) -> Option<usize>,
{
    let data_type = DataType::Struct(schema.fields().clone());
    let mut budget = Budget { nodes: 0, bytes: 0 };
    // C produces independently-owned, finite descriptor trees with zero offsets.
    unsafe {
        validate(&array, &data_type, &sizes, 0, &mut budget)?;
    }
    if array.null_count != 0 {
        return Err(invalid("record-batch root cannot be nullable"));
    }
    // The importer shares every nonempty buffer with an Arc owning the root C
    // descriptor. All clones/slices preserve its final release callback.
    let data = unsafe { from_ffi_and_data_type(array, data_type) }
        .map_err(|e| invalid(&format!("C Data import: {e}")))?;
    data.validate_full()
        .map_err(|e| invalid(&format!("invalid Arrow data: {e}")))?;
    let values = StructArray::from(data);
    RecordBatch::try_new_with_options(
        schema,
        values.columns().to_vec(),
        &RecordBatchOptions::new().with_row_count(Some(values.len())),
    )
    .map_err(|e| invalid(&format!("Arrow record batch: {e}")))
}
fn invalid(message: &str) -> Status {
    Status::internal(format!("JVM Arrow C Data: {message}"))
}
struct Budget {
    nodes: usize,
    bytes: usize,
}
// Keep in sync with SPARK_CONNECT_MAX_ARROW_NODES in native/bridge.h.
const MAX_ARROW_NODES: usize = 65_536;

unsafe fn validate<F>(
    array: &FFI_ArrowArray,
    kind: &DataType,
    sizes: &F,
    depth: usize,
    budget: &mut Budget,
) -> Result<(), Status>
where
    F: Fn(&FFI_ArrowArray, usize) -> Option<usize>,
{
    budget.nodes += 1;
    if depth > 128 || budget.nodes > MAX_ARROW_NODES {
        return Err(invalid("descriptor tree exceeds limits"));
    }
    if array.release.is_none()
        || array.length < 0
        || array.offset != 0
        || array.null_count < 0
        || array.null_count > array.length
    {
        return Err(invalid(
            "invalid length, offset, null count or release callback",
        ));
    }
    if !array.dictionary.is_null() {
        return Err(Status::unimplemented("JVM Arrow dictionaries"));
    }
    if matches!(
        kind,
        DataType::Dictionary(_, _)
            | DataType::Union(_, _)
            | DataType::RunEndEncoded(_, _)
            | DataType::Utf8View
            | DataType::BinaryView
            | DataType::ListView(_)
            | DataType::LargeListView(_)
    ) {
        return Err(Status::unimplemented(format!(
            "JVM Arrow C Data type: {kind}"
        )));
    }
    let len =
        usize::try_from(array.length).map_err(|_| invalid("array length exceeds address space"))?;
    let layout = layout(kind);
    if layout.variadic {
        return Err(Status::unimplemented("JVM Arrow variadic buffers"));
    }
    let expected_buffers = layout.buffers.len() + usize::from(layout.can_contain_null_mask);
    if array.n_buffers != expected_buffers as i64
        || (expected_buffers > 0 && array.buffers.is_null())
    {
        return Err(invalid("buffer count or buffer table differs from schema"));
    }
    let child_types = match kind {
        DataType::Struct(fields) => fields.iter().map(|f| f.data_type()).collect::<Vec<_>>(),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::Map(field, _)
        | DataType::FixedSizeList(field, _) => vec![field.data_type()],
        _ => vec![],
    };
    if array.n_children != child_types.len() as i64
        || (!child_types.is_empty() && array.children.is_null())
    {
        return Err(invalid("child count or child table differs from schema"));
    }
    let mut caps = Vec::with_capacity(expected_buffers);
    for index in 0..expected_buffers {
        let cap = sizes(array, index).ok_or_else(|| invalid("missing buffer capacity"))?;
        if cap > isize::MAX as usize {
            return Err(invalid("buffer exceeds addressable capacity"));
        }
        budget.bytes = budget
            .bytes
            .checked_add(cap)
            .ok_or_else(|| invalid("buffer size overflow"))?;
        if budget.bytes > MAX_BYTES {
            return Err(Status::resource_exhausted(
                "JVM Arrow batch allocation exceeds 64 MiB",
            ));
        }
        caps.push(cap);
    }
    let check = |index: usize, required: usize, alignment: usize| -> Result<(), Status> {
        let pointer = unsafe { *array.buffers.add(index) };
        if required > caps[index] {
            return Err(invalid("buffer is shorter than its declared array"));
        }
        if required != 0 && (pointer.is_null() || !(pointer as usize).is_multiple_of(alignment)) {
            return Err(invalid("nonempty buffer is null or misaligned"));
        }
        Ok(())
    };
    let start = usize::from(layout.can_contain_null_mask);
    if layout.can_contain_null_mask {
        let pointer = unsafe { *array.buffers };
        if pointer.is_null() {
            if array.null_count != 0 {
                return Err(invalid("null values without a validity buffer"));
            }
        } else {
            check(0, len.div_ceil(8), 1)?;
        }
    }
    let offsets = matches!(
        kind,
        DataType::Utf8
            | DataType::Binary
            | DataType::LargeUtf8
            | DataType::LargeBinary
            | DataType::List(_)
            | DataType::LargeList(_)
            | DataType::Map(_, _)
    );
    let offset_items = if offsets {
        len.checked_add(1)
            .ok_or_else(|| invalid("offset count overflow"))?
    } else {
        len
    };
    // First check all fixed-width/bitmap bounds, before reading any offsets.
    for (position, spec) in layout.buffers.iter().enumerate() {
        match spec {
            BufferSpec::FixedWidth {
                byte_width,
                alignment,
            } => {
                let items = if offsets && position == 0 {
                    offset_items
                } else {
                    len
                };
                check(
                    start + position,
                    items
                        .checked_mul(*byte_width)
                        .ok_or_else(|| invalid("buffer size overflow"))?,
                    *alignment,
                )?;
            }
            BufferSpec::BitMap => check(start + position, len.div_ceil(8), 1)?,
            BufferSpec::AlwaysNull => {}
            BufferSpec::VariableWidth => {}
        }
    }
    let mut last_offset = None;
    if offsets {
        let large = matches!(
            kind,
            DataType::LargeUtf8 | DataType::LargeBinary | DataType::LargeList(_)
        );
        let pointer = unsafe { *array.buffers.add(start) };
        let mut previous = 0_i64;
        for index in 0..offset_items {
            // Fixed-width bounds and alignment were checked above. Java owns
            // these immutable offset buffers for the duration of the lease.
            let offset = if large {
                unsafe { *pointer.cast::<i64>().add(index) }
            } else {
                i64::from(unsafe { *pointer.cast::<i32>().add(index) })
            };
            if offset < previous {
                return Err(invalid("negative or decreasing Arrow offsets"));
            }
            previous = offset;
        }
        last_offset =
            Some(usize::try_from(previous).map_err(|_| invalid("offset exceeds address space"))?);
        if matches!(
            kind,
            DataType::Utf8 | DataType::Binary | DataType::LargeUtf8 | DataType::LargeBinary
        ) {
            check(start + 1, last_offset.unwrap(), 1)?;
        }
    }
    for (index, child_type) in child_types.iter().enumerate() {
        let pointer = unsafe { *array.children.add(index) };
        if pointer.is_null() {
            return Err(invalid("null child descriptor"));
        }
        let child = unsafe { &*pointer };
        if child.length < 0 {
            return Err(invalid("negative child length"));
        }
        let child_len = usize::try_from(child.length)
            .map_err(|_| invalid("child length exceeds address space"))?;
        match kind {
            DataType::Struct(_) if child_len != len => {
                return Err(invalid("struct child length differs from parent"))
            }
            DataType::List(_) | DataType::LargeList(_) | DataType::Map(_, _) => {
                if last_offset.unwrap() > child_len {
                    return Err(invalid("list/map offset exceeds child length"));
                }
            }
            DataType::FixedSizeList(_, size) => {
                let size =
                    usize::try_from(*size).map_err(|_| invalid("negative fixed-list width"))?;
                if len
                    .checked_mul(size)
                    .ok_or_else(|| invalid("fixed-list size overflow"))?
                    != child_len
                {
                    return Err(invalid("fixed-list child length differs from parent"));
                }
            }
            _ => {}
        }
        unsafe {
            validate(child, child_type, sizes, depth + 1, budget)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_node_budget_accepts_its_limit_and_rejects_the_next_node() {
        let kind = DataType::Null;
        let array = FFI_ArrowArray::new(&arrow::array::ArrayData::new_empty(&kind));
        let mut budget = Budget {
            nodes: MAX_ARROW_NODES - 1,
            bytes: 0,
        };
        let sizes = |_: &FFI_ArrowArray, _: usize| Some(0);
        unsafe { validate(&array, &kind, &sizes, 0, &mut budget) }.unwrap();
        assert_eq!(budget.nodes, MAX_ARROW_NODES);
        let error = unsafe { validate(&array, &kind, &sizes, 0, &mut budget) }.unwrap_err();
        assert_eq!(error.code(), tonic::Code::Internal);
        assert!(error.message().contains("descriptor tree exceeds limits"));
    }
}
