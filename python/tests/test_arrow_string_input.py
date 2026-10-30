#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Explicit Arrow-backed StringDtype input checks against the offline RPC service."""

import json
from types import MethodType

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pandas.core.arrays import string_arrow as string_arrow_module
from pandas.core.internals import blocks as blocks_module, managers as managers_module
from pyspark.sql import types as T
from pyspark.sql import _pandas_input as pandas_input
from pyspark.sql._pandas_input import prepare_string_input

from tests.test_arrow_handoff import _arrow_input_service, protocol  # noqa: F401


@pytest.fixture(params=["pd_na", "nan"])
def missing_kind(request):
    return request.param


def _dtype(missing_kind, storage="pyarrow"):
    return pd.StringDtype(
        storage=storage, na_value=pd.NA if missing_kind == "pd_na" else np.nan
    )


def _frame(values, missing_kind="pd_na"):
    """Always request the backend; pandas' default string inference is not evidence."""
    frame = pd.DataFrame({
        name: pd.Series(pd.array(column, dtype=_dtype(missing_kind)))
        for name, column in values.items()
    })
    # Isolate data-array hooks from pandas 3's independently inferred label
    # storage. Canonical extension-backed labels have their own existing tests.
    frame.columns = pd.Index(list(values), dtype=object)
    return frame


def _backing(frame, index=0):
    return frame.iloc[:, index].array


def _assert_backend(frame, missing_kind):
    expected_class = string_arrow_module.ArrowStringArray
    if missing_kind == "nan":
        expected_class = vars(string_arrow_module).get(
            "ArrowStringArrayNumpySemantics", expected_class
        )
    dtypes = frame.dtypes
    for index, column in enumerate(frame.columns):
        array = frame[column].array
        assert type(array) is expected_class
        assert type(array.dtype) is pd.StringDtype
        assert array.dtype is dtypes.iloc[index]
        assert array.dtype.storage == "pyarrow"
        if missing_kind == "pd_na":
            assert array.dtype.na_value is pd.NA
        else:
            assert type(array.dtype.na_value) is float and np.isnan(array.dtype.na_value)
        assert type(array._pa_array) is pa.ChunkedArray
        assert all(type(chunk) in (pa.StringArray, pa.LargeStringArray)
                   for chunk in array._pa_array.chunks)


def _schema(names=("text",), nullable=True):
    return T.StructType([T.StructField(name, T.StringType(), nullable) for name in names])


def _assert_inline(state, schema, expected):
    assert state["config_requests"] == []
    assert state["uploads"] == 0
    assert state["cache"] == {}
    assert len(state["relations"]) == 1
    relation = state["relations"][0]
    assert relation.HasField("local_relation")
    assert json.loads(relation.local_relation.schema) == json.loads(schema.json())
    table = state["input_tables"][0]
    assert table.schema.names == schema.names
    assert table.schema.metadata is None
    assert table.num_rows == len(expected[0])
    for index, (field, values) in enumerate(zip(schema.fields, expected)):
        actual_field = table.schema.field(index)
        assert actual_field.type == pa.string()
        assert actual_field.nullable == field.nullable
        # Spark metadata belongs to LocalRelation's logical schema on the row path.
        assert actual_field.metadata is None
        assert table.column(index).to_pylist() == values


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("arrow_type", [pa.string(), pa.large_string()], ids=["string", "large"])
def test_arrow_string_preserve_every_cell_and_inline_schema(
    protocol, monkeypatch, missing_kind, verify, arrow_type
):
    strings = ["", "plain", "é", "e\u0301", "\x00embedded\x00", "漢字", "🙂",
               "\U0010ffff", "\uffff", "\u202eright-to-left"] * 1000
    nullable = [None if i % 7 == 0 else value for i, value in enumerate(reversed(strings))]
    expected = [strings, nullable, [None] * len(strings)]
    frame = _frame(dict(zip(["original", "b", "c"], expected)), missing_kind)
    # These are ordinary concrete Arrow string buffers with deliberately different
    # chunk boundaries. Every chunk is sliced from a larger buffer with offset > 0.
    partitions = ([1, 0, 997, 1001, 8001], [3333, 1, 6666], [5001, 4999])
    for index, (values, sizes) in enumerate(zip(expected, partitions)):
        chunks, start = [], 0
        for size in sizes:
            chunk = pa.array(["padding"] + values[start:start + size] + ["padding"],
                             type=arrow_type).slice(1, size)
            assert chunk.offset == 1
            chunks.append(chunk)
            start += size
        assert start == len(strings)
        _backing(frame, index)._pa_array = pa.chunked_array(chunks, type=arrow_type)
    assert [len(chunk) for chunk in _backing(frame)._pa_array.chunks] == partitions[0]
    _assert_backend(frame, missing_kind)
    schema = T.StructType([
        T.StructField("renamed.`text", T.StringType(), False,
                      {"source": "arrow-string", "id": 3}),
        T.StructField("nullable", T.StringType(), True, {"tag": [1, 2]}),
        T.StructField("all_null", T.StringType()),
    ])
    ipc = prepare_string_input(frame, schema)
    assert type(ipc) is bytes
    private_table = pa.ipc.open_stream(ipc).read_all()
    assert [column.to_pylist() for column in private_table.columns] == expected

    def no_rows(*args, **kwargs):
        raise AssertionError("canonical Arrow strings must not materialize Python rows")

    for method in ("to_numpy", "iterrows", "itertuples"):
        monkeypatch.setattr(pd.DataFrame, method, no_rows)
    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    monkeypatch.setattr(pd.Series, "tolist", no_rows)
    with _arrow_input_service(protocol, config_error=True, **{
        "spark.sql.session.localRelationCacheThreshold": "1",
        "spark.sql.session.localRelationSizeLimit": "1",
        "spark.sql.session.localRelationChunkSizeRows": "997",
        "spark.sql.execution.arrow.useLargeVarTypes": "true",
    }) as (spark, state):
        spark.createDataFrame(frame, schema, verifySchema=verify).toArrow()
        _assert_inline(state, schema, expected)


@pytest.mark.parametrize("rows,eligible", [(0, False), (9999, False), (10000, True)])
def test_arrow_string_threshold_is_explicit(missing_kind, rows, eligible):
    frame = _frame({"text": [""] * rows}, missing_kind)
    _assert_backend(frame, missing_kind)
    result = prepare_string_input(frame, _schema())
    assert (type(result) is bytes) is eligible
    if not eligible:
        assert result is None


@pytest.mark.parametrize("reason", [
    "inferred", "names", "ddl", "atomic", "cast", "mixed_schema", "width",
    "duplicate_input", "duplicate_schema", "input_nul", "input_number", "schema_nul",
    "collation", "nonnullable_null", "python", "generic_arrow", "object_mixed",
    "python_mixed", "frame_subclass", "array_subclass", "dtype_subclass",
    "array_storage", "binary_storage", "integer_storage", "short_storage", "invalid_utf8",
    "invalid_offsets",
])
def test_arrow_string_guard_keeps_unsupported_inputs_on_rows(missing_kind, reason):
    values = ["plain"] * 10000
    frame = _frame({"text": values}, missing_kind)
    schema = _schema()
    if reason in ("inferred", "names", "ddl", "atomic", "cast"):
        schema = {
            "inferred": None, "names": ["text"], "ddl": "text STRING",
            "atomic": T.StringType(),
            "cast": T.StructType([T.StructField("text", T.LongType())]),
        }[reason]
    elif reason == "mixed_schema":
        frame["long"] = np.arange(10000)
        schema.add("long", T.LongType())
    elif reason == "width":
        schema.add("extra", T.StringType())
    elif reason in ("duplicate_input", "duplicate_schema"):
        frame["other"] = pd.Series(pd.array(values, dtype=_dtype(missing_kind)))
        schema.add("other", T.StringType())
        if reason == "duplicate_input":
            frame.columns = ["text", "text"]
        else:
            schema = _schema(("text", "text"))
    elif reason in ("input_nul", "input_number"):
        frame.columns = ["text\x00" if reason == "input_nul" else 1]
    elif reason == "schema_nul":
        schema = _schema(("text\x00",))
    elif reason == "collation":
        schema = T.StructType([T.StructField("text", T.StringType("UTF8_LCASE"))])
    elif reason == "nonnullable_null":
        frame = _frame({"text": values[:-1] + [None]}, missing_kind)
        schema = _schema(nullable=False)
    elif reason == "python":
        frame = pd.DataFrame({"text": pd.Series(values, dtype=_dtype(missing_kind, "python"))})
    elif reason == "generic_arrow":
        frame = pd.DataFrame({"text": pd.Series(values, dtype=pd.ArrowDtype(pa.large_string()))})
    elif reason in ("object_mixed", "python_mixed"):
        dtype = object if reason == "object_mixed" else _dtype(missing_kind, "python")
        frame["other"] = pd.Series(values, dtype=dtype)
        schema.add("other", T.StringType())
    elif reason == "frame_subclass":
        class Frame(pd.DataFrame):
            pass

        frame = Frame(frame)
    elif reason == "array_subclass":
        array = _backing(frame)

        class Array(type(array)):
            pass

        array.__class__ = Array
    elif reason == "dtype_subclass":
        class Dtype(pd.StringDtype):
            pass

        _backing(frame)._dtype = Dtype(storage="pyarrow", na_value=_dtype(missing_kind).na_value)
    elif reason == "array_storage":
        _backing(frame)._pa_array = pa.array(values, type=pa.large_string())
    elif reason == "binary_storage":
        _backing(frame)._pa_array = pa.chunked_array([pa.array([b"plain"] * 10000)])
    elif reason == "integer_storage":
        _backing(frame)._pa_array = pa.chunked_array([pa.array([1] * 10000)])
    elif reason == "short_storage":
        _backing(frame)._pa_array = pa.chunked_array([pa.array(values[:-1])])
    elif reason in ("invalid_utf8", "invalid_offsets"):
        offsets = np.arange(10001, dtype=np.int64)
        data = b"a" * 10000
        if reason == "invalid_utf8":
            data = data[:-1] + b"\xff"
        else:
            offsets[2] = 0
        invalid = pa.Array.from_buffers(pa.large_string(), 10000, [
            None, pa.py_buffer(offsets), pa.py_buffer(data),
        ])
        _backing(frame)._pa_array = pa.chunked_array([invalid])
    assert prepare_string_input(frame, schema) is None


@pytest.mark.parametrize("location", ["instance", "class"])
@pytest.mark.parametrize("method", [
    "astype", "isna", "__array__", "to_numpy", "_hasna", "fillna", "__getitem__", "__len__",
])
def test_arrow_string_custom_array_hooks_decline_without_calling(
    monkeypatch, missing_kind, location, method
):
    frame = _frame({"text": ["plain"] * 10000}, missing_kind)
    array = _backing(frame)
    calls = []

    def customized(*args, **kwargs):
        calls.append(method)
        raise AssertionError("eligibility must not invoke customized conversion hooks")

    if location == "instance":
        # _hasna is a data descriptor with no setter. Populate the instance
        # dictionary directly so the guard can reject every customized name.
        monkeypatch.setitem(vars(array), method, MethodType(customized, array))
    else:
        replacement = property(customized) if method == "_hasna" else customized
        monkeypatch.setattr(type(array), method, replacement)
    assert prepare_string_input(frame, _schema()) is None
    assert calls == []


@pytest.mark.parametrize("attribute", [
    "name", "na_value", "storage", "type", "kind", "__eq__", "__ne__", "__from_arrow__",
])
def test_arrow_string_custom_dtype_hooks_decline_without_calling(
    monkeypatch, missing_kind, attribute
):
    frame = _frame({"text": ["plain"] * 10000}, missing_kind)
    calls = []

    def customized(*args, **kwargs):
        calls.append(attribute)
        raise AssertionError("eligibility must not invoke customized dtype hooks")

    replacement = customized if attribute.startswith("__") else property(customized)
    # pandas 2 stores storage directly on each dtype; pandas 3 has a property.
    monkeypatch.setattr(pd.StringDtype, attribute, replacement, raising=False)
    assert prepare_string_input(frame, _schema()) is None
    assert calls == []


def test_arrow_string_custom_array_dtype_descriptor_declines(monkeypatch, missing_kind):
    frame = _frame({"text": ["plain"] * 10000}, missing_kind)
    array = _backing(frame)
    dtype = array.dtype
    # frame.dtypes already consults this descriptor in the established public
    # path. The contract here is to decline, rather than serialize raw storage.
    monkeypatch.setattr(type(array), "dtype", property(lambda self: dtype))
    assert prepare_string_input(frame, _schema()) is None


@pytest.mark.parametrize("location", [
    "manager_instance", "manager_class", "block_instance", "block_class",
])
def test_arrow_string_custom_storage_extraction_hooks_decline_without_calling(
    monkeypatch, missing_kind, location
):
    frame = _frame({"text": ["plain"] * 10000}, missing_kind)
    manager = vars(frame)["_mgr"]
    block = manager.blocks[0]
    calls = []

    def customized(*args, **kwargs):
        calls.append(location)
        raise AssertionError("eligibility must not invoke customized storage extraction")

    target = manager if location.startswith("manager_") else block
    method = "iget_values" if location.startswith("manager_") else "iget"
    if location.endswith("_instance"):
        monkeypatch.setitem(vars(target), method, MethodType(customized, target))
    else:
        monkeypatch.setattr(type(target), method, customized)
    assert prepare_string_input(frame, _schema()) is None
    assert calls == []


@pytest.mark.parametrize("cache_name", ["blknos", "blklocs"])
def test_arrow_string_public_cached_index_subclass_matches_native_row_control(
    protocol, monkeypatch, missing_kind, cache_name
):
    calls = []

    class CachedIndexError(Exception):
        pass

    class CustomIndexes(np.ndarray):
        def __getitem__(self, key):
            calls.append(cache_name)
            raise CachedIndexError("custom cached index must not be read during eligibility")

    values = ["plain", "\x00", "é", "🙂"] * 2500
    tables = []
    for row_control in (False, True):
        frame = _frame({"text": values}, missing_kind)
        _assert_backend(frame, missing_kind)
        manager = vars(frame)["_mgr"]
        cache = getattr(manager, cache_name)
        assert type(cache) is np.ndarray
        assert cache.dtype == np.dtype("intp")
        assert cache.shape == (1,)
        custom_cache = cache.view(CustomIndexes)
        setattr(manager, "_" + cache_name, custom_cache)
        assert type(getattr(manager, cache_name)) is CustomIndexes
        schema = _schema()
        calls.clear()
        # Both cached arrays normally feed iget_values through __getitem__.
        # Decline this subclass before indexing, while preserving ordinary row
        # normalization, which does not need to index the original caches.
        assert prepare_string_input(frame, schema) is None
        assert calls == []
        with monkeypatch.context() as control:
            if row_control:
                control.setattr(pandas_input, "prepare_string_input", lambda *args: None)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                spark.createDataFrame(frame, schema).toArrow()
                _assert_inline(state, schema, [values])
                tables.append(state["input_tables"][0])
        assert calls == []
        with pytest.raises(CachedIndexError, match="custom cached index must not be read"):
            custom_cache[0]
        assert calls == [cache_name]
    assert tables[0].equals(tables[1], check_metadata=True)


@pytest.mark.parametrize("target", ["manager", "block"])
def test_arrow_string_public_storage_class_alias_matches_native_row_control(
    protocol, monkeypatch, missing_kind, target
):
    calls = []

    class StorageError(Exception):
        pass

    errors = []
    for row_control in (False, True):
        with monkeypatch.context() as control:
            # Start with ordinary storage before rebinding the public internal
            # class. Matching the new alias must not authenticate a subclass.
            frame = _frame({"text": ["plain"] * 10000}, missing_kind)
            manager = vars(frame)["_mgr"]
            if target == "manager":
                class CustomManager(managers_module.BlockManager):
                    def apply(self, *args, **kwargs):
                        calls.append("manager.apply")
                        raise StorageError("manager.apply row sentinel")

                vars(frame)["_mgr"] = CustomManager(
                    manager.blocks, manager.axes, verify_integrity=False
                )
                control.setattr(managers_module, "BlockManager", CustomManager)
            else:
                class CustomBlock(blocks_module.ExtensionBlock):
                    def astype(self, *args, **kwargs):
                        calls.append("block.astype")
                        raise StorageError("block.astype row sentinel")

                block = manager.blocks[0]
                manager.blocks = (CustomBlock(
                    block.values, placement=block._mgr_locs, ndim=2, refs=block.refs
                ),)
                control.setattr(blocks_module, "ExtensionBlock", CustomBlock)
            schema = _schema()
            calls.clear()
            assert prepare_string_input(frame, schema) is None
            assert calls == []
            if row_control:
                control.setattr(pandas_input, "prepare_string_input", lambda *args: None)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                method = "manager.apply" if target == "manager" else "block.astype"
                with pytest.raises(StorageError,
                                   match=f"^{method} row sentinel$") as error:
                    spark.createDataFrame(frame, schema)
                errors.append((type(error.value), str(error.value)))
                assert calls == [method]
                assert state["config_requests"] == []
                assert state["uploads"] == 0
                assert state["relations"] == []
                assert state["input_tables"] == []
    assert errors[0] == errors[1]


@pytest.mark.parametrize("location", ["class_none", "instance"])
def test_arrow_string_custom_factory_declines_without_calling(
    monkeypatch, missing_kind, location
):
    frame = _frame({"text": ["plain"] * 10000}, missing_kind)
    array = _backing(frame)
    calls = []

    def factory(*args, **kwargs):
        calls.append("factory")
        raise AssertionError("eligibility must not invoke a customized Arrow factory")

    if location == "class_none":
        # pandas 3 normally defines this factory; pandas 2 must also decline an
        # explicitly installed None rather than mistake it for absence.
        monkeypatch.setattr(string_arrow_module.ArrowStringArray,
                            "_from_pyarrow_array", None, raising=False)
    else:
        monkeypatch.setitem(vars(array), "_from_pyarrow_array", MethodType(factory, array))
    assert prepare_string_input(frame, _schema()) is None
    assert calls == []


def test_arrow_string_deleted_factory_matches_pandas_layout(monkeypatch, missing_kind):
    frame = _frame({"text": ["plain"] * 10000}, missing_kind)
    requires_factory = type(vars(pd.StringDtype).get("storage")) is property
    monkeypatch.delattr(string_arrow_module.ArrowStringArray,
                        "_from_pyarrow_array", raising=False)
    result = prepare_string_input(frame, _schema())
    if requires_factory:
        # pandas 3's constructor and row conversion use this factory.
        assert result is None
    else:
        # pandas 2 ordinarily has no such factory; absence alone is not a hook.
        assert type(result) is bytes


def test_arrow_string_custom_numpy_semantics_alias_declines_without_calling(
    monkeypatch, missing_kind
):
    frame = _frame({"text": ["plain"] * 10000}, missing_kind)
    array = _backing(frame)
    calls = []

    class CustomNumpyStrings(string_arrow_module.ArrowStringArray):
        def __arrow_array__(self, *args, **kwargs):
            calls.append("arrow")
            raise AssertionError("eligibility must not invoke an aliased subclass")

    # pandas 3 has no NumPy-semantics alias; creating one must be as conservative
    # as rebinding the existing alias in pandas 2. Keep the actual custom type in
    # the input, so an identity check against the rebound alias alone is unsafe.
    array.__class__ = CustomNumpyStrings
    monkeypatch.setattr(string_arrow_module, "ArrowStringArrayNumpySemantics",
                        CustomNumpyStrings, raising=False)
    assert prepare_string_input(frame, _schema()) is None
    assert calls == []


def test_arrow_string_custom_public_array_alias_declines_without_calling(
    monkeypatch, missing_kind
):
    frame = _frame({"text": ["plain"] * 10000}, missing_kind)
    calls = []

    class CustomStrings(string_arrow_module.ArrowStringArray):
        def __arrow_array__(self, *args, **kwargs):
            calls.append("arrow")
            raise AssertionError("eligibility must not invoke a rebound public class")

    monkeypatch.setattr(pd.arrays, "ArrowStringArray", CustomStrings)
    assert prepare_string_input(frame, _schema()) is None
    assert calls == []


def test_arrow_string_public_fake_arrow_backing_matches_native_row_control(
    protocol, monkeypatch, missing_kind
):
    calls = []

    class RowBackingError(Exception):
        pass

    class CastBypassError(Exception):
        pass

    class FakeChunkedArray:
        def __init__(self, genuine):
            self.genuine = genuine

        @property
        def type(self):
            calls.append("type")
            return self.genuine.type

        @property
        def chunks(self):
            calls.append("chunks")
            return self.genuine.chunks

        @property
        def null_count(self):
            calls.append("null_count")
            return self.genuine.null_count

        def __len__(self):
            calls.append("len")
            return len(self.genuine)

        def validate(self, *args, **kwargs):
            calls.append("validate")
            return self.genuine.validate(*args, **kwargs)

        def to_numpy(self, *args, **kwargs):
            calls.append("to_numpy")
            raise RowBackingError("fake to_numpy row sentinel")

        def cast(self, *args, **kwargs):
            calls.append("cast")
            raise CastBypassError("fake cast bypass sentinel")

    errors, orders = [], []
    for row_control in (False, True):
        # Construct the ordinary array while all public Arrow aliases are still
        # genuine. Only its private backing and the public class alias change.
        frame = _frame({"text": ["plain"] * 10000}, missing_kind)
        array = _backing(frame)
        ordinary_array_class = type(array)
        fake = FakeChunkedArray(array._pa_array)
        array._pa_array = fake
        schema = _schema()
        calls.clear()
        with monkeypatch.context() as control:
            control.setattr(pa, "ChunkedArray", FakeChunkedArray)
            assert type(array) is ordinary_array_class
            assert prepare_string_input(frame, schema) is None
            assert calls == []
            if row_control:
                control.setattr(pandas_input, "prepare_string_input", lambda *args: None)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                with pytest.raises(RowBackingError,
                                   match="^fake to_numpy row sentinel$") as error:
                    spark.createDataFrame(frame, schema)
                errors.append((type(error.value), str(error.value)))
                orders.append(list(calls))
                assert calls[-1] == "to_numpy"
                assert "cast" not in calls
                assert state["config_requests"] == []
                assert state["uploads"] == 0
                assert state["relations"] == []
                assert state["input_tables"] == []
            # The tempting direct-storage conversion has a different observable
            # failure. It must never replace the public row sentinel above.
            with pytest.raises(CastBypassError, match="^fake cast bypass sentinel$"):
                fake.cast(pa.string(), safe=True)
    assert errors[0] == errors[1]
    assert orders[0] == orders[1]


def test_arrow_string_public_subclass_len_matches_native_row_control(
    protocol, monkeypatch, missing_kind
):
    frame = _frame({"text": ["plain"] * 10000}, missing_kind)
    array = _backing(frame)
    calls = []

    class LengthError(Exception):
        pass

    class CustomArray(type(array)):
        def __len__(self):
            calls.append("len")
            raise LengthError("subclass len before astype")

        def astype(self, *args, **kwargs):
            calls.append("astype")
            raise AssertionError("astype must follow the failing isna length check")

    array.__class__ = CustomArray
    schema = _schema()
    assert prepare_string_input(frame, schema) is None
    assert calls == []
    errors = []
    for row_control in (False, True):
        calls.clear()
        with monkeypatch.context() as control:
            if row_control:
                control.setattr(pandas_input, "prepare_string_input", lambda *args: None)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                with pytest.raises(LengthError,
                                   match="^subclass len before astype$") as error:
                    spark.createDataFrame(frame, schema)
                errors.append((type(error.value), str(error.value)))
                assert state["config_requests"] == []
                assert state["uploads"] == 0
                assert state["relations"] == []
                assert state["input_tables"] == []
        assert calls == ["len"]
    assert errors[0] == errors[1]


def test_arrow_string_public_invalid_utf8_matches_native_row_control(
    protocol, monkeypatch, missing_kind
):
    errors = []
    for row_control in (False, True):
        frame = _frame({"text": ["plain"] * 10000}, missing_kind)
        validity = bytearray([0xFF] * 1250)
        validity[0] = 0xFE
        invalid = pa.Array.from_buffers(pa.large_string(), 10000, [
            pa.py_buffer(validity), pa.py_buffer(np.arange(10001, dtype=np.int64)),
            pa.py_buffer(b"a" * 9999 + b"\xff"),
        ], null_count=1)
        _backing(frame)._pa_array = pa.chunked_array([invalid])
        schema = _schema(nullable=False)
        assert prepare_string_input(frame, schema) is None
        with monkeypatch.context() as control:
            if row_control:
                control.setattr(pandas_input, "prepare_string_input", lambda *args: None)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                # Normalization decodes the final invalid string before the
                # native row validator can reject the first, nonnullable null.
                with pytest.raises((UnicodeDecodeError, pa.ArrowException)) as error:
                    spark.createDataFrame(frame, schema)
                errors.append((type(error.value), str(error.value)))
                assert state["config_requests"] == []
                assert state["uploads"] == 0
                assert state["relations"] == []
                assert state["input_tables"] == []
    assert errors[0] == errors[1]


@pytest.mark.parametrize("target", ["array_astype", "array_isna", "array_to_numpy", "frame_isna"])
@pytest.mark.parametrize("behavior", ["mutate", "raise"])
def test_arrow_string_public_hooks_match_native_row_control(
    protocol, monkeypatch, missing_kind, target, behavior
):
    frame = _frame({"text": ["plain"] * 10000}, missing_kind)
    array = _backing(frame)
    method = target.removeprefix("array_").removeprefix("frame_")
    owner = frame if target == "frame_isna" else type(array)
    original = getattr(owner, method)
    calls = []
    expected = ["plain"] * 10000
    expected[0] = None if method == "isna" else "changed"
    # A serializer that reads only the original Arrow buffers cannot reproduce
    # these sentinels. This is the negative control for bypassing row hooks.
    original_storage = array._pa_array.to_pylist()
    assert original_storage != expected

    class HookError(Exception):
        pass

    def customized(*args, **kwargs):
        calls.append(target)
        if behavior == "raise":
            raise HookError(f"{target} sentinel")
        result = original(*args, **kwargs).copy()
        if target == "frame_isna":
            result.iat[0, 0] = True
        else:
            result[0] = True if method == "isna" else "changed"
        return result

    monkeypatch.setattr(owner, method, customized)
    schema = _schema()
    assert prepare_string_input(frame, schema) is None
    assert calls == []
    tables = []
    for row_control in (False, True):
        calls.clear()
        with monkeypatch.context() as control:
            if row_control:
                control.setattr(pandas_input, "prepare_string_input", lambda *args: None)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                if behavior == "raise":
                    with pytest.raises(HookError, match=f"^{target} sentinel$"):
                        spark.createDataFrame(frame, schema)
                    assert state["config_requests"] == []
                    assert state["uploads"] == 0
                    assert state["relations"] == []
                else:
                    spark.createDataFrame(frame, schema).toArrow()
                    _assert_inline(state, schema, [expected])
                    tables.append(state["input_tables"][0])
        assert calls == [target]
    if behavior == "mutate":
        assert tables[0].equals(tables[1], check_metadata=True)
        assert array._pa_array.to_pylist() == original_storage


def test_arrow_string_public_dtype_error_matches_native_row_control(
    protocol, monkeypatch, missing_kind
):
    frame = _frame({"text": ["plain"] * 10000}, missing_kind)
    array = _backing(frame)
    calls = []

    class DtypeError(Exception):
        pass

    def dtype(self):
        calls.append("dtype")
        raise DtypeError("dtype sentinel")

    monkeypatch.setattr(type(array), "dtype", property(dtype))
    # Eligibility and pandas normalization may consult array.dtype. Both public
    # calls must preserve its exact descriptor error before any RPC.
    for row_control in (False, True):
        calls.clear()
        with monkeypatch.context() as control:
            if row_control:
                control.setattr(pandas_input, "prepare_string_input", lambda *args: None)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                with pytest.raises(DtypeError, match="^dtype sentinel$"):
                    spark.createDataFrame(frame, _schema())
                assert state["config_requests"] == []
                assert state["uploads"] == 0
                assert state["relations"] == []
        assert calls == ["dtype"]


def test_arrow_string_public_dtype_na_value_error_matches_native_row_control(
    protocol, monkeypatch, missing_kind
):
    frame = _frame({"text": [None] + ["plain"] * 9999}, missing_kind)
    array = _backing(frame)
    schema = _schema()
    original_storage = array._pa_array
    original_storage.validate(full=True)
    assert original_storage.null_count == 1
    calls = []

    class MissingValueError(Exception):
        pass

    def na_value(self):
        calls.append("na_value")
        raise MissingValueError("StringDtype.na_value sentinel")

    monkeypatch.setattr(pd.StringDtype, "na_value", property(na_value))
    assert prepare_string_input(frame, schema) is None
    assert calls == []
    errors = []
    for row_control in (False, True):
        calls.clear()
        with monkeypatch.context() as control:
            if row_control:
                control.setattr(pandas_input, "prepare_string_input", lambda *args: None)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                with pytest.raises(MissingValueError,
                                   match=r"^StringDtype\.na_value sentinel$") as error:
                    spark.createDataFrame(frame, schema)
                errors.append((type(error.value), str(error.value)))
                assert state["config_requests"] == []
                assert state["uploads"] == 0
                assert state["relations"] == []
                assert state["input_tables"] == []
        assert calls == ["na_value"]
    assert errors[0] == errors[1]
    # A direct serializer of this still-valid storage would succeed and omit
    # the user-visible missing-value descriptor error proved above.
    assert array._pa_array is original_storage
    original_storage.validate(full=True)


def test_arrow_string_public_row_hook_error_order(protocol, monkeypatch, missing_kind):
    frame = _frame({"text": ["plain"] * 10000}, missing_kind)
    array = _backing(frame)
    calls = []

    class HookError(Exception):
        pass

    def isna(self):
        calls.append("isna")
        raise HookError("isna before astype")

    def astype(self, *args, **kwargs):
        calls.append("astype")
        raise HookError("astype must follow notna")

    monkeypatch.setattr(type(array), "isna", isna)
    monkeypatch.setattr(type(array), "astype", astype)
    assert prepare_string_input(frame, _schema()) is None
    assert calls == []
    with _arrow_input_service(protocol, config_error=True) as (spark, state):
        with pytest.raises(HookError, match="isna before astype"):
            spark.createDataFrame(frame, _schema())
        assert calls == ["isna"]
        assert state["config_requests"] == []
        assert state["uploads"] == 0
        assert state["relations"] == []


@pytest.mark.parametrize("reason", ["nonnullable_null", "cast", "width"])
@pytest.mark.parametrize("verify", [False, True])
def test_arrow_string_public_errors_precede_rpc(protocol, missing_kind, reason, verify):
    from pyspark.errors import PySparkRuntimeError, PySparkValueError

    values = ["plain"] * 10000
    if reason == "nonnullable_null":
        values[-1] = None
    frame = _frame({"text": values}, missing_kind)
    schema = _schema(nullable=reason != "nonnullable_null")
    error = PySparkRuntimeError
    match = "Column 'text' is declared as non-nullable but contains null values"
    if reason == "cast":
        schema = T.StructType([T.StructField("text", T.LongType())])
        match = "Type mismatch in row data"
    elif reason == "width":
        schema.add("extra", T.StringType())
        error = PySparkValueError
        match = "Row 0 has 1 values, but the schema has 2 fields"
    assert prepare_string_input(frame, schema) is None
    with _arrow_input_service(protocol, config_error=True) as (spark, state):
        with pytest.raises(error, match=match):
            spark.createDataFrame(frame, schema, verifySchema=verify)
        assert state["config_requests"] == []
        assert state["uploads"] == 0
        assert state["relations"] == []
