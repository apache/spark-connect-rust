#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Runtime for Rust UDFs compiled to WebAssembly and run on Spark.

The Rust client (``spark_connect::wasm_udf``) invokes :mod:`pyspark_wasm_udf.pack`,
which uses ``cloudpickle`` to serialize a :class:`WasmScalarUDF` **by value** into
a Spark ``PythonUDF`` command. Because the runner is embedded by value, the Spark
executors do **not** need this package installed -- only the ``wasmtime`` package.

Spark's Python worker unpickles the ``(WasmScalarUDF, return_type)`` tuple and
calls the instance once per input row. On first call the instance instantiates
the embedded ``.wasm`` module with ``wasmtime`` and invokes the exported
entrypoint.

The binary ABI (little-endian, matching ``spark_connect::wasm_udf::AbiType``):

* ``i32``/``f32``: 4 bytes; ``i64``/``f64``: 8 bytes; ``bool``: 1 byte.
* ``string``: ``u32`` byte-length then UTF-8; ``binary``: ``u32`` length then bytes.
* ``array:<T>``: ``u32`` count then each element; ``option:<T>``: 1 tag byte
  (0=None, 1=Some) then the value if present.
* Arguments are concatenated into one buffer; the result is a single value.

The module exports ``spark_udf_alloc(len)->ptr`` / ``spark_udf_dealloc(ptr,len)``
and each UDF as ``fn(args_ptr, args_len) -> (ptr << 32 | len)`` of the result.
"""

import struct
from typing import Any, List, Tuple

__all__ = ["WasmScalarUDF", "encode_args", "decode_value"]


# --- binary codec (the exact contract mirrored by the Rust runtime) ----------


def _split(desc: str) -> Tuple[str, str]:
    """Split ``"array:option:i64"`` into ``("array", "option:i64")``."""
    head, _, tail = desc.partition(":")
    return head, tail


def _encode(out: bytearray, desc: str, value: Any) -> None:
    kind, inner = _split(desc)
    if kind == "option":
        if value is None:
            out.append(0)
        else:
            out.append(1)
            _encode(out, inner, value)
        return
    if value is None:
        raise ValueError(
            f"null value for non-nullable WASM UDF argument of type {desc!r}; "
            f"use Option<...> in the Rust signature"
        )
    if kind == "i32":
        out.extend(struct.pack("<i", int(value)))
    elif kind == "i64":
        out.extend(struct.pack("<q", int(value)))
    elif kind == "f32":
        out.extend(struct.pack("<f", float(value)))
    elif kind == "f64":
        out.extend(struct.pack("<d", float(value)))
    elif kind == "bool":
        out.append(1 if value else 0)
    elif kind == "string":
        b = str(value).encode("utf-8")
        out.extend(struct.pack("<I", len(b)))
        out.extend(b)
    elif kind == "binary":
        b = bytes(value)
        out.extend(struct.pack("<I", len(b)))
        out.extend(b)
    elif kind == "array":
        out.extend(struct.pack("<I", len(value)))
        for elem in value:
            _encode(out, inner, elem)
    else:
        raise ValueError(f"unsupported WASM ABI type: {desc!r}")


def _decode(buf: bytes, off: int, desc: str) -> Tuple[Any, int]:
    kind, inner = _split(desc)
    if kind == "option":
        tag = buf[off]
        off += 1
        if tag == 0:
            return None, off
        return _decode(buf, off, inner)
    if kind == "i32":
        return struct.unpack_from("<i", buf, off)[0], off + 4
    if kind == "i64":
        return struct.unpack_from("<q", buf, off)[0], off + 8
    if kind == "f32":
        return struct.unpack_from("<f", buf, off)[0], off + 4
    if kind == "f64":
        return struct.unpack_from("<d", buf, off)[0], off + 8
    if kind == "bool":
        return (buf[off] != 0), off + 1
    if kind == "string":
        (n,) = struct.unpack_from("<I", buf, off)
        off += 4
        return buf[off : off + n].decode("utf-8"), off + n
    if kind == "binary":
        (n,) = struct.unpack_from("<I", buf, off)
        off += 4
        return bytes(buf[off : off + n]), off + n
    if kind == "array":
        (n,) = struct.unpack_from("<I", buf, off)
        off += 4
        items = []
        for _ in range(n):
            item, off = _decode(buf, off, inner)
            items.append(item)
        return items, off
    raise ValueError(f"unsupported WASM ABI type: {desc!r}")


def encode_args(arg_types: List[str], args: Tuple[Any, ...]) -> bytes:
    """Encode a row's arguments into the ABI byte buffer."""
    out = bytearray()
    for desc, value in zip(arg_types, args):
        _encode(out, desc, value)
    return bytes(out)


def decode_value(ret_type: str, buf: bytes) -> Any:
    """Decode a single ABI-encoded value (the result buffer)."""
    value, _ = _decode(buf, 0, ret_type)
    return value


# --- the picklable, per-row callable -----------------------------------------


class WasmScalarUDF:
    """A per-row callable that runs a WASM export via ``wasmtime``.

    Serialized by value with cloudpickle; all live wasmtime state is created
    lazily on the executor (never pickled).
    """

    def __init__(
        self,
        wasm: bytes,
        entrypoint: str,
        arg_types: List[str],
        ret_type: str,
    ) -> None:
        self.wasm = wasm
        self.entrypoint = entrypoint
        self.arg_types = list(arg_types)
        self.ret_type = ret_type
        self._rt = None  # lazily-built wasmtime state

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_rt"] = None
        return state

    def _ensure(self):
        if self._rt is not None:
            return self._rt
        try:
            import wasmtime
        except ImportError as exc:  # pragma: no cover - depends on executor env
            raise ImportError(
                "The 'wasmtime' package is required to run WASM UDFs on the "
                "Spark executors. Install it in the workers' Python env."
            ) from exc

        engine = wasmtime.Engine()
        module = wasmtime.Module(engine, self.wasm)
        store = wasmtime.Store(engine)
        instance = wasmtime.Instance(store, module, [])
        exports = instance.exports(store)

        def need(name):
            e = exports.get(name)
            if e is None:
                raise ValueError(f"WASM module does not export '{name}'")
            return e

        self._rt = {
            "store": store,
            "memory": need("memory"),
            "alloc": need("spark_udf_alloc"),
            "dealloc": need("spark_udf_dealloc"),
            "entry": need(self.entrypoint),
        }
        return self._rt

    def __call__(self, *args: Any) -> Any:
        if len(args) != len(self.arg_types):
            raise ValueError(
                f"WASM UDF '{self.entrypoint}' expected {len(self.arg_types)} "
                f"argument(s), got {len(args)}"
            )
        rt = self._ensure()
        store, memory = rt["store"], rt["memory"]
        alloc, dealloc, entry = rt["alloc"], rt["dealloc"], rt["entry"]

        buf = encode_args(self.arg_types, args)
        args_ptr = alloc(store, len(buf))
        if buf:
            memory.write(store, buf, args_ptr)

        packed = entry(store, args_ptr, len(buf)) & 0xFFFFFFFFFFFFFFFF
        res_ptr = (packed >> 32) & 0xFFFFFFFF
        res_len = packed & 0xFFFFFFFF

        res = bytes(memory.read(store, res_ptr, res_ptr + res_len))
        value = decode_value(self.ret_type, res)

        dealloc(store, args_ptr, len(buf))
        dealloc(store, res_ptr, res_len)
        return value
