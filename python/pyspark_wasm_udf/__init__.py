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

The Rust client (``spark_connect::wasm_udf``) does not build the Spark
``PythonUDF`` command itself. Instead it invokes :mod:`pyspark_wasm_udf.pack`,
which uses ``cloudpickle`` to serialize a :class:`WasmScalarUDF` instance
**by value** into the command. Because the runner is embedded by value, the
Spark executors do **not** need this package installed -- they only need the
``wasmtime`` package to be importable.

Spark's Python worker unpickles the ``(WasmScalarUDF, return_type)`` tuple and
calls the instance once per input row (eval type ``SQL_ARROW_BATCHED_UDF``). On
first call the instance instantiates the embedded ``.wasm`` module with
``wasmtime`` and invokes the exported entrypoint.

Scope: numeric scalar signatures only. Value-type tags are ``"i32"``, ``"i64"``,
``"f32"`` and ``"f64"``. String/binary arguments need a memory-passing ABI and
are not handled yet.
"""

from typing import Any, List

__all__ = ["WasmScalarUDF"]

_INT_TAGS = ("i32", "i64")
_FLOAT_TAGS = ("f32", "f64")


class WasmScalarUDF:
    """A per-row callable that runs a WASM export via ``wasmtime``.

    Instances are serialized by value with cloudpickle, so all runtime state
    (the wasmtime store/function) is created lazily on the executor rather than
    captured at pickling time.
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
        # Runtime-only state; excluded from pickling (see __getstate__).
        self._store = None
        self._func = None

    # Never pickle the live wasmtime handles -- they are not picklable and are
    # rebuilt lazily on the executor.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_store"] = None
        state["_func"] = None
        return state

    def _ensure_instance(self) -> None:
        if self._func is not None:
            return
        try:
            import wasmtime
        except ImportError as exc:  # pragma: no cover - depends on executor env
            raise ImportError(
                "The 'wasmtime' package is required to run WASM UDFs on the "
                "Spark executors. Install it in the workers' Python env."
            ) from exc

        engine = wasmtime.Engine()
        module = wasmtime.Module(engine, self.wasm)
        self._store = wasmtime.Store(engine)
        instance = wasmtime.Instance(self._store, module, [])
        func = instance.exports(self._store).get(self.entrypoint)
        if func is None:
            raise ValueError(
                f"WASM module does not export a function named "
                f"'{self.entrypoint}'"
            )
        self._func = func

    @staticmethod
    def _coerce_in(value: Any, tag: str) -> Any:
        if value is None:
            # WASM has no null; substitute a zero of the right kind. UDFs that
            # must distinguish null should filter upstream.
            return 0 if tag in _INT_TAGS else 0.0
        if tag in _INT_TAGS:
            return int(value)
        if tag in _FLOAT_TAGS:
            return float(value)
        raise ValueError(f"unsupported WASM arg type tag: {tag!r}")

    def _coerce_out(self, value: Any) -> Any:
        if self.ret_type in _INT_TAGS:
            return int(value)
        if self.ret_type in _FLOAT_TAGS:
            return float(value)
        raise ValueError(f"unsupported WASM return type tag: {self.ret_type!r}")

    def __call__(self, *args: Any) -> Any:
        self._ensure_instance()
        if len(args) != len(self.arg_types):
            raise ValueError(
                f"WASM UDF '{self.entrypoint}' expected {len(self.arg_types)} "
                f"argument(s), got {len(args)}"
            )
        coerced = [
            self._coerce_in(v, tag) for v, tag in zip(args, self.arg_types)
        ]
        result = self._func(self._store, *coerced)
        return self._coerce_out(result)
