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
"""The import hook of the vendored module that needs ``protobuf`` itself.

The vendored ``pyspark.pipelines.add_pipeline_analysis_context`` packs its message into
protobuf's own ``google.protobuf.any_pb2.Any``. When ``pyspark.sql.connect.proto`` uses the Rust
protobuf runtime (without ``protobuf``), this hook gives that module, as it is imported, the same
context manager of the runtime's ``Any`` (``pyspark.sql.connect.proto._rust``). The vendored file
itself stays as it is, and nothing stands in for the ``google.protobuf`` package.
"""
import importlib.abc
import sys
from collections.abc import Sequence
from typing import Any, Optional

_ANALYSIS_CONTEXT = "pyspark.pipelines.add_pipeline_analysis_context"


class _Loader(importlib.abc.Loader):
    """The loader of the module, which runs it and then gives it the runtime's context manager
    when the runtime is in use."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)

    def create_module(self, spec: Any) -> Any:
        return self._loader.create_module(spec)

    def exec_module(self, module: Any) -> None:
        self._loader.exec_module(module)
        from pyspark.sql.connect import proto

        if proto._implementation == "rust":
            from pyspark.sql.connect.proto._rust import add_pipeline_analysis_context

            # Named as the vendored function, so that it reports and pickles as that one.
            add_pipeline_analysis_context.__module__ = module.__name__
            module.add_pipeline_analysis_context = add_pipeline_analysis_context


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(
        self, fullname: str, path: Optional[Sequence[str]], target: Any = None
    ) -> Any:
        if fullname != _ANALYSIS_CONTEXT:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None:
                    spec.loader = _Loader(spec.loader)
                return spec
        return None


def install() -> None:
    """Add the hook to ``sys.meta_path``, once."""
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
