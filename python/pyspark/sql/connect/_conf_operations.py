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
"""The operations of the requests of PySpark Connect's `RuntimeConf`, for a client of another
class than this client's.

A `RuntimeConf` keeps its client as its `_client`, as PySpark's does, and makes each request of
it: the native transport of a client of this client, or else `client.config(operation)`, as
PySpark's methods call it, of the `ConfigRequest.Operation` that `config_operation` builds as they
build it (crates/pyspark-rs/src/conf.rs).
"""

from typing import Any, Callable

import pyspark.sql.connect.proto as proto


def _set(key: str, value: Any) -> Any:
    op_set = proto.ConfigRequest.Set(pairs=[proto.KeyValue(key=key, value=value)])
    return proto.ConfigRequest.Operation(set=op_set)


def _get(key: str) -> Any:
    op_get = proto.ConfigRequest.Get(keys=[key])
    return proto.ConfigRequest.Operation(get=op_get)


def _get_with_default(key: str, default: Any) -> Any:
    op_get_with_default = proto.ConfigRequest.GetWithDefault(
        pairs=[proto.KeyValue(key=key, value=default)]
    )
    return proto.ConfigRequest.Operation(get_with_default=op_get_with_default)


def _get_all() -> Any:
    op_get_all = proto.ConfigRequest.GetAll()
    return proto.ConfigRequest.Operation(get_all=op_get_all)


def _unset(key: str) -> Any:
    op_unset = proto.ConfigRequest.Unset(keys=[key])
    return proto.ConfigRequest.Operation(unset=op_unset)


def _is_modifiable(key: str) -> Any:
    op_is_modifiable = proto.ConfigRequest.IsModifiable(keys=[key])
    return proto.ConfigRequest.Operation(is_modifiable=op_is_modifiable)


_OPERATIONS = {
    "set": _set,
    "get": _get,
    "get_with_default": _get_with_default,
    "get_all": _get_all,
    "unset": _unset,
    "is_modifiable": _is_modifiable,
}


def config_operation(kind: str) -> Callable[..., Any]:
    """The builder of the `ConfigRequest.Operation` of PySpark's `RuntimeConf` method `kind`, of
    its arguments as the method converts them."""
    return _OPERATIONS[kind]
