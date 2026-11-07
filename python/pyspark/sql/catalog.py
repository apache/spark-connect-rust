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
"""Catalog + its metadata result classes, mirroring ``pyspark.sql.catalog``.

The Catalog is implemented natively in the Rust core and re-exported here. Its methods return
the NamedTuples of PySpark's ``pyspark.sql.catalog`` below (Database, Table, Column, Function,
CatalogMetadata and TablePartition), which are tuples, as PySpark's are."""

from typing import List, NamedTuple, Optional

from pyspark._pyspark import Catalog


class CatalogMetadata(NamedTuple):
    name: str
    description: Optional[str]


class Database(NamedTuple):
    name: str
    catalog: Optional[str]
    description: Optional[str]
    locationUri: str


class Table(NamedTuple):
    name: str
    catalog: Optional[str]
    namespace: Optional[List[str]]
    description: Optional[str]
    tableType: str
    isTemporary: bool

    @property
    def database(self) -> Optional[str]:
        if self.namespace is not None and len(self.namespace) == 1:
            return self.namespace[0]
        else:
            return None


class Column(NamedTuple):
    name: str
    description: Optional[str]
    dataType: str
    nullable: bool
    isPartition: bool
    isBucket: bool
    isCluster: bool


class Function(NamedTuple):
    name: str
    catalog: Optional[str]
    namespace: Optional[List[str]]
    description: Optional[str]
    className: str
    isTemporary: bool


class TablePartition(NamedTuple):
    partition: str


__all__ = [
    "Catalog",
    "CatalogMetadata",
    "Database",
    "Table",
    "Column",
    "Function",
    "TablePartition",
]
