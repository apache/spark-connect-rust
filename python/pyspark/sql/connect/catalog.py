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

# The Catalog of Spark Connect, as PySpark Connect's ``pyspark.sql.connect.catalog``: the native
# Catalog of the Rust core, whose methods return the NamedTuples of PySpark's classic
# ``pyspark.sql.catalog`` (Database, Table, Column, Function, CatalogMetadata and TablePartition),
# which PySpark Connect's module imports too.
from pyspark._pyspark import Catalog
from pyspark.sql.catalog import (
    CatalogMetadata,
    Column,
    Database,
    Function,
    Table,
    TablePartition,
)

# Of PySpark Connect's pyspark.sql.connect.catalog, whose namespace the official doctests of the
# Catalog take as their globals, and whose examples use it (Catalog.cacheTable).
from pyspark.storagelevel import StorageLevel  # noqa: F401

__all__ = ["Catalog", "CatalogMetadata", "Column", "Database", "Function", "Table", "TablePartition"]
