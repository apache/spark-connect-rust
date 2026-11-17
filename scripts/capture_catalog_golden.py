#!/usr/bin/env python3
"""Capture golden Spark Connect protobuf catalog operations from the real PySpark.

For a curated set of Catalog calls, this records the plans that the reference PySpark Connect
client sends, then normalizes out run-to-run noise (plan_id counters and Python
`origin`/call-site metadata). Every Catalog method of PySpark Connect runs its catalog relation at
once, also the ones that return lists, and `listViews` of a pattern and no database runs
`currentDatabase` first: so a case is the plans that its call sends, in order. The result is a
stable golden file the Rust catalog builders are tested against for byte-level parity.

Usage:
    python3 scripts/capture_catalog_golden.py --remote sc://localhost:15002 \
        --out tests/golden

Requires a live Spark Connect server, which runs the calls. A plan is recorded as it is sent,
whatever the server answers, so the calls that name a table that does not exist count too. The
tables and the database that the calls create are dropped by the later calls.
"""

from __future__ import annotations

import argparse
import base64
import json
import warnings
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.connect.client.core import SparkConnectClient
from pyspark.sql.types import LongType, StringType, StructField, StructType
from pyspark.storagelevel import StorageLevel


def normalize(msg) -> None:
    """Recursively clear non-deterministic fields in a protobuf message in place.

    - ``plan_id`` (common.plan_id): a session-scoped counter.
    - ``origin`` / ``source_info``: Python source location metadata that a
      non-Python client will not reproduce identically.
    """
    from google.protobuf.message import Message

    for field, value in list(msg.ListFields()):
        if field.name in ("plan_id", "origin", "source_info"):
            msg.ClearField(field.name)
            continue
        if field.type == field.TYPE_MESSAGE:
            # A singular field is a message; a repeated one or a map is a container of them. The
            # value tells which, as protobuf 7 has no `FieldDescriptor.label`.
            if isinstance(value, Message):
                normalize(value)
            elif field.message_type.GetOptions().map_entry:
                # map<...>: values may be messages
                for k in value:
                    v = value[k]
                    if isinstance(v, Message):
                        normalize(v)
            else:
                for item in value:
                    if isinstance(item, Message):
                        normalize(item)


# The plans that the client sends while a call is recorded, or None between the calls.
_SENT: list | None = None


def _record_sent_plans() -> None:
    """Wrap the execute path of the client, which every catalog call takes, to record its plans."""
    original = SparkConnectClient._execute_and_fetch_as_iterator

    def recorded(self, req, *args, **kwargs):
        if _SENT is not None:
            plan = type(req.plan)()
            plan.CopyFrom(req.plan)
            _SENT.append(plan)
        return original(self, req, *args, **kwargs)

    SparkConnectClient._execute_and_fetch_as_iterator = recorded


def cap_catalog_call(spark, name, call):
    """Run a catalog call and capture the plans it sends, whatever it returns or raises."""
    global _SENT
    _SENT = []
    try:
        with warnings.catch_warnings():
            # The deprecated forms (dbName, createExternalTable) warn.
            warnings.simplefilter("ignore", FutureWarning)
            call(spark.catalog)
    except Exception as e:  # noqa: BLE001 - the server's answer is not part of the golden
        print(f"note: {name} raised {type(e).__name__}; its plans are recorded")
    sent, _SENT = _SENT, None
    if not sent:
        raise AssertionError(f"{name} sent no plan")
    for plan in sent:
        normalize(plan)
    return dict(
        name=name,
        kind="catalog",
        b64=[base64.b64encode(p.SerializeToString(deterministic=True)).decode() for p in sent],
        text=[str(p) for p in sent],
    )


# The table, view, function and database that the calls name. The table is created and dropped
# by the calls; the others do not exist.
TABLE = "golden_tbl"
VIEW = "golden_view"
FUNCTION = "golden_fn"
DATABASE = "golden_db"
PATH = "/tmp/spark-connect-rust-golden-catalog"
SCHEMA = StructType([StructField("a", LongType()), StructField("b", StringType())])


def catalog_cases():
    """Yield (name, call) for the catalog calls, each called with the session's Catalog."""
    yield "current_catalog", lambda c: c.currentCatalog()
    yield "set_current_catalog", lambda c: c.setCurrentCatalog("spark_catalog")
    yield "list_catalogs", lambda c: c.listCatalogs()
    yield "list_catalogs_pattern", lambda c: c.listCatalogs("spark*")
    yield "current_database", lambda c: c.currentDatabase()
    yield "set_current_database", lambda c: c.setCurrentDatabase("default")
    yield "list_databases", lambda c: c.listDatabases()
    yield "list_databases_pattern", lambda c: c.listDatabases("def*")
    yield "get_database", lambda c: c.getDatabase("default")
    yield "database_exists", lambda c: c.databaseExists("default")
    yield "create_database", lambda c: c.createDatabase(DATABASE, True, {"k": "v"})
    yield "create_database_defaults", lambda c: c.createDatabase(DATABASE)
    yield "drop_database", lambda c: c.dropDatabase(DATABASE, ifExists=True, cascade=True)
    yield "drop_database_defaults", lambda c: c.dropDatabase(DATABASE)
    yield (
        "create_table",
        lambda c: c.createTable(
            TABLE, source="parquet", schema=SCHEMA, description="a table", compression="snappy"
        ),
    )
    yield "create_table_path", lambda c: c.createTable(TABLE, PATH)
    # createExternalTable creates the table as createTable does, of a `description` option bound
    # to createTable's parameter.
    yield (
        "create_external_table",
        lambda c: c.createExternalTable(TABLE, PATH, "parquet", SCHEMA, compression="snappy"),
    )
    yield (
        "create_external_table_description",
        lambda c: c.createExternalTable(TABLE, source="parquet", description="d", k="v"),
    )
    yield "list_tables", lambda c: c.listTables()
    yield "list_tables_db", lambda c: c.listTables("default")
    yield "list_tables_pattern", lambda c: c.listTables(pattern="golden*")
    yield "list_tables_db_pattern", lambda c: c.listTables("default", "golden*")
    yield "get_table", lambda c: c.getTable(TABLE)
    yield "table_exists", lambda c: c.tableExists(TABLE)
    yield "table_exists_db", lambda c: c.tableExists(TABLE, "default")
    yield "list_columns", lambda c: c.listColumns(TABLE)
    yield "list_columns_db", lambda c: c.listColumns(TABLE, "default")
    yield "list_partitions", lambda c: c.listPartitions(TABLE)
    yield "get_table_properties", lambda c: c.getTableProperties(TABLE)
    yield "get_create_table_string", lambda c: c.getCreateTableString(TABLE)
    yield "get_create_table_string_serde", lambda c: c.getCreateTableString(TABLE, asSerde=True)
    yield "analyze_table", lambda c: c.analyzeTable(TABLE)
    yield "analyze_table_no_scan", lambda c: c.analyzeTable(TABLE, noScan=True)
    yield "is_cached", lambda c: c.isCached(TABLE)
    yield "cache_table", lambda c: c.cacheTable(TABLE)
    yield "cache_table_storage_level", lambda c: c.cacheTable(TABLE, StorageLevel.DISK_ONLY)
    yield "uncache_table", lambda c: c.uncacheTable(TABLE)
    yield "clear_cache", lambda c: c.clearCache()
    yield "refresh_table", lambda c: c.refreshTable(TABLE)
    yield "refresh_by_path", lambda c: c.refreshByPath(PATH)
    yield "recover_partitions", lambda c: c.recoverPartitions(TABLE)
    yield "truncate_table", lambda c: c.truncateTable(TABLE)
    yield "drop_table", lambda c: c.dropTable(TABLE, ifExists=True, purge=True)
    yield "drop_table_defaults", lambda c: c.dropTable(TABLE)
    yield "list_functions", lambda c: c.listFunctions()
    yield "list_functions_db", lambda c: c.listFunctions("default")
    yield "list_functions_pattern", lambda c: c.listFunctions(pattern="golden*")
    yield "list_functions_db_pattern", lambda c: c.listFunctions("default", "golden*")
    yield "get_function", lambda c: c.getFunction(FUNCTION)
    yield "function_exists", lambda c: c.functionExists(FUNCTION)
    yield "function_exists_db", lambda c: c.functionExists(FUNCTION, "default")
    yield "list_views", lambda c: c.listViews()
    yield "list_views_db", lambda c: c.listViews("default")
    yield "list_views_db_pattern", lambda c: c.listViews("default", "golden*")
    # A pattern of no database: PySpark runs currentDatabase first, and lists the views of it.
    yield "list_views_pattern", lambda c: c.listViews(pattern="golden*")
    yield "drop_temp_view", lambda c: c.dropTempView(VIEW)
    yield "drop_global_temp_view", lambda c: c.dropGlobalTempView(VIEW)
    yield "drop_view", lambda c: c.dropView(VIEW, ifExists=True)
    yield "drop_view_defaults", lambda c: c.dropView(VIEW)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default="sc://localhost:15002")
    ap.add_argument("--out", default="tests/golden")
    args = ap.parse_args()

    spark = SparkSession.builder.remote(args.remote).getOrCreate()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _record_sent_plans()
    plans = [cap_catalog_call(spark, name, call) for name, call in catalog_cases()]

    (out / "catalog.jsonl").write_text("\n".join(json.dumps(p) for p in plans) + "\n")
    print(f"wrote {len(plans)} catalog operations -> {out / 'catalog.jsonl'}")
    spark.stop()


if __name__ == "__main__":
    main()
