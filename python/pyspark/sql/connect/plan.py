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
"""Logical plans under the connect import path.

The Rust core builds the plans of the DataFrame API. ``LogicalPlan`` is PySpark Connect's, from
Apache Spark v4.3.0, the base of the plans that the modules of PySpark build themselves, such as
those of ``pyspark.ml.connect.proto``: a plan of a child, of a ``plan(session)`` that is its
``spark.connect.Relation``, which ``DataFrame(plan, session)`` sends as it is. It has no
``to_proto``, ``_collect_references``, ``print`` or ``_repr_html_``, which are of the plans and the
Columns of PySpark Connect's DataFrame, which the core builds. ``_DataFramePlan`` is the plan of a
DataFrame of the core, its ``DataFrame._plan``. ``Read``, ``Range``, ``SQL``, ``SubqueryAlias``,
``CachedRemoteRelation`` and ``RemoveRemoteCachedRelation`` are PySpark Connect's, which
``pyspark.testing.connectutils`` and ``pyspark.sql.connect.conversion`` build DataFrames of. The
messages are those of the generated ``pyspark.sql.connect.proto``, which needs the ``protobuf``
package.
"""

from collections.abc import Sequence
from threading import Lock
from typing import Any, Optional

import pyspark.sql.connect.proto as proto
from pyspark.sql.column import Column
from pyspark.sql.connect.logging import logger


class LogicalPlan:
    _lock: Lock = Lock()
    # The plan IDs of these plans start past those that the Rust core gives the relations of a
    # session (from 1), so that the two never number a relation the same.
    _nextPlanId: int = 1 << 48

    def __init__(
        self, child: Optional["LogicalPlan"], references: Optional[Sequence["LogicalPlan"]] = None
    ) -> None:
        """

        Parameters
        ----------
        child : :class:`LogicalPlan`, optional.
            The child logical plan.
        references : list of :class:`LogicalPlan`, optional.
            The list of logical plans that are referenced as subqueries in this logical plan.
        """
        self._child = child
        self._root_plan_id = LogicalPlan._fresh_plan_id()

        self._references: Sequence[LogicalPlan] = references or []
        self._plan_id_with_rel: Optional[int] = None
        if len(self._references) > 0:
            assert all(isinstance(r, LogicalPlan) for r in self._references)
            self._plan_id_with_rel = LogicalPlan._fresh_plan_id()

    @property
    def _plan_id(self) -> int:
        return self._plan_id_with_rel or self._root_plan_id

    @staticmethod
    def _fresh_plan_id() -> int:
        plan_id: Optional[int] = None
        with LogicalPlan._lock:
            plan_id = LogicalPlan._nextPlanId
            LogicalPlan._nextPlanId += 1

        assert plan_id is not None
        return plan_id

    def _create_proto_relation(self) -> "proto.Relation":
        plan = proto.Relation()
        plan.common.plan_id = self._root_plan_id
        return plan

    def plan(self, session: Any) -> "proto.Relation":  # type: ignore[empty-body]
        ...

    def command(self, session: Any) -> "proto.Command":  # type: ignore[empty-body]
        ...

    def _verify(self, session: Any) -> bool:
        """This method is used to verify that the current logical plan
        can be serialized to Proto and back and afterwards is identical."""
        plan = proto.Plan()
        plan.root.CopyFrom(self.plan(session))

        serialized_plan = plan.SerializeToString()
        test_plan = proto.Plan()
        test_plan.ParseFromString(serialized_plan)

        return test_plan == plan

    @property
    def observations(self) -> dict[str, Any]:
        if self._child is None:
            return {}
        else:
            return self._child.observations

    def _with_relations(self, root: "proto.Relation", session: Any) -> "proto.Relation":
        if len(self._references) == 0:
            return root
        else:
            # When there are references to other DataFrame, e.g., subqueries, build new plan like:
            # with_relations [id 10]
            #     root: plan  [id 9]
            #     reference:
            #          refs#1: [id 8]
            #          refs#2: [id 5]
            plan = proto.Relation()
            assert isinstance(self._plan_id_with_rel, int)
            plan.common.plan_id = self._plan_id_with_rel
            plan.with_relations.root.CopyFrom(root)
            plan.with_relations.references.extend([ref.plan(session) for ref in self._references])
            return plan


class _DataFramePlan(LogicalPlan):
    """The plan of a DataFrame of the Rust core, as its ``DataFrame._plan``: its relation, as the
    core sends it, of the plan IDs of its session.

    Its ``plan`` is a placeholder of that relation, which the core replaces by the relation when
    the plan built over it reaches the core, as ``DataFrame(plan, session)`` or a command does.
    So a plan of ``pyspark.ml`` over a deep DataFrame is never decoded in Python or in Rust, as
    protobuf decoders limit the depth of what they decode."""

    def __init__(self, df: Any) -> None:
        super().__init__(None)
        self._df = df

    def plan(self, session: Any) -> "proto.Relation":
        return proto.Relation.FromString(self._df._relation_placeholder())


class Read(LogicalPlan):
    def __init__(
        self,
        table_name: str,
        options: Optional[dict[str, str]] = None,
        is_streaming: Optional[bool] = None,
    ) -> None:
        super().__init__(None)
        self.table_name = table_name
        self.options = options or {}
        self._is_streaming = is_streaming

    def plan(self, session: Any) -> "proto.Relation":
        plan = self._create_proto_relation()
        plan.read.named_table.unparsed_identifier = self.table_name
        if self._is_streaming is not None:
            plan.read.is_streaming = self._is_streaming
        for k, v in self.options.items():
            plan.read.named_table.options[k] = v
        return plan

    def print(self, indent: int = 0) -> str:
        return f"{' ' * indent}<Read table_name={self.table_name}>\n"


class CachedRemoteRelation(LogicalPlan):
    """Logical plan object for a DataFrame reference which represents a DataFrame that's been
    cached on the server with a given id."""

    def __init__(self, relation_id: str, spark_session: Any):
        super().__init__(None)
        self._relation_id = relation_id
        # Needs to hold the session to make a request itself.
        self._spark_session = spark_session

    def plan(self, session: Any) -> "proto.Relation":
        plan = self._create_proto_relation()
        plan.cached_remote_relation.relation_id = self._relation_id
        return plan

    def __del__(self) -> None:
        session = self._spark_session
        # If session is already closed, all cached DataFrame should be released.
        if session is not None and not session.client.is_closed and self._relation_id is not None:
            try:
                # The command, as PySpark Connect's sends it, whose responses are ignored: they
                # are not converted, as that of a checkpoint is into another cached relation.
                command = RemoveRemoteCachedRelation(self).command(session=session.client)
                session._execute_command(command.SerializeToString())
            except Exception as e:
                logger.warning(f"RemoveRemoteCachedRelation failed with exception: {e}.")


class SubqueryAlias(LogicalPlan):
    """Alias for a relation."""

    def __init__(self, child: Optional["LogicalPlan"], alias: str) -> None:
        super().__init__(child)
        self._alias = alias

    def plan(self, session: Any) -> "proto.Relation":
        plan = self._create_proto_relation()
        if self._child is not None:
            plan.subquery_alias.input.CopyFrom(self._child.plan(session))
        plan.subquery_alias.alias = self._alias
        return plan


class SQL(LogicalPlan):
    def __init__(
        self,
        query: str,
        args: Optional[list[Column]] = None,
        named_args: Optional[dict[str, Column]] = None,
        views: Optional[Sequence[SubqueryAlias]] = None,
    ) -> None:
        if args is not None:
            assert isinstance(args, list)
            assert all(isinstance(arg, Column) for arg in args)

        if named_args is not None:
            assert isinstance(named_args, dict)
            for k, arg in named_args.items():
                assert isinstance(k, str)
                assert isinstance(arg, Column)

        if views is not None:
            assert isinstance(views, list)
            assert all(isinstance(v, SubqueryAlias) for v in views)

        super().__init__(None, views)
        self._query = query
        self._args = args
        self._named_args = named_args
        self._views = views

    def plan(self, session: Any) -> "proto.Relation":
        plan = self._create_proto_relation()
        plan.sql.query = self._query

        if self._args is not None and len(self._args) > 0:
            plan.sql.pos_arguments.extend([arg.to_plan(session) for arg in self._args])
        if self._named_args is not None and len(self._named_args) > 0:
            for k, arg in self._named_args.items():
                plan.sql.named_arguments[k].CopyFrom(arg.to_plan(session))

        return self._with_relations(plan, session)

    def command(self, session: Any) -> "proto.Command":
        cmd = proto.Command()
        cmd.sql_command.input.CopyFrom(self.plan(session))
        return cmd


class Range(LogicalPlan):
    def __init__(
        self,
        start: int,
        end: int,
        step: int,
        num_partitions: Optional[int] = None,
    ) -> None:
        super().__init__(None)
        self._start = start
        self._end = end
        self._step = step
        self._num_partitions = num_partitions

    def plan(self, session: Any) -> "proto.Relation":
        plan = self._create_proto_relation()
        plan.range.start = self._start
        plan.range.end = self._end
        plan.range.step = self._step
        if self._num_partitions is not None:
            plan.range.num_partitions = self._num_partitions
        return plan


class RemoveRemoteCachedRelation(LogicalPlan):
    def __init__(self, relation: CachedRemoteRelation) -> None:
        super().__init__(None)
        self._relation = relation

    def command(self, session: Any) -> "proto.Command":
        plan = self._create_proto_relation()
        plan.cached_remote_relation.relation_id = self._relation._relation_id
        cmd = proto.Command()
        cmd.remove_cached_remote_relation_command.relation.CopyFrom(plan.cached_remote_relation)
        return cmd


__all__ = ["LogicalPlan", "Read", "Range", "SQL", "SubqueryAlias", "CachedRemoteRelation"]
