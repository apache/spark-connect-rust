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
DataFrame of the core, its ``DataFrame._plan``. ``Read``, ``Range`` and ``SQL`` are not plans of
this client. The messages are those of the generated ``pyspark.sql.connect.proto``, which needs
the ``protobuf`` package.
"""

from threading import Lock
from typing import Any, Dict, Optional, Sequence

import pyspark.sql.connect.proto as proto


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

        self._references: Sequence["LogicalPlan"] = references or []
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
    def observations(self) -> Dict[str, Any]:
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
    core sends it, of the plan IDs of its session."""

    def __init__(self, df: Any) -> None:
        super().__init__(None)
        self._df = df

    def plan(self, session: Any) -> "proto.Relation":
        return proto.Relation.FromString(self._df._relation_bytes())


class Read(LogicalPlan):
    """Not a plan of this client: the Rust core reads tables, as ``spark.read.table``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class Range(LogicalPlan):
    """Not a plan of this client: the Rust core builds ranges, as ``spark.range``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class SQL(LogicalPlan):
    """Not a plan of this client: the Rust core builds queries, as ``spark.sql``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


__all__ = ["LogicalPlan", "Read", "Range", "SQL"]
