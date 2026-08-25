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
from threading import RLock
from typing import Dict, Union, Optional

from pyspark.resource.requests import (
    TaskResourceRequest,
    TaskResourceRequests,
    ExecutorResourceRequests,
    ExecutorResourceRequest,
)


class ResourceProfile:
    """
    Resource profile to associate with an RDD. A :class:`pyspark.resource.ResourceProfile`
    allows the user to specify executor and task requirements for an RDD that will get
    applied during a stage. This allows the user to change the resource requirements between
    stages. This is meant to be immutable so user cannot change it after building.

    .. versionadded:: 3.1.0

    .. versionchanged:: 4.0.0
        Supports Spark Connect.

    Notes
    -----
    This API is evolving.

    Examples
    --------
    Create Executor resource requests.

    >>> executor_requests = (
    ...     ExecutorResourceRequests()
    ...     .cores(2)
    ...     .memory("6g")
    ...     .memoryOverhead("1g")
    ...     .pysparkMemory("2g")
    ...     .offheapMemory("3g")
    ...     .resource("gpu", 2, "testGpus", "nvidia.com")
    ... )

    Create task resource requasts.

    >>> task_requests = TaskResourceRequests().cpus(2).resource("gpu", 2)

    Create a resource profile.

    >>> builder = ResourceProfileBuilder()
    >>> resource_profile = builder.require(executor_requests).require(task_requests).build

    Create an RDD with the resource profile.

    >>> rdd = sc.parallelize(range(10)).withResources(resource_profile)
    >>> rdd.getResourceProfile()
    <pyspark.resource.profile.ResourceProfile object ...>
    >>> rdd.getResourceProfile().taskResources
    {'cpus': <...TaskResourceRequest...>, 'gpu': <...TaskResourceRequest...>}
    >>> rdd.getResourceProfile().executorResources
    {'gpu': <...ExecutorResourceRequest...>,
     'cores': <...ExecutorResourceRequest...>,
     'offHeap': <...ExecutorResourceRequest...>,
     'memoryOverhead': <...ExecutorResourceRequest...>,
     'pyspark.memory': <...ExecutorResourceRequest...>,
     'memory': <...ExecutorResourceRequest...>}
    """

    def __init__(
        self,
        _exec_req: Optional[Dict[str, ExecutorResourceRequest]] = None,
        _task_req: Optional[Dict[str, TaskResourceRequest]] = None,
    ):
        # profile id
        self._id: Optional[int] = None
        # lock to protect _id
        self._lock = RLock()
        self._executor_resource_requests = _exec_req or {}
        self._task_resource_requests = _task_req or {}

    @property
    def id(self) -> int:
        """
        Returns
        -------
        int
            A unique id of this :class:`ResourceProfile`
        """
        with self._lock:
            if self._id is None:
                from pyspark._pyspark import (
                    ExecutorResourceRequests as _ER,
                    TaskResourceRequests as _TR,
                    ResourceProfileBuilder as _RPB,
                )
                from pyspark.sql import SparkSession

                session = SparkSession.getActiveSession()
                if session is None:
                    raise RuntimeError(
                        "An active SparkSession is required to get the profile id."
                    )

                er = _ER()
                for name, req in self._executor_resource_requests.items():
                    er = er.resource(
                        name,
                        int(req.amount),
                        req.discoveryScript or None,
                        req.vendor or None,
                    )

                tr = _TR()
                for name, req in self._task_resource_requests.items():
                    tr = tr.resource(name, float(req.amount))

                profile = _RPB().executor_resources(er).task_resources(tr).build()
                self._id = session.buildResourceProfile(profile)

            return self._id

    @property
    def taskResources(self) -> Dict[str, TaskResourceRequest]:
        """
        Returns
        -------
        dict
            a dictionary of resources to :class:`TaskResourceRequest`
        """
        return self._task_resource_requests

    @property
    def executorResources(self) -> Dict[str, ExecutorResourceRequest]:
        """
        Returns
        -------
        dict
            a dictionary of resources to :class:`ExecutorResourceRequest`
        """
        return self._executor_resource_requests


class ResourceProfileBuilder:
    """
    Resource profile Builder to build a resource profile to associate with an RDD.
    A ResourceProfile allows the user to specify executor and task requirements for
    an RDD that will get applied during a stage. This allows the user to change the
    resource requirements between stages.

    .. versionadded:: 3.1.0

    See Also
    --------
    :class:`pyspark.resource.ResourceProfile`

    Notes
    -----
    This API is evolving.
    """

    def __init__(self) -> None:
        self._executor_resource_requests: Dict[str, ExecutorResourceRequest] = {}
        self._task_resource_requests: Dict[str, TaskResourceRequest] = {}

    def require(
        self, resourceRequest: Union[ExecutorResourceRequests, TaskResourceRequests]
    ) -> "ResourceProfileBuilder":
        """
        Add executor resource requests

        Parameters
        ----------
        resourceRequest : :class:`ExecutorResourceRequests` or :class:`TaskResourceRequests`
            The detailed executor resource requests, see :class:`ExecutorResourceRequests`

        Returns
        -------
        dict
            a dictionary of resources to :class:`ExecutorResourceRequest`
        """

        if isinstance(resourceRequest, TaskResourceRequests):
            self._task_resource_requests.update(resourceRequest.requests)
        else:
            self._executor_resource_requests.update(resourceRequest.requests)
        return self

    def clearExecutorResourceRequests(self) -> None:
        self._executor_resource_requests = {}

    def clearTaskResourceRequests(self) -> None:
        self._task_resource_requests = {}

    @property
    def taskResources(self) -> Dict[str, TaskResourceRequest]:
        """
        Returns
        -------
        dict
            a dictionary of resources to :class:`TaskResourceRequest`
        """
        return self._task_resource_requests

    @property
    def executorResources(self) -> Dict[str, ExecutorResourceRequest]:
        """
        Returns
        -------
        dict
            a dictionary of resources to :class:`ExecutorResourceRequest`
        """
        return self._executor_resource_requests

    @property
    def build(self) -> ResourceProfile:
        return ResourceProfile(
            _exec_req=self._executor_resource_requests, _task_req=self._task_resource_requests
        )
