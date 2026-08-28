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

"""
User-defined function (UDF) support for Spark Connect client.
Mirrors pyspark.sql.connect.udf and pyspark.sql.functions.udf/pandas_udf.
"""

import sys
from typing import Callable, Optional, Any

from pyspark.sql.types import DataType, StringType
from pyspark.serializers import CloudPickleSerializer


class UserDefinedFunction:
    """
    Represents a user-defined function (UDF) that can be called with columns.
    Stores the function, return type, eval type, and pickled command.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        returnType: DataType,
        evalType: int = 100,  # SQL_BATCHED_UDF
        name: Optional[str] = None,
        deterministic: bool = True,
    ):
        self.func = func
        self.returnType = returnType
        self.evalType = evalType
        self.name = name or (
            func.__name__ if hasattr(func, "__name__") else "udf"
        )
        self.deterministic = deterministic
        self.python_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

        # Cloudpickle the function (the command is the pickled (func, returnType) tuple)
        serializer = CloudPickleSerializer()
        self.command = serializer.dumps((func, returnType))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Call the UDF with Column arguments.
        Returns a Column representing the UDF call.
        """
        from pyspark._pyspark.functions import _make_udf
        from pyspark._pyspark.types import PyDataType

        # Convert returnType to PyDataType if needed
        if not isinstance(self.returnType, PyDataType):
            # Wrap the returnType in a PyDataType
            ret_type_obj = PyDataType()
            ret_type_obj._set_type(self.returnType)
        else:
            ret_type_obj = self.returnType

        return _make_udf(
            self.name,
            ret_type_obj,
            self.evalType,
            self.command,
            self.python_ver,
            *args,
            **kwargs,
        )


def udf(
    f: Optional[Callable[..., Any]] = None,
    returnType: Optional[DataType] = None,
    *,
    useArrow: Optional[bool] = None,
) -> Any:
    """
    Create a user-defined function (UDF).

    Parameters
    ----------
    f : callable, optional
        The Python function to wrap as a UDF.
    returnType : DataType, optional
        The return type of the UDF. Defaults to StringType().
    useArrow : bool, optional
        Whether to use Arrow optimization. Defaults to None (auto-detect).

    Returns
    -------
    UserDefinedFunction or callable
        If f is provided, returns a UDF. If f is None, returns a decorator.

    Examples
    --------
    >>> from pyspark.sql import functions as F
    >>> from pyspark.sql.types import IntegerType
    >>> u = F.udf(lambda x: x + 1, IntegerType())
    >>> result = spark.range(3).select(u(F.col('id')).alias('inc_id'))
    """
    if returnType is None:
        returnType = StringType()

    evalType = 100  # SQL_BATCHED_UDF (default)
    if useArrow is True:
        evalType = 101  # SQL_ARROW_BATCHED_UDF

    def _udf_decorator(func):
        return UserDefinedFunction(func, returnType, evalType)

    if f is not None:
        # Direct call: @udf(f, returnType)
        return _udf_decorator(f)
    else:
        # Decorator call: @udf(...) or @udf
        return _udf_decorator


def pandas_udf(
    f: Optional[Callable[..., Any]] = None,
    returnType: Optional[DataType] = None,
    functionType: str = "scalar",
) -> Any:
    """
    Create a pandas UDF.

    Parameters
    ----------
    f : callable, optional
        The Python function to wrap as a pandas UDF.
    returnType : DataType, optional
        The return type of the UDF. Defaults to StringType().
    functionType : str, optional
        The type of pandas UDF. Defaults to "scalar".
        Options: "scalar", "grouped_map", "grouped_agg", "cogrouped_map", ...

    Returns
    -------
    UserDefinedFunction or callable
        If f is provided, returns a UDF. If f is None, returns a decorator.

    Examples
    --------
    >>> from pyspark.sql import functions as F
    >>> from pyspark.sql.types import IntegerType
    >>> @F.pandas_udf(IntegerType())
    ... def inc_id(s):
    ...     return s + 1
    >>> result = spark.range(3).select(inc_id(F.col('id')).alias('inc_id'))
    """
    if returnType is None:
        returnType = StringType()

    # Map function type names to eval types
    eval_type_map = {
        "scalar": 200,  # SQL_SCALAR_PANDAS_UDF
        "grouped_map": 201,  # SQL_GROUPED_MAP_PANDAS_UDF
        "grouped_agg": 202,  # SQL_GROUPED_AGG_PANDAS_UDF
        "window_agg": 203,  # SQL_WINDOW_AGG_PANDAS_UDF
        "scalar_iter": 204,  # SQL_SCALAR_PANDAS_ITER_UDF
        "map_iter": 205,  # SQL_MAP_PANDAS_ITER_UDF
        "cogrouped_map": 206,  # SQL_COGROUPED_MAP_PANDAS_UDF
    }

    evalType = eval_type_map.get(functionType, 200)  # Default to scalar

    def _pandas_udf_decorator(func):
        return UserDefinedFunction(func, returnType, evalType)

    if f is not None:
        # Direct call
        return _pandas_udf_decorator(f)
    else:
        # Decorator call
        return _pandas_udf_decorator


class UDFRegistration:
    """
    Wrapper for user-defined function registration (spark.udf.register).
    """

    def __init__(self, spark_session: Any):
        self.spark_session = spark_session

    def register(
        self,
        name: str,
        f: Callable[..., Any],
        returnType: Optional[DataType] = None,
    ) -> UserDefinedFunction:
        """
        Register a Python UDF with the given name.

        Parameters
        ----------
        name : str
            Name to register the UDF with.
        f : callable
            The Python function to register.
        returnType : DataType, optional
            The return type of the UDF. Defaults to StringType().

        Returns
        -------
        UserDefinedFunction
            The registered UDF that can be called with columns.

        Examples
        --------
        >>> from pyspark.sql.types import IntegerType
        >>> spark.udf.register("inc_id", lambda x: x + 1, IntegerType())
        >>> result = spark.sql("SELECT inc_id(id) AS incremented FROM range(3)")
        """
        if returnType is None:
            returnType = StringType()

        # Create a UDF with the given name
        udf = UserDefinedFunction(f, returnType, 100, name)
        return udf
