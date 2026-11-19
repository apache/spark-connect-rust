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
"""The functions of Spark Connect, as PySpark Connect's ``pyspark.sql.connect.functions.builtin``,
which the Rust core builds.

As in PySpark, ``pyspark.sql.functions`` is PySpark's package, whose functions run these when the
session is remote (``try_remote_functions`` of ``pyspark.sql.utils``). The functions that
scripts/gen_fn_dispatch.py generates in ``_generated.py`` are defined in this module, which runs
the code of that file; see its header.
"""

# sys and random as py_random: of PySpark Connect's module, whose namespace the official doctests
# of the functions take as their globals, and whose examples use them (try_add, try_multiply, ...).
import decimal
import functools
import importlib.machinery as _importlib_machinery
import inspect
import os as _os
import random as py_random
import sys
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    ValuesView,
    cast,
    overload,
)

from pyspark import _pyspark
from pyspark.errors import PySparkTypeError, PySparkValueError
from pyspark.sql import functions as pysparkfuncs
from pyspark.sql.column import Column
from pyspark.sql.connect.udtf import AnalyzeArgument, AnalyzeResult
from pyspark.sql.types import (
    ArrayType,
    DataType,
    MapType,
    StringType,
    StructType,
)

_functions = getattr(_pyspark, 'functions', None)

if _functions is None:
    raise ImportError("_pyspark.functions not found")

# Get the native functions
_pyfunc_col = _functions.pyfunc_col
_pyfunc_lit = _functions.pyfunc_lit
_pyfunc_expr = _functions.pyfunc_expr
_pyfunc_count = _functions.pyfunc_count
_pyfunc_when = _functions.pyfunc_when
_call_function = _functions.pyfunc_call_function

# Utility functions for wrapping/unwrapping columns
def _unwrap(obj):
    """Convert a Column to its internal representation for passing to _call_function."""
    return obj


def _name_to_col(obj):
    """Coerce a ``ColumnOrName`` argument of ``_dispatch``: a str is resolved to a column via
    col(), and any other value passes as it is. Not PySpark Connect's ``_to_col`` (below), which
    raises NOT_EXPECTED_TYPE for a value that is neither a Column nor a str."""
    if isinstance(obj, str):
        return _pyfunc_col(obj)
    return obj


def _wrap(obj):
    """Wrap the result from _call_function as a Column."""
    return obj

# Auto-derived from PySpark 4.3.0 signatures: for generated functions, which positional
# args are ColumnOrName (a bare str is a COLUMN NAME). Functions NOT listed here have
# ALL args as columns. Value = (column_positions, variadic_col_start_or_None).
_FN_COL_ARGS = {
    'any_value': ((0,), None),
    'approxCountDistinct': ((0,), None),
    'approx_count_distinct': ((0,), None),
    'approx_percentile': ((0,), None),
    'array_append': ((0,), None),
    'array_contains': ((0,), None),
    'array_insert': ((0, 1), None),
    'array_join': ((0,), None),
    'array_position': ((0,), None),
    'array_prepend': ((0,), None),
    'array_remove': ((0,), None),
    'array_sort': ((0,), None),
    'assert_true': ((0,), None),
    'bround': ((0,), None),
    'bucket': ((1,), None),
    'ceil': ((0,), None),
    'ceiling': ((0,), None),
    'collate': ((0,), None),
    'concat_ws': ((), 1),
    'conv': ((0,), None),
    'convert_timezone': ((2,), None),
    'count_min_sketch': ((0,), None),
    'curdate': ((), None),
    'current_catalog': ((), None),
    'current_database': ((), None),
    'current_date': ((), None),
    'current_path': ((), None),
    'current_schema': ((), None),
    'current_time': ((), None),
    'current_timestamp': ((), None),
    'current_timezone': ((), None),
    'current_user': ((), None),
    'date_format': ((0,), None),
    'date_part': ((1,), None),
    'date_trunc': ((1,), None),
    'datepart': ((1,), None),
    'decode': ((0,), None),
    'dense_rank': ((), None),
    'e': ((), None),
    'element_at': ((0,), None),
    'encode': ((0,), None),
    'extract': ((1,), None),
    'first': ((0,), None),
    'first_value': ((0,), None),
    'floor': ((0,), None),
    'format_number': ((0,), None),
    'format_string': ((), 1),
    'from_csv': ((0,), None),
    'from_json': ((0,), None),
    'from_unixtime': ((0,), None),
    'from_utc_timestamp': ((0,), None),
    'from_xml': ((0,), None),
    'get_json_object': ((0,), None),
    'histogram_numeric': ((0,), None),
    'hll_sketch_agg': ((0,), None),
    'hll_union': ((0, 1), None),
    'hll_union_agg': ((0,), None),
    'ilike': ((0, 1), None),
    'input_file_block_length': ((), None),
    'input_file_block_start': ((), None),
    'input_file_name': ((), None),
    'instr': ((0,), None),
    'kll_merge_agg_bigint': ((0,), None),
    'kll_merge_agg_double': ((0,), None),
    'kll_merge_agg_float': ((0,), None),
    'kll_sketch_agg_bigint': ((0,), None),
    'kll_sketch_agg_double': ((0,), None),
    'kll_sketch_agg_float': ((0,), None),
    'lag': ((0,), None),
    'last': ((0,), None),
    'last_value': ((0,), None),
    'lead': ((0,), None),
    'levenshtein': ((0, 1), None),
    'like': ((0, 1), None),
    'listagg': ((0,), None),
    'listagg_distinct': ((0,), None),
    'localtimestamp': ((), None),
    'locate': ((1,), None),
    'lpad': ((0,), None),
    'map_contains_key': ((0,), None),
    'max_by': ((0, 1), None),
    'min_by': ((0, 1), None),
    'mode': ((0,), None),
    'monotonically_increasing_id': ((), None),
    'months_between': ((0, 1), None),
    'next_day': ((0,), None),
    'now': ((), None),
    'nth_value': ((0,), None),
    'ntile': ((), None),
    'percent_rank': ((), None),
    'percentile': ((0,), None),
    'percentile_approx': ((0,), None),
    'pi': ((), None),
    'raise_error': ((), None),
    'rand': ((), None),
    'randn': ((), None),
    'randstr': ((), None),
    'rank': ((), None),
    'regexp_extract': ((0,), None),
    'regexp_extract_all': ((0, 1), None),
    'regexp_instr': ((0, 1), None),
    'regexp_replace': ((0,), None),
    'round': ((0,), None),
    'row_number': ((), None),
    'rpad': ((0,), None),
    'schema_of_csv': ((), None),
    'schema_of_json': ((), None),
    'schema_of_xml': ((), None),
    'session_user': ((), None),
    'session_window': ((0,), None),
    'shiftLeft': ((0,), None),
    'shiftRight': ((0,), None),
    'shiftRightUnsigned': ((0,), None),
    'shiftleft': ((0,), None),
    'shiftright': ((0,), None),
    'shiftrightunsigned': ((0,), None),
    'shuffle': ((0,), None),
    'sort_array': ((0,), None),
    'spark_partition_id': ((), None),
    'split': ((0, 2), None),
    'string_agg': ((0,), None),
    'string_agg_distinct': ((0,), None),
    'substring_index': ((0,), None),
    'theta_sketch_agg': ((0,), None),
    'theta_union': ((0, 1), None),
    'theta_union_agg': ((0,), None),
    'time_bucket': ((1,), None),
    'timestamp_add': ((1, 2), None),
    'timestamp_diff': ((1, 2), None),
    'to_avro': ((0,), None),
    'to_csv': ((0,), None),
    'to_date': ((0,), None),
    'to_json': ((0,), None),
    'to_timestamp': ((0,), None),
    'to_utc_timestamp': ((0,), None),
    'to_xml': ((0,), None),
    'translate': ((0,), None),
    'trunc': ((0,), None),
    'try_to_date': ((0,), None),
    'try_variant_array_append': ((0, 2), None),
    'try_variant_get': ((0,), None),
    'try_variant_insert': ((0, 2), None),
    'try_variant_set': ((0, 2), None),
    'tuple_intersection_agg_double': ((0,), None),
    'tuple_intersection_agg_integer': ((0,), None),
    'tuple_intersection_double': ((0, 1), None),
    'tuple_intersection_integer': ((0, 1), None),
    'tuple_intersection_theta_double': ((0, 1), None),
    'tuple_intersection_theta_integer': ((0, 1), None),
    'tuple_sketch_agg_double': ((0, 1), None),
    'tuple_sketch_agg_integer': ((0, 1), None),
    'tuple_sketch_summary_double': ((0,), None),
    'tuple_sketch_summary_integer': ((0,), None),
    'tuple_union_agg_double': ((0,), None),
    'tuple_union_agg_integer': ((0,), None),
    'tuple_union_double': ((0, 1), None),
    'tuple_union_integer': ((0, 1), None),
    'tuple_union_theta_double': ((0, 1), None),
    'tuple_union_theta_integer': ((0, 1), None),
    'uniform': ((), None),
    'unix_timestamp': ((0,), None),
    'user': ((), None),
    'uuid': ((), None),
    'variant_array_append': ((0, 2), None),
    'variant_delete': ((0,), None),
    'variant_get': ((0,), None),
    'variant_insert': ((0, 2), None),
    'variant_set': ((0, 2), None),
    'version': ((), None),
    'when': ((), None),
}

class _UnsetType:
    """Sentinel for optional generated-function args that were not supplied.

    The generated wrappers that dispatch to the Rust core (those without a PySpark Connect
    implementation to generate from) carry PySpark's real parameter names, so keyword
    calls work, but default every optional parameter to ``_UNSET`` and forward only the
    args the caller actually passed.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<unset>"


_UNSET = _UnsetType()


def _dispatch(fname, args):
    """Convert positional args per the ColumnOrName contract and dispatch to the core.

    A bare ``str`` at a column position is a COLUMN NAME (resolved via ``_name_to_col``);
    literal-position args (a format/pattern string, an int count, ...) pass through
    unchanged. Functions absent from ``_FN_COL_ARGS`` have every positional arg as a
    column. See ``_FN_COL_ARGS``.
    """
    override = _FN_COL_ARGS.get(fname)
    if override is None:
        conv = [_name_to_col(a) for a in args]
    else:
        cols, vstart = override
        conv = [
            _name_to_col(a) if (i in cols or (vstart is not None and i >= vstart)) else a
            for i, a in enumerate(args)
        ]
    return _wrap(_call_function(fname, *conv))


def _create_wrapper(fname):
    """Fallback wrapper for a generated function with no reference PySpark signature.

    Used only for functions absent from the reference (e.g. Rust-only extras); these
    keep the historical ``*args`` form. Functions present in the reference get an
    explicit signature emitted into ``_generated.py`` instead.
    """
    def wrapper(*args):
        return _dispatch(fname, list(args))
    wrapper.__doc__ = f"Auto-generated wrapper for {fname}"
    wrapper.__name__ = fname
    return wrapper

def _dict_to_options_column(options_dict):
    """Convert a Python dict[str, str] to a Column map literal.

    This is used for avro/protobuf options which are dictionaries in the Python API
    but need to be Column map literals in the Rust API. As in PySpark's _options_to_col,
    the options are read through items(), so a non-mapping such as a Column is rejected.
    """
    # Convert dict to alternating keys and values: [k1, v1, k2, v2, ...]
    args = []
    for k, v in options_dict.items():
        args.append(str(k))
        args.append(str(v))
    # Use create_map with literal strings - we'll build the Column through _call_function
    # since create_map is auto-generated
    return _call_function("create_map", *[_pyfunc_lit(arg) for arg in args])

# Import UDF / UDTF functions. pandas_udf and arrow_udf are PySpark's, as PySpark Connect's
# module imports them.
from pyspark.sql.connect.udf import udf
from pyspark.sql.connect.udtf import arrow_udtf, udtf
from pyspark.sql.pandas.functions import arrow_udf, pandas_udf


# Hand-written special functions. Thin wrappers so the public parameter names match
# the reference PySpark 4.3.0 signatures (keyword-call parity), forwarding to the native
# bindings whose Rust arg names differ.
def col(col):
    """Returns a Column based on the given column name."""
    return _pyfunc_col(col)


def lit(col):
    """Creates a Column of literal value.

    Mirrors pyspark.sql.connect.functions.lit: a list is the array of the literals of its
    elements, and a one-dimensional NumPy array that array cast to the array type of its
    dtype.
    """
    if isinstance(col, _pyspark.Column):
        return col
    if isinstance(col, list):
        if any(isinstance(c, _pyspark.Column) for c in col):
            from pyspark.errors import PySparkValueError

            raise PySparkValueError(
                errorClass="COLUMN_IN_LIST", messageParameters={"func_name": "lit"}
            )
        return array(*[lit(c) for c in col])  # noqa: F821 - from _generated

    # An ndarray exists only when NumPy is imported, so this does not import it.
    np = sys.modules.get("numpy")
    if np is not None and isinstance(col, np.ndarray) and col.ndim == 1:
        from pyspark.errors import PySparkTypeError
        from pyspark.sql.types import ArrayType, _from_numpy_type

        dt = _from_numpy_type(col.dtype)
        if dt is None:
            raise PySparkTypeError(
                errorClass="UNSUPPORTED_NUMPY_ARRAY_SCALAR",
                messageParameters={"dtype": col.dtype.name},
            )
        return array(*[lit(c) for c in col]).cast(ArrayType(dt))  # noqa: F821 - from _generated
    return _pyfunc_lit(col)


def expr(str):
    """Parses the expression string into the column that it represents."""
    return _pyfunc_expr(str)


def when(condition, value):
    """when(condition, value): a CASE WHEN of one branch.

    Mirrors pyspark.sql.connect.functions.when: the condition is a Column, and the value a
    Column or the literal of a value.
    """
    if not isinstance(condition, _pyspark.Column):
        from pyspark.errors import PySparkTypeError

        raise PySparkTypeError(
            errorClass="NOT_EXPECTED_TYPE",
            messageParameters={
                "expected_type": "Column",
                "arg_name": "condition",
                "arg_type": type(condition).__name__,
            },
        )
    from pyspark.sql.utils import enum_to_value

    value = enum_to_value(value)
    return _pyfunc_when(condition, value if isinstance(value, _pyspark.Column) else lit(value))


# The functions of _generated.py (see its header) are defined in the namespace of this module, as
# PySpark Connect's are in its module: they call the lit, column, _to_col and _invoke_function of
# this module, as they are at the call, and are of this module. The loader takes the code of the
# file of its cached bytecode, as an import does. Of what the generated code sets, this module keeps
# its own __doc__, and drops __all__, as PySpark Connect's module has none, and to_avro, which is
# pyspark.sql.avro.functions.to_avro, of the schema of its data: PySpark has no to_avro here.
_generated = _importlib_machinery.SourceFileLoader(
    "pyspark.sql.connect.functions._generated",
    _os.path.join(_os.path.dirname(__file__), "_generated.py"),
)
_doc = __doc__
exec(_generated.get_code(_generated.name), globals())
__doc__ = _doc
for _name in ("__all__", "to_avro", "_doc", "_generated"):
    globals().pop(_name)

# PySpark Connect's _to_col, of _generated.py, which raises PySparkTypeError NOT_EXPECTED_TYPE for
# an argument that is neither a Column nor a column name.
_column_or_name = _to_col  # noqa: F821 - from _generated


def count(col):
    """count(col): the number of the rows of a non-null col.

    Mirrors pyspark.sql.connect.functions.count: of a Column or a column name, else
    NOT_EXPECTED_TYPE, where a star Column counts the rows, as count(1).
    """
    return _pyfunc_count(col)


def sum_distinct(col):
    """sum_distinct(col): the sum of the distinct values of col.

    Mirrors pyspark.sql.connect.functions.sum_distinct: of a Column or a column name, else
    NOT_EXPECTED_TYPE.
    """
    return _dispatch("sum_distinct", [_column_or_name(col)])


def sumDistinct(col):
    """sumDistinct(col): deprecated since 3.4, use sum_distinct."""
    warnings.warn("Deprecated in 3.4, use sum_distinct instead.", FutureWarning)
    return sum_distinct(col)


def count_distinct(col, *cols):
    """count_distinct(col, *cols): the number of distinct combinations of the columns.

    Mirrors pyspark.sql.connect.functions.count_distinct.
    """
    return _dispatch("count_distinct", [_column_or_name(c) for c in [col, *cols]])


def countDistinct(col, *cols):
    """countDistinct(col, *cols): an alias of count_distinct."""
    return count_distinct(col, *cols)


def listagg_distinct(col, delimiter=None):
    """listagg_distinct(col, delimiter=None): the distinct values of col, concatenated.

    Mirrors pyspark.sql.connect.functions.listagg_distinct: the delimiter is sent when
    given.
    """
    args = [col] if delimiter is None else [col, lit(delimiter)]
    return _dispatch("listagg_distinct", [_column_or_name(c) for c in args])


def string_agg_distinct(col, delimiter=None):
    """string_agg_distinct(col, delimiter=None): the distinct values of col, concatenated.

    Mirrors pyspark.sql.connect.functions.string_agg_distinct: the delimiter is sent when
    given.
    """
    args = [col] if delimiter is None else [col, lit(delimiter)]
    return _dispatch("string_agg_distinct", [_column_or_name(c) for c in args])


# ---------------------------------------------------------------------------
# Higher-order functions.
#
# Mirrors pyspark.sql.connect.functions._get_lambda_parameters / _create_lambda /
# _invoke_higher_order_function exactly: a monotonic counter yields fresh variable
# names (``x_0``, ``y_1`` ...), placeholder Columns wrap UnresolvedNamedLambdaVariable,
# the user's callable is invoked to build the body, and the resulting LambdaFunction
# is passed as an argument to an UnresolvedFunction call. Defined AFTER the generated
# import so they take precedence over any generic wrapper.
# ---------------------------------------------------------------------------
import inspect as _inspect

from pyspark.errors import PySparkValueError as _PySparkValueError

_named_lambda_variable = _functions.pyfunc_named_lambda_variable
_lambda_function = _functions.pyfunc_lambda_function
_call_named_function = _functions.pyfunc_call_named_function
# Generic UnresolvedFunction invoker for ANY name (not gated by the builtin
# dispatch allowlist). Mirrors pyspark.sql.connect.functions._invoke_function.
_invoke_function = _functions.pyfunc_invoke_function

def _fresh_lambda_var_name(name):
    # The counter of UnresolvedNamedLambdaVariable, as PySpark Connect's, which a test resets.
    from pyspark.sql.connect.expressions import UnresolvedNamedLambdaVariable

    return UnresolvedNamedLambdaVariable.fresh_var_name(name)


def _get_lambda_parameters(f):
    parameters = list(_inspect.signature(f).parameters.values())
    supported_parameter_types = {
        _inspect.Parameter.POSITIONAL_OR_KEYWORD,
        _inspect.Parameter.POSITIONAL_ONLY,
    }
    if not (1 <= len(parameters) <= 3):
        raise _PySparkValueError(
            errorClass="WRONG_NUM_ARGS_FOR_HIGHER_ORDER_FUNCTION",
            messageParameters={"func_name": f.__name__, "num_args": str(len(parameters))},
        )
    if not all(p.kind in supported_parameter_types for p in parameters):
        raise _PySparkValueError(
            errorClass="UNSUPPORTED_PARAM_TYPE_FOR_HIGHER_ORDER_FUNCTION",
            messageParameters={"func_name": f.__name__},
        )
    return parameters


def _create_lambda(f):
    parameters = _get_lambda_parameters(f)
    arg_names = ["x", "y", "z"][: len(parameters)]
    var_names = [_fresh_lambda_var_name(n) for n in arg_names]
    arg_cols = [_named_lambda_variable(v) for v in var_names]
    result = f(*arg_cols)
    if not isinstance(result, _pyspark.Column):
        raise _PySparkValueError(
            errorClass="HIGHER_ORDER_FUNCTION_SHOULD_RETURN_COLUMN",
            messageParameters={"func_name": f.__name__, "return_type": type(result).__name__},
        )
    return _lambda_function(result, var_names)


def _invoke_higher_order_function(name, cols, funs):
    _cols = [_column_or_name(c) for c in cols]
    _funs = [_create_lambda(f) for f in funs]
    return _invoke_function(name, *_cols, *_funs)


def transform(col, f):
    return _invoke_higher_order_function("transform", [col], [f])


def exists(col, f):
    return _invoke_higher_order_function("exists", [col], [f])


def forall(col, f):
    return _invoke_higher_order_function("forall", [col], [f])


def filter(col, f):  # noqa: A001  (shadows builtin, as in official pyspark)
    return _invoke_higher_order_function("filter", [col], [f])


def aggregate(col, initialValue, merge, finish=None):
    if finish is not None:
        return _invoke_higher_order_function("aggregate", [col, initialValue], [merge, finish])
    return _invoke_higher_order_function("aggregate", [col, initialValue], [merge])


def reduce(col, initialValue, merge, finish=None):
    if finish is not None:
        return _invoke_higher_order_function("reduce", [col, initialValue], [merge, finish])
    return _invoke_higher_order_function("reduce", [col, initialValue], [merge])


def zip_with(left, right, f):
    return _invoke_higher_order_function("zip_with", [left, right], [f])


def transform_keys(col, f):
    return _invoke_higher_order_function("transform_keys", [col], [f])


def transform_values(col, f):
    return _invoke_higher_order_function("transform_values", [col], [f])


def map_filter(col, f):
    return _invoke_higher_order_function("map_filter", [col], [f])


def map_zip_with(col1, col2, f):
    return _invoke_higher_order_function("map_zip_with", [col1, col2], [f])


# ``random`` and ``column`` are the functions ``rand`` and ``col``, as PySpark binds them, so
# their ``__name__`` is that of the function and a pickle names it.
random = rand  # noqa: F821 - from _generated
column = col


def call_udf(udfName, *cols):
    # Mirrors _invoke_function(udfName, *cols): an UnresolvedFunction call.
    return _invoke_function(udfName, *[_column_or_name(c) for c in cols])


def call_function(funcName, *cols):
    # Mirrors ConnectColumn(CallFunction(funcName, expressions)).
    return _call_named_function(funcName, *[_column_or_name(c) for c in cols])


def broadcast(df):
    if not isinstance(df, _pyspark.DataFrame):
        raise PySparkTypeError(
            errorClass="NOT_EXPECTED_TYPE",
            messageParameters={
                "expected_type": "DataFrame",
                "arg_name": "df",
                "arg_type": type(df).__name__,
            },
        )
    return df.hint("broadcast")


# The partition transforms, deprecated here, as in PySpark: they warn and call those of
# pyspark.sql.connect.functions.partitioning, which imports this module.
def years(col):
    warnings.warn("Deprecated in 4.0.0, use partitioning.years instead.", FutureWarning)
    from pyspark.sql.connect.functions import partitioning

    return partitioning.years(col)


def months(col):
    warnings.warn("Deprecated in 4.0.0, use partitioning.months instead.", FutureWarning)
    from pyspark.sql.connect.functions import partitioning

    return partitioning.months(col)


def days(col):
    warnings.warn("Deprecated in 4.0.0, use partitioning.days instead.", FutureWarning)
    from pyspark.sql.connect.functions import partitioning

    return partitioning.days(col)


def hours(col):
    warnings.warn("Deprecated in 4.0.0, use partitioning.hours instead.", FutureWarning)
    from pyspark.sql.connect.functions import partitioning

    return partitioning.hours(col)


def bucket(numBuckets, col):
    warnings.warn("Deprecated in 4.0.0, use partitioning.bucket instead.", FutureWarning)
    from pyspark.sql.connect.functions import partitioning

    return partitioning.bucket(numBuckets, col)


# As PySpark Connect's, the functions are of this module, but pandas_udf and arrow_udf (of
# pyspark.sql.pandas.functions): udf, udtf and arrow_udtf, of pyspark.sql.connect.udf and udtf, are
# of it too, so that pickle, and cloudpickle, pickle them by reference, by this name, and the
# Python worker of the server imports PySpark's. The functions have the docstrings of PySpark's
# (``col.__doc__ = pysparkfuncs.col.__doc__``), but date_part, datepart and uuid, which PySpark
# Connect's leave without one, and its annotations, as PySpark Connect's are annotated, when they
# take the same parameters.
def _parameters(function):
    # Of no builtin function of Python, which the functions of this module shadow (bin, sum, ...).
    code = function.__code__
    varargs = (code.co_flags & _inspect.CO_VARARGS) != 0
    varkw = (code.co_flags & _inspect.CO_VARKEYWORDS) != 0
    count = code.co_argcount + code.co_kwonlyargcount + varargs + varkw
    return code.co_varnames[:count], code.co_posonlyargcount, code.co_kwonlyargcount, varargs, varkw


_NO_DOCSTRING = ("date_part", "datepart", "uuid")
for _name, _fn in list(globals().items()):
    if (
        _name.startswith("_")
        or not _inspect.isfunction(_fn)
        or _fn.__module__ not in (__name__, "pyspark.sql.connect.udf", "pyspark.sql.connect.udtf")
        or globals().get(_fn.__qualname__) is not _fn
    ):
        continue
    _fn.__module__ = __name__
    _classic = getattr(pysparkfuncs, _name, None)
    if _classic is None:
        continue
    if _name not in _NO_DOCSTRING:
        _fn.__doc__ = _classic.__doc__
    _classic = _inspect.unwrap(_classic)
    if _parameters(_classic) == _parameters(_fn):
        _fn.__annotations__ = dict(_classic.__annotations__)
del _name, _fn, _classic, _parameters
