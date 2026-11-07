"""pyspark.sql.functions module exposing Rust-backed SQL functions."""

import decimal
import functools
import importlib.util as _importlib_util
import inspect
import os as _os
import sys
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
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
from pyspark.sql.column import Column
from pyspark.sql.types import (
    ArrayType,
    ByteType,
    DataType,
    MapType,
    NumericType,
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


def _to_col(obj):
    """Coerce a ``ColumnOrName`` argument: a str is resolved to a column via col()."""
    if isinstance(obj, str):
        return _pyfunc_col(obj)
    return obj


def _wrap(obj):
    """Wrap the result from _call_function as a Column."""
    return obj

# Auto-derived from PySpark 4.2.0 signatures: for generated functions, which positional
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

    A bare ``str`` at a column position is a COLUMN NAME (resolved via ``_to_col``);
    literal-position args (a format/pattern string, an int count, ...) pass through
    unchanged. Functions absent from ``_FN_COL_ARGS`` have every positional arg as a
    column. See ``_FN_COL_ARGS``.
    """
    override = _FN_COL_ARGS.get(fname)
    if override is None:
        conv = [_to_col(a) for a in args]
    else:
        cols, vstart = override
        conv = [
            _to_col(a) if (i in cols or (vstart is not None and i >= vstart)) else a
            for i, a in enumerate(args)
        ]
    return _wrap(_call_function(fname, *conv))


def _create_wrapper(fname):
    """Fallback wrapper for a generated function with no reference PySpark signature.

    Used only for functions absent from the reference (e.g. Rust-only extras); these
    keep the historical ``*args`` form. Functions present in the reference get an
    explicit signature emitted into ``functions_generated.py`` instead.
    """
    def wrapper(*args):
        return _dispatch(fname, list(args))
    wrapper.__doc__ = f"Auto-generated wrapper for {fname}"
    wrapper.__name__ = fname
    return wrapper

def _dict_to_options_column(options_dict):
    """Convert a Python dict[str, str] to a Column map literal.

    This is used for avro/protobuf options which are dictionaries in the Python API
    but need to be Column map literals in the Rust API.
    """
    if options_dict is None:
        return None
    if isinstance(options_dict, dict):
        # Convert dict to alternating keys and values: [k1, v1, k2, v2, ...]
        args = []
        for k, v in options_dict.items():
            args.append(str(k))
            args.append(str(v))
        # Use create_map with literal strings - we'll build the Column through _call_function
        # since create_map is auto-generated
        return _call_function("create_map", *[_pyfunc_lit(arg) for arg in args])
    # If already a Column, pass through
    return options_dict

# Import UDF / UDTF functions
from pyspark.sql.udf import udf
from pyspark.sql.pandas.functions import arrow_udf, pandas_udf
from pyspark.sql.udtf import (
    udtf,
    arrow_udtf,
    OrderingColumn,
    PartitioningColumn,
    SelectedColumn,
)

# Hand-written special functions. Thin wrappers so the public parameter names match
# the reference PySpark 4.2.0 signatures (keyword-call parity), forwarding to the native
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
        return array(*[lit(c) for c in col])
    import sys

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
        return array(*[lit(c) for c in col]).cast(ArrayType(dt))
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


# Now we can import generated wrappers (they use _create_wrapper defined above)
# We do this via exec to make _create_wrapper available in the generated module's namespace
_spec = _importlib_util.spec_from_file_location(
    "functions_generated", _os.path.join(_os.path.dirname(__file__), "functions_generated.py")
)
_gen_module = _importlib_util.module_from_spec(_spec)
# Inject our helper functions into the generated module's namespace before loading. The
# functions generated from their PySpark Connect implementations call lit, column and
# _invoke_function (an UnresolvedFunction of the name and arguments), as PySpark Connect's
# do; the module defines PySpark Connect's _to_col and the other helpers itself.
_gen_module._create_wrapper = _create_wrapper
_gen_module._dispatch = _dispatch
_gen_module._UNSET = _UNSET
_gen_module._wrap = _wrap
_gen_module._unwrap = _unwrap
_gen_module._call_function = _call_function
_gen_module.lit = lit
_gen_module.column = _pyfunc_col
_gen_module._invoke_function = _functions.pyfunc_invoke_function
_spec.loader.exec_module(_gen_module)

# Import all generated functions into this module, except those that PySpark does not expose
# here: to_avro is pyspark.sql.avro.functions.to_avro, which takes the schema of its data.
_NOT_IN_PYSPARK_FUNCTIONS = ("to_avro",)
for name in _gen_module.__all__:
    if name not in _NOT_IN_PYSPARK_FUNCTIONS:
        globals()[name] = getattr(_gen_module, name)

# PySpark Connect's _to_col, which raises PySparkTypeError NOT_EXPECTED_TYPE for an
# argument that is neither a Column nor a column name.
_column_or_name = _gen_module._to_col


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
    import warnings

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

# Global monotonic counter, mirroring UnresolvedNamedLambdaVariable._nextVarNameId.
_lambda_var_name_id = [0]


def _fresh_lambda_var_name(name):
    _id = _lambda_var_name_id[0]
    _lambda_var_name_id[0] += 1
    return f"{name}_{_id}"


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
    _cols = [_to_col(c) for c in cols]
    _funs = [_create_lambda(f) for f in funs]
    return _invoke_function(name, *_cols, *_funs)


_gen_module._invoke_higher_order_function = _invoke_higher_order_function


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


# ``random`` is an alias for ``rand`` (matches pyspark.sql.functions).
random = rand


# ``column`` is an alias of ``col`` (matches pyspark.sql.functions).
def column(col):
    """Returns a Column based on the given column name (alias of ``col``)."""
    return _pyfunc_col(col)


def call_udf(udfName, *cols):
    # Mirrors _invoke_function(udfName, *cols): an UnresolvedFunction call.
    return _invoke_function(udfName, *[_column_or_name(c) for c in cols])


def call_function(funcName, *cols):
    # Mirrors ConnectColumn(CallFunction(funcName, expressions)).
    return _call_named_function(funcName, *[_column_or_name(c) for c in cols])


def broadcast(df):
    from pyspark.errors import PySparkTypeError

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


# Expose the `partitioning` submodule (bucket/days/hours/months/years) at
# ``pyspark.sql.functions.partitioning``. Since this module is a plain module (not a
# package), register a synthetic submodule holding the partition transforms, as PySpark
# Connect's, so both ``import pyspark.sql.functions.partitioning`` and
# ``F.partitioning.bucket`` work. The functions of the same names here are deprecated, as
# in PySpark: they warn and call those of the submodule.
import sys as _sys
import types as _types
import warnings as _warnings


def _partition_transforms():
    _over_columns = _gen_module._invoke_function_over_columns

    def years(col):
        return _over_columns("years", col)

    def months(col):
        return _over_columns("months", col)

    def days(col):
        return _over_columns("days", col)

    def hours(col):
        return _over_columns("hours", col)

    def bucket(numBuckets, col):
        if isinstance(numBuckets, int):
            _numBuckets = lit(numBuckets)
        elif isinstance(numBuckets, _pyspark.Column):
            _numBuckets = numBuckets
        else:
            from pyspark.errors import PySparkTypeError

            raise PySparkTypeError(
                errorClass="NOT_EXPECTED_TYPE",
                messageParameters={
                    "expected_type": "Column or int",
                    "arg_name": "numBuckets",
                    "arg_type": type(numBuckets).__name__,
                },
            )
        return _invoke_function("bucket", _numBuckets, _column_or_name(col))

    return years, months, days, hours, bucket


partitioning = _types.ModuleType("pyspark.sql.functions.partitioning")
partitioning.__doc__ = "Partition transform functions (bucket/days/hours/months/years)."
for _transform in _partition_transforms():
    _transform.__module__ = partitioning.__name__
    _transform.__qualname__ = _transform.__name__
    setattr(partitioning, _transform.__name__, _transform)
del _transform
partitioning.__all__ = ["years", "months", "days", "hours", "bucket"]


def years(col):
    _warnings.warn("Deprecated in 4.0.0, use partitioning.years instead.", FutureWarning)
    return partitioning.years(col)


def months(col):
    _warnings.warn("Deprecated in 4.0.0, use partitioning.months instead.", FutureWarning)
    return partitioning.months(col)


def days(col):
    _warnings.warn("Deprecated in 4.0.0, use partitioning.days instead.", FutureWarning)
    return partitioning.days(col)


def hours(col):
    _warnings.warn("Deprecated in 4.0.0, use partitioning.hours instead.", FutureWarning)
    return partitioning.hours(col)


def bucket(numBuckets, col):
    _warnings.warn("Deprecated in 4.0.0, use partitioning.bucket instead.", FutureWarning)
    return partitioning.bucket(numBuckets, col)


_sys.modules["pyspark.sql.functions.partitioning"] = partitioning

# Public classes/enums that pyspark.sql.functions re-exports (so `from
# pyspark.sql.functions import PandasUDFType` etc. work, matching reference pyspark).
# Guarded so a partial import never breaks the functions module.
try:
    from pyspark.sql.udf import UserDefinedFunction  # noqa: F401
except Exception:  # pragma: no cover
    pass
try:
    from pyspark.sql.udtf import (  # noqa: F401
        UserDefinedTableFunction,
        AnalyzeArgument,
        AnalyzeResult,
        SkipRestOfInputTableException,
    )
except Exception:  # pragma: no cover
    pass
from pyspark.sql.pandas.functions import PandasUDFType, ArrowUDFType  # noqa: E402,F401

# The public names of PySpark 4.3.0's pyspark.sql.functions, in its order. A star import
# brings in exactly these; the other names of this module (e.g. chr, random, uuid, which
# PySpark also has as attributes) are not exported, so they do not shadow the builtins and
# the stdlib modules of the same names.
__all__ = [
    "broadcast",
    "call_function",
    "col",
    "column",
    "lit",
    "expr",
    "coalesce",
    "ifnull",
    "nanvl",
    "nullif",
    "nullifzero",
    "nvl",
    "nvl2",
    "when",
    "zeroifnull",
    "equal_null",
    "ilike",
    "isnan",
    "isnotnull",
    "isnull",
    "like",
    "regexp",
    "regexp_like",
    "rlike",
    "asc",
    "asc_nulls_first",
    "asc_nulls_last",
    "desc",
    "desc_nulls_first",
    "desc_nulls_last",
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bin",
    "bround",
    "cbrt",
    "ceil",
    "ceiling",
    "conv",
    "cos",
    "cosh",
    "cot",
    "csc",
    "degrees",
    "e",
    "exp",
    "expm1",
    "factorial",
    "floor",
    "greatest",
    "hex",
    "hypot",
    "least",
    "ln",
    "log",
    "log10",
    "log1p",
    "log2",
    "negate",
    "negative",
    "pi",
    "pmod",
    "positive",
    "pow",
    "power",
    "radians",
    "rand",
    "randn",
    "rint",
    "round",
    "sec",
    "sign",
    "signum",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "try_add",
    "try_divide",
    "try_mod",
    "try_multiply",
    "try_subtract",
    "unhex",
    "uniform",
    "width_bucket",
    "ascii",
    "base64",
    "bit_length",
    "btrim",
    "char",
    "char_length",
    "character_length",
    "collate",
    "collation",
    "concat_ws",
    "contains",
    "decode",
    "elt",
    "encode",
    "endswith",
    "find_in_set",
    "format_number",
    "format_string",
    "from_base32",
    "initcap",
    "instr",
    "is_valid_utf8",
    "jaro_winkler_similarity",
    "lcase",
    "left",
    "length",
    "levenshtein",
    "locate",
    "lower",
    "lpad",
    "ltrim",
    "make_valid_utf8",
    "mask",
    "octet_length",
    "overlay",
    "position",
    "printf",
    "quote",
    "randstr",
    "regexp_count",
    "regexp_extract",
    "regexp_extract_all",
    "regexp_instr",
    "regexp_replace",
    "regexp_substr",
    "repeat",
    "replace",
    "right",
    "rpad",
    "rtrim",
    "sentences",
    "soundex",
    "split",
    "split_part",
    "startswith",
    "substr",
    "substring",
    "substring_index",
    "to_base32",
    "to_binary",
    "to_char",
    "to_number",
    "to_varchar",
    "translate",
    "trim",
    "try_to_binary",
    "try_to_number",
    "try_validate_utf8",
    "ucase",
    "unbase64",
    "upper",
    "validate_utf8",
    "bit_count",
    "bit_get",
    "bitwise_not",
    "getbit",
    "shiftleft",
    "shiftright",
    "shiftrightunsigned",
    "add_months",
    "convert_timezone",
    "curdate",
    "current_date",
    "current_time",
    "current_timestamp",
    "current_timezone",
    "date_add",
    "date_diff",
    "date_format",
    "date_from_unix_date",
    "date_part",
    "date_sub",
    "date_trunc",
    "dateadd",
    "datediff",
    "datepart",
    "day",
    "dayname",
    "dayofmonth",
    "dayofweek",
    "dayofyear",
    "extract",
    "from_unixtime",
    "from_utc_timestamp",
    "hour",
    "last_day",
    "localtimestamp",
    "make_date",
    "make_dt_interval",
    "make_interval",
    "make_time",
    "make_timestamp",
    "make_timestamp_ltz",
    "make_timestamp_ntz",
    "make_ym_interval",
    "minute",
    "month",
    "monthname",
    "months_between",
    "next_day",
    "now",
    "quarter",
    "second",
    "session_window",
    "timestamp_add",
    "timestamp_diff",
    "timestamp_micros",
    "timestamp_millis",
    "timestamp_nanos",
    "timestamp_seconds",
    "time_bucket",
    "time_diff",
    "time_from_micros",
    "time_from_millis",
    "time_from_seconds",
    "time_to_micros",
    "time_to_millis",
    "time_to_seconds",
    "time_trunc",
    "to_date",
    "to_time",
    "to_timestamp",
    "to_timestamp_ltz",
    "to_timestamp_ntz",
    "to_unix_timestamp",
    "to_utc_timestamp",
    "trunc",
    "try_make_interval",
    "try_make_timestamp",
    "try_make_timestamp_ltz",
    "try_make_timestamp_ntz",
    "try_to_time",
    "try_to_timestamp",
    "try_to_date",
    "unix_date",
    "unix_micros",
    "unix_millis",
    "unix_nanos",
    "unix_seconds",
    "unix_timestamp",
    "weekday",
    "weekofyear",
    "window",
    "window_time",
    "year",
    "crc32",
    "hash",
    "md5",
    "sha",
    "sha1",
    "sha2",
    "xxhash64",
    "aggregate",
    "array_sort",
    "cardinality",
    "concat",
    "element_at",
    "exists",
    "filter",
    "forall",
    "map_filter",
    "map_zip_with",
    "reduce",
    "reverse",
    "size",
    "transform",
    "transform_keys",
    "transform_values",
    "try_element_at",
    "zip_with",
    "array",
    "array_append",
    "array_compact",
    "array_contains",
    "array_distinct",
    "array_except",
    "array_insert",
    "array_intersect",
    "array_join",
    "array_max",
    "array_min",
    "array_position",
    "array_prepend",
    "array_remove",
    "array_repeat",
    "array_size",
    "array_union",
    "arrays_overlap",
    "arrays_zip",
    "flatten",
    "get",
    "sequence",
    "shuffle",
    "slice",
    "sort_array",
    "named_struct",
    "struct",
    "create_map",
    "map_concat",
    "map_contains_key",
    "map_entries",
    "map_from_arrays",
    "map_from_entries",
    "map_keys",
    "map_values",
    "str_to_map",
    "any_value",
    "approx_count_distinct",
    "approx_percentile",
    "array_agg",
    "avg",
    "bit_and",
    "bit_or",
    "bit_xor",
    "bitmap_and_agg",
    "bitmap_construct_agg",
    "bitmap_or_agg",
    "bool_and",
    "bool_or",
    "collect_list",
    "collect_set",
    "collect_union",
    "corr",
    "count",
    "count_distinct",
    "count_if",
    "count_min_sketch",
    "covar_pop",
    "covar_samp",
    "every",
    "first",
    "first_value",
    "grouping",
    "grouping_id",
    "histogram_numeric",
    "hll_sketch_agg",
    "hll_union_agg",
    "kll_sketch_agg_bigint",
    "kll_sketch_agg_double",
    "kll_sketch_agg_float",
    "kll_merge_agg_bigint",
    "kll_merge_agg_double",
    "kll_merge_agg_float",
    "kurtosis",
    "last",
    "last_value",
    "listagg",
    "listagg_distinct",
    "max",
    "max_by",
    "mean",
    "median",
    "min",
    "min_by",
    "mode",
    "percentile",
    "percentile_approx",
    "product",
    "regr_avgx",
    "regr_avgy",
    "regr_count",
    "regr_intercept",
    "regr_r2",
    "regr_slope",
    "regr_sxx",
    "regr_sxy",
    "regr_syy",
    "skewness",
    "some",
    "std",
    "stddev",
    "stddev_pop",
    "stddev_samp",
    "string_agg",
    "string_agg_distinct",
    "sum",
    "sum_distinct",
    "theta_intersection_agg",
    "theta_sketch_agg",
    "theta_union_agg",
    "tuple_intersection_agg_double",
    "tuple_intersection_agg_integer",
    "tuple_sketch_agg_double",
    "tuple_sketch_agg_integer",
    "tuple_union_agg_double",
    "tuple_union_agg_integer",
    "try_avg",
    "try_sum",
    "var_pop",
    "var_samp",
    "variance",
    "counter_diff",
    "cume_dist",
    "dense_rank",
    "lag",
    "lead",
    "nth_value",
    "ntile",
    "percent_rank",
    "rank",
    "row_number",
    "explode",
    "explode_outer",
    "inline",
    "inline_outer",
    "posexplode",
    "posexplode_outer",
    "stack",
    "years",
    "months",
    "days",
    "hours",
    "bucket",
    "from_csv",
    "schema_of_csv",
    "to_csv",
    "from_json",
    "get_json_object",
    "json_array_length",
    "json_object_keys",
    "json_tuple",
    "schema_of_json",
    "to_json",
    "is_valid_variant",
    "is_variant_null",
    "parse_json",
    "schema_of_variant",
    "schema_of_variant_agg",
    "try_variant_get",
    "variant_array_append",
    "try_variant_array_append",
    "variant_delete",
    "variant_strip_nulls",
    "variant_get",
    "variant_insert",
    "try_variant_insert",
    "variant_set",
    "try_variant_set",
    "try_parse_json",
    "to_variant_object",
    "from_xml",
    "schema_of_xml",
    "to_xml",
    "xpath",
    "xpath_boolean",
    "xpath_double",
    "xpath_float",
    "xpath_int",
    "xpath_long",
    "xpath_number",
    "xpath_short",
    "xpath_string",
    "parse_url",
    "try_parse_url",
    "url_decode",
    "url_encode",
    "try_url_decode",
    "aes_decrypt",
    "aes_encrypt",
    "assert_true",
    "bitmap_bit_position",
    "bitmap_bucket_number",
    "bitmap_count",
    "current_catalog",
    "current_database",
    "current_path",
    "current_schema",
    "current_user",
    "hmac",
    "input_file_block_length",
    "input_file_block_start",
    "input_file_name",
    "java_method",
    "monotonically_increasing_id",
    "raise_error",
    "reflect",
    "session_user",
    "spark_partition_id",
    "try_aes_decrypt",
    "try_reflect",
    "typeof",
    "user",
    "version",
    "AnalyzeArgument",
    "AnalyzeResult",
    "ArrowUDFType",
    "OrderingColumn",
    "PandasUDFType",
    "PartitioningColumn",
    "SelectedColumn",
    "SkipRestOfInputTableException",
    "UserDefinedFunction",
    "UserDefinedTableFunction",
    "arrow_udf",
    "hll_sketch_estimate",
    "hll_union",
    "kll_sketch_get_n_bigint",
    "kll_sketch_get_n_double",
    "kll_sketch_get_n_float",
    "kll_sketch_get_quantile_bigint",
    "kll_sketch_get_quantile_double",
    "kll_sketch_get_quantile_float",
    "kll_sketch_get_rank_bigint",
    "kll_sketch_get_rank_double",
    "kll_sketch_get_rank_float",
    "kll_sketch_merge_bigint",
    "kll_sketch_merge_double",
    "kll_sketch_merge_float",
    "kll_sketch_to_string_bigint",
    "kll_sketch_to_string_double",
    "kll_sketch_to_string_float",
    "theta_difference",
    "theta_intersection",
    "theta_sketch_estimate",
    "theta_union",
    "tuple_difference_double",
    "tuple_difference_integer",
    "tuple_difference_theta_double",
    "tuple_difference_theta_integer",
    "tuple_intersection_double",
    "tuple_intersection_integer",
    "tuple_intersection_theta_double",
    "tuple_intersection_theta_integer",
    "tuple_sketch_estimate_double",
    "tuple_sketch_estimate_integer",
    "tuple_sketch_summary_double",
    "tuple_sketch_summary_integer",
    "tuple_sketch_theta_double",
    "tuple_sketch_theta_integer",
    "tuple_union_double",
    "tuple_union_integer",
    "tuple_union_theta_double",
    "tuple_union_theta_integer",
    "st_asbinary",
    "st_geogfromwkb",
    "st_geomfromwkb",
    "st_setsrid",
    "st_srid",
    "vector_cosine_similarity",
    "vector_inner_product",
    "vector_l2_distance",
    "vector_norm",
    "vector_normalize",
    "vector_avg",
    "vector_sum",
    "call_udf",
    "pandas_udf",
    "udf",
    "udtf",
    "arrow_udtf",
    "unwrap_udt",
]
# Leave no loop variables behind, as PySpark has no `functions.name` or `functions.obj`.
del name

# Expose the `builtin` submodule at ``pyspark.sql.functions.builtin``. Upstream v4.2.0 splits
# functions into a package whose ``builtin`` submodule holds the actual functions and whose
# ``__init__`` re-exports them; here this module IS the implementation, so register a synthetic
# ``builtin`` submodule mirroring it (so ``import pyspark.sql.functions.builtin`` and
# ``from pyspark.sql.functions.builtin import col`` work like reference pyspark).
builtin = _types.ModuleType("pyspark.sql.functions.builtin")
builtin.__doc__ = "Built-in DataFrame functions (re-exported from pyspark.sql.functions)."
for _bname in list(__all__):
    _bobj = globals().get(_bname)
    if _bobj is not None:
        setattr(builtin, _bname, _bobj)
builtin.__all__ = list(__all__)
_sys.modules["pyspark.sql.functions.builtin"] = builtin
