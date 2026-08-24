"""pyspark.sql.functions module exposing Rust-backed SQL functions."""

import os

from pyspark import _pyspark

_functions = getattr(_pyspark, 'functions', None)

if _functions is None:
    raise ImportError("_pyspark.functions not found")

# Get the native functions
_pyfunc_col = _functions.pyfunc_col
_pyfunc_lit = _functions.pyfunc_lit
_pyfunc_expr = _functions.pyfunc_expr
_pyfunc_sum = _functions.pyfunc_sum
_pyfunc_count = _functions.pyfunc_count
_pyfunc_avg = _functions.pyfunc_avg
_pyfunc_max = _functions.pyfunc_max
_pyfunc_min = _functions.pyfunc_min
_pyfunc_when = _functions.pyfunc_when
_call_function = _functions.pyfunc_call_function
# Mixed function bindings
_pyfunc_sha2 = _functions.pyfunc_sha2
_pyfunc_window = _functions.pyfunc_window
_pyfunc_from_avro = _functions.pyfunc_from_avro
_pyfunc_from_avro_with_options = _functions.pyfunc_from_avro_with_options
_pyfunc_to_avro_with_schema = _functions.pyfunc_to_avro_with_schema
_pyfunc_from_protobuf = _functions.pyfunc_from_protobuf
_pyfunc_from_protobuf_with_descriptor = _functions.pyfunc_from_protobuf_with_descriptor
_pyfunc_from_protobuf_with_descriptor_and_options = _functions.pyfunc_from_protobuf_with_descriptor_and_options
_pyfunc_from_protobuf_with_options = _functions.pyfunc_from_protobuf_with_options
_pyfunc_to_protobuf = _functions.pyfunc_to_protobuf
_pyfunc_to_protobuf_with_descriptor = _functions.pyfunc_to_protobuf_with_descriptor
_pyfunc_to_protobuf_with_descriptor_and_options = _functions.pyfunc_to_protobuf_with_descriptor_and_options
_pyfunc_to_protobuf_with_options = _functions.pyfunc_to_protobuf_with_options

# Utility functions for wrapping/unwrapping columns
def _unwrap(obj):
    """Convert a Column to its internal representation for passing to _call_function."""
    return obj


def _to_col(obj):
    """Coerce a ``ColumnOrName`` argument: a str is resolved to a column via col()."""
    if isinstance(obj, str):
        return _pyfunc_col(obj)
    return obj


def _col_wrapper(native):
    """Wrap a native single-column function so it accepts a str column name too."""
    def wrapper(col):
        return native(_to_col(col))
    return wrapper

def _wrap(obj):
    """Wrap the result from _call_function as a Column."""
    return obj

def _create_wrapper(fname):
    """Create a wrapper function that calls _call_function."""
    def wrapper(*args):
        return _wrap(_call_function(fname, *[_unwrap(a) for a in args]))
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
            args.append(k)
            args.append(v)
        # Use create_map with literal strings - we'll build the Column through _call_function
        # since create_map is auto-generated
        return _call_function("create_map", *[_pyfunc_lit(arg) for arg in args])
    # If already a Column, pass through
    return options_dict

# Import UDF functions
from pyspark.sql.udf import udf, pandas_udf

# Hand-written special functions
col = _pyfunc_col
lit = _pyfunc_lit
expr = _pyfunc_expr
sum = _col_wrapper(_pyfunc_sum)
count = _col_wrapper(_pyfunc_count)
avg = _col_wrapper(_pyfunc_avg)
max = _col_wrapper(_pyfunc_max)
min = _col_wrapper(_pyfunc_min)
when = _pyfunc_when

# Mixed/special functions - explicit wrappers for non-generic dispatch
def sha2(col, num_bits):
    """Returns the hex string result of SHA2 digest of the given data.

    Args:
        col: column to hash
        num_bits: either 256 or 512
    """
    return _pyfunc_sha2(_to_col(col), num_bits)

def window(time_column, window_duration, slide_duration=None, start_time=None):
    """Buckets rows into one or more time windows specified by the given parameters.

    Args:
        time_column: the column containing timestamps
        window_duration: a string specifying the width of the window, e.g. '10 minutes'
        slide_duration: optional, the slide interval
        start_time: optional, the start time of the first window
    """
    # Note: core only supports the 2-arg version for now
    if slide_duration is not None or start_time is not None:
        raise NotImplementedError("slide_duration and start_time not yet supported")
    return _pyfunc_window(_to_col(time_column), window_duration)

def from_avro(data, json_format_schema):
    """Deserialize Avro data into a column.

    Args:
        data: column containing binary Avro data
        json_format_schema: JSON string schema for the Avro data
    """
    return _pyfunc_from_avro(_to_col(data), json_format_schema)

def from_avro_with_options(data, json_format_schema, options=None):
    """Deserialize Avro data into a column with options.

    Args:
        data: column containing binary Avro data
        json_format_schema: JSON string schema for the Avro data
        options: dict or Column of options
    """
    if options is None:
        return _pyfunc_from_avro(_to_col(data), json_format_schema)
    options_col = _dict_to_options_column(options)
    return _pyfunc_from_avro_with_options(_to_col(data), json_format_schema, options_col)

def to_avro_with_schema(data, json_format_schema):
    """Serialize a column to Avro binary format with schema.

    Args:
        data: column to serialize
        json_format_schema: JSON string schema for the Avro data
    """
    return _pyfunc_to_avro_with_schema(_to_col(data), json_format_schema)

def from_protobuf(data, message_name, desc_file_path=None, options=None, binary_descriptor_set=None):
    """Deserialize Protobuf data into a column.

    Args:
        data: column containing binary Protobuf data
        message_name: name of the Protobuf message type
        desc_file_path: optional, path to descriptor file
        options: optional, dict or Column of options
        binary_descriptor_set: optional, binary descriptor set bytes
    """
    if binary_descriptor_set is not None and options is not None:
        options_col = _dict_to_options_column(options)
        return _pyfunc_from_protobuf_with_descriptor_and_options(
            _to_col(data), message_name, binary_descriptor_set, options_col
        )
    elif binary_descriptor_set is not None:
        return _pyfunc_from_protobuf_with_descriptor(
            _to_col(data), message_name, binary_descriptor_set
        )
    elif options is not None:
        options_col = _dict_to_options_column(options)
        return _pyfunc_from_protobuf_with_options(_to_col(data), message_name, options_col)
    else:
        return _pyfunc_from_protobuf(_to_col(data), message_name)

def to_protobuf(data, message_name, desc_file_path=None, options=None, binary_descriptor_set=None):
    """Serialize a column to Protobuf binary format.

    Args:
        data: column to serialize
        message_name: name of the Protobuf message type
        desc_file_path: optional, path to descriptor file
        options: optional, dict or Column of options
        binary_descriptor_set: optional, binary descriptor set bytes
    """
    if binary_descriptor_set is not None and options is not None:
        options_col = _dict_to_options_column(options)
        return _pyfunc_to_protobuf_with_descriptor_and_options(
            _to_col(data), message_name, binary_descriptor_set, options_col
        )
    elif binary_descriptor_set is not None:
        return _pyfunc_to_protobuf_with_descriptor(
            _to_col(data), message_name, binary_descriptor_set
        )
    elif options is not None:
        options_col = _dict_to_options_column(options)
        return _pyfunc_to_protobuf_with_options(_to_col(data), message_name, options_col)
    else:
        return _pyfunc_to_protobuf(_to_col(data), message_name)

# Now we can import generated wrappers (they use _create_wrapper defined above)
# We do this via exec to make _create_wrapper available in the generated module's namespace
import importlib.util
spec = importlib.util.spec_from_file_location("functions_generated",
                                               os.path.join(os.path.dirname(__file__), "functions_generated.py"))
_gen_module = importlib.util.module_from_spec(spec)
# Inject our helper functions into the generated module's namespace before loading
_gen_module._create_wrapper = _create_wrapper
_gen_module._wrap = _wrap
_gen_module._unwrap = _unwrap
_gen_module._call_function = _call_function
spec.loader.exec_module(_gen_module)

# Import all generated functions into this module
for name in dir(_gen_module):
    if not name.startswith('_'):
        globals()[name] = getattr(_gen_module, name)

# Build __all__ with all function names
__all__ = [
    "col",
    "lit",
    "expr",
    "sum",
    "count",
    "avg",
    "max",
    "min",
    "when",
    "udf",
    "pandas_udf",
]

# Add all other function names (dynamically)
import inspect
for name, obj in list(globals().items()):
    if (callable(obj) and
        not name.startswith('_') and
        name not in __all__ and
        name not in ['inspect', 'importlib', '_gen_module', 'spec', 'os']):
        __all__.append(name)
