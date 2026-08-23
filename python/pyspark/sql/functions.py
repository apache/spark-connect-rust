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
