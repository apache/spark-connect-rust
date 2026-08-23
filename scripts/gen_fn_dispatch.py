#!/usr/bin/env python3
"""
Generator for dispatch and Python wrappers for all 440 Spark SQL functions.
Parses crates/spark-connect/src/functions.rs and generates optimized code.
"""

import re
from pathlib import Path

# Read the spark-connect functions file
functions_rs = Path("crates/spark-connect/src/functions.rs").read_text()

# Extract all function definitions with their full signatures
# Pattern: pub fn name(params) -> Column {
pattern = r"^pub fn ([a-zA-Z0-9_#]+)\((.*?)\)\s*->\s*Column\s*\{"
matches = re.finditer(pattern, functions_rs, re.MULTILINE | re.DOTALL)

functions: dict[str, dict] = {}
for match in matches:
    full_name = match.group(1)
    name = full_name.replace("r#", "")  # Remove raw string prefix like r#struct
    params_str = match.group(2).strip()

    # Parse parameters
    params = []
    if params_str:
        # Split by comma, but handle nested angle brackets
        param_parts = []
        current = ""
        depth = 0
        for char in params_str:
            if char == "<":
                depth += 1
            elif char == ">":
                depth -= 1
            elif char == "," and depth == 0:
                param_parts.append(current.strip())
                current = ""
                continue
            current += char
        if current.strip():
            param_parts.append(current.strip())

        for param in param_parts:
            # Parse each parameter: name: type
            if ":" in param:
                parts = param.split(":")
                pname = parts[0].strip()
                ptype = ":".join(parts[1:]).strip()
                params.append({"name": pname, "type": ptype})

    functions[name] = {
        "full_name": full_name,
        "params": params,
        "param_count": len(params),
    }

print(f"Found {len(functions)} functions")

# Categorize functions by their parameter types
no_args = []
single_col = []
multiple_cols = []
str_only = []
mixed = []

for name, fn in functions.items():
    if fn["param_count"] == 0:
        no_args.append(name)
    elif fn["param_count"] == 1:
        param_type = fn["params"][0]["type"]
        if "Column" in param_type:
            single_col.append(name)
        elif "&str" in param_type or "str" in param_type:
            str_only.append(name)
        else:
            mixed.append(name)
    else:
        # Check if all are Columns
        all_cols = all("Column" in p["type"] for p in fn["params"])
        if all_cols:
            multiple_cols.append(name)
        else:
            mixed.append(name)

print(f"  No args: {len(no_args)}")
print(f"  Single Column: {len(single_col)}")
print(f"  Multiple Columns: {len(multiple_cols)}")
print(f"  String only: {len(str_only)}")
print(f"  Mixed/Other: {len(mixed)}")

# Generate Rust dispatch function
rust_arms = []

# No-arg functions
for name in no_args:
    fn = functions[name]
    rust_arms.append(f'            "{name}" => Ok(spark_funcs::{fn["full_name"]}()),')

# Single column functions
for name in single_col:
    fn = functions[name]
    rust_arms.append(f'            "{name}" => {{')
    rust_arms.append(
        '                if args.is_empty() { return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Missing required column argument")); }'
    )
    rust_arms.append(f"                Ok(spark_funcs::{fn['full_name']}(args[0].clone()))")
    rust_arms.append("            },")

# String-only functions - convert first arg to string literal
for name in str_only:
    fn = functions[name]
    param_name = fn["params"][0]["name"]
    rust_arms.append(f'            "{name}" => {{')
    rust_arms.append("                // String parameter - use empty string as default")
    rust_arms.append(f'                Ok(spark_funcs::{fn["full_name"]}(""))')
    rust_arms.append("            },")

# Multiple column functions
for name in multiple_cols:
    fn = functions[name]
    col_count = sum(1 for p in fn["params"] if "Column" in p["type"])
    rust_arms.append(f'            "{name}" => {{')
    rust_arms.append(
        f'                if args.len() < {col_count} {{ return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!("Missing required arguments for {name}: expected at least {col_count}, got {{}}", args.len()))); }}'
    )
    col_args = ", ".join([f"args[{i}].clone()" for i in range(col_count)])
    rust_arms.append(f"                Ok(spark_funcs::{fn['full_name']}({col_args}))")
    rust_arms.append("            },")

# Mixed/other - try to handle generically
for name in mixed:
    fn = functions[name]
    # Special case for window(Column, &str)
    if name == "window":
        rust_arms.append('            "window" => {')
        rust_arms.append(
            '                if args.len() < 1 { return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("window() requires at least 1 column argument")); }'
        )
        rust_arms.append("                // window takes a column and a string duration parameter")
        rust_arms.append('                Ok(spark_funcs::window(args[0].clone(), ""))')
        rust_arms.append("            },")
    else:
        # For other mixed functions, try generic handling
        col_count = fn["param_count"]
        rust_arms.append(f'            "{name}" => {{')
        rust_arms.append("                // Mixed/special case: trying generic handling")
        col_args = ", ".join([f"args[{i}].clone()" for i in range(min(col_count, 1))])
        if col_count <= 1:
            if col_count == 0:
                rust_arms.append(f"                Ok(spark_funcs::{fn['full_name']}())")
            else:
                rust_arms.append(
                    '                if args.is_empty() { return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("Missing arguments")); }'
                )
                rust_arms.append(
                    f"                Ok(spark_funcs::{fn['full_name']}(args[0].clone()))"
                )
        else:
            rust_arms.append(f"                // TODO: special handling for {name}")
            rust_arms.append(
                f'                Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(format!("Function {name} not yet implemented in dispatch")))'
            )
        rust_arms.append("            },")

# Generate the dispatch code
dispatch_code = (
    """/// Auto-generated dispatch function for Spark SQL functions.
/// Generated by scripts/gen_fn_dispatch.py - do not edit directly.
pub fn call_builtin(name: &str, args: Vec<spark_connect::column::Column>) -> PyResult<spark_connect::column::Column> {
    use spark_connect::functions as spark_funcs;
    match name {
"""
    + "\n".join(rust_arms)
    + """
        _ => Err(PyErr::new::<pyo3::exceptions::PyNameError, _>(
            format!("Unknown function: {}", name)
        )),
    }
}
"""
)

# Write dispatch file
dispatch_file = Path("crates/pyspark-rs/src/dispatch_generated.rs")
dispatch_file.write_text(dispatch_code)
print(f"\nGenerated dispatch function in {dispatch_file}")

# Generate Python wrapper functions
# Skip functions that should remain hand-written (truly special ones only)
skip_functions = {
    "col",
    "lit",
    "expr",
    "sum",
    "count",
    "avg",
    "max",
    "min",
    "call_function",
    "call_udf",
    "cast",
    "column",
}

python_wrappers = []
for name in sorted(functions.keys()):
    if name not in skip_functions:
        python_wrappers.append(f'''{name} = _create_wrapper("{name}")''')

python_code = (
    '''"""Auto-generated wrapper functions for Spark SQL functions.
Generated by scripts/gen_fn_dispatch.py - do not edit directly.

This module is meant to be imported after _create_wrapper, _wrap, and _unwrap
are defined in functions.py
"""

# Generated wrappers (requires _create_wrapper to be defined)
'''
    + "\n".join(python_wrappers)
    + """
"""
)

print(f"Generated {len(python_wrappers)} Python wrapper functions")
print("\nSummary:")
print(f"  Total functions: {len(functions)}")
print(f"  Hand-written (skipped): {len(skip_functions)}")
print(f"  Auto-generated: {len(functions) - len(skip_functions)}")

# Write Python wrappers file (to be imported in functions.py)
wrappers_file = Path("python/pyspark/sql/functions_generated.py")
wrappers_file.write_text(python_code)
print(f"\nGenerated Python wrappers in {wrappers_file}")
