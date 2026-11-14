# SPDX-License-Identifier: Apache-2.0
"""Print the `params:` table of `native_pymethods!` of official pyspark.ml classes.

    python scripts/gen_native_params_table.py LinearSVC [LinearSVCModel ...]

From tests/golden/ml_api.json: each param of the class, with its getter's Python name and the Rust
names of its getter and Param attribute, in PySpark's order. The rows are indented as the tables of
the macro invocations are (8 spaces), and a row longer than 100 columns is wrapped, as rustfmt
does not format the macros.
"""

import json
import re
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden" / "ml_api.json"


def snake(name):
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()


INDENT = " " * 8
WIDTH = 100


def row(name, getter):
    """The row of the param `name`, wrapped if it is longer than WIDTH."""
    parts = [f'"{getter}"', f"get_{snake(name)}", f"param_{snake(name)}"]
    line = f'{INDENT}"{name}" => ({", ".join(parts)}),'
    if len(line) <= WIDTH:
        return line
    inner = ",\n".join(f"{INDENT}    {part}" for part in parts)
    return f'{INDENT}"{name}" => (\n{inner}\n{INDENT}),'


def main(classes):
    api = json.loads(GOLDEN.read_text())
    for cls in classes:
        entry = api[cls]
        print(f"    // {cls}")
        print("    params: [")
        for param in entry["params"]:
            name = param["name"]
            getter = "get" + name[0].upper() + name[1:]
            if getter not in entry["members"]:
                print(f"{INDENT}// {name}: the class has no {getter}")
                continue
            print(row(name, getter))
        print("    ];")


if __name__ == "__main__":
    main(sys.argv[1:])
