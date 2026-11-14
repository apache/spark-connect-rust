# SPDX-License-Identifier: Apache-2.0
"""Print the `params:` table of `native_pymethods!` of official pyspark.ml classes.

    python scripts/gen_native_params_table.py LinearSVC [LinearSVCModel ...]

From tests/golden/ml_api.json: each param of the class, with its getter's Python name and the Rust
names of its getter and Param attribute, in PySpark's order.
"""

import json
import re
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / "tests" / "golden" / "ml_api.json"


def snake(name):
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()


def main(classes):
    api = json.loads(GOLDEN.read_text())
    for cls in classes:
        entry = api[cls]
        print(f"// {cls}")
        print("params: [")
        for param in entry["params"]:
            name = param["name"]
            getter = "get" + name[0].upper() + name[1:]
            if getter not in entry["members"]:
                print(f"    // {name}: the class has no {getter}")
                continue
            print(f'    "{name}" => ("{getter}", get_{snake(name)}, param_{snake(name)}),')
        print("];")


if __name__ == "__main__":
    main(sys.argv[1:])
