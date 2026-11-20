# SPDX-License-Identifier: Apache-2.0
"""Capture official PySpark 4.3.0's docstrings of the modules of pyspark.sql and pyspark.errors at
run time, as digests of their exact text: of each module, of each public class and function that it
defines, and of each public member of such a class, as Python finds it on the class, of the
``__doc__`` of each, or of the ``inspect.getdoc`` of one of no docstring of its own that inherits
one (python/tests/_runtime_docstrings.py). python/tests/test_dropin_class_modules_offline.py
compares the drop-in's with them. The modules of the session and its client, whose docstrings are
not all PySpark's yet, are left out (``EXCLUDED`` of the helper).

Run with official PySpark 4.3.0 (``pyspark-client``), without a server, from the repository::

    python scripts/capture_runtime_docstrings_golden.py --out tests/golden/runtime_docstrings.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "python" / "tests"))
    import pyspark
    import pyspark.sql.connect.session  # noqa: F401  (as PySpark's modules import it)
    from _runtime_docstrings import KINDS, docstrings, module_names

    modules = {}
    for name in module_names():
        try:
            found = docstrings(name)
        except Exception as e:  # noqa: BLE001
            print(f"skip {name}: {type(e).__name__}", file=sys.stderr)
            continue
        # By the rest of the key after the module ("" for the module itself).
        modules[name] = {key.partition(":")[2]: value for key, value in sorted(found.items())}
    golden = {"pyspark": pyspark.__version__, "kinds": KINDS, "modules": modules}
    Path(args.out).write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
    count = sum(len(v) for v in modules.values())
    print(f"{count} docstrings of {len(modules)} modules of PySpark {pyspark.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
