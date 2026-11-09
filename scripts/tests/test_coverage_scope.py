#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check that the Python coverage leaves out every file vendored verbatim from upstream.

python/.coveragerc measures the hand-written drop-in only; the files that
scripts/check_vendored_upstream.sh keeps byte-identical to the Apache Spark tag are covered
by Apache's own tests, so they must be in its omit list.
"""

import configparser
import fnmatch
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def vendored_entries():
    src = (ROOT / "scripts" / "check_vendored_upstream.sh").read_text()

    def array(name):
        body = re.search(name + r"=\((.*?)\n\)", src, re.S).group(1)
        return re.findall(r'^\s*"([^"]+)"', body, re.M)

    return array("VENDORED_DIRS"), array("VENDORED_FILES")


def omit_patterns():
    config = configparser.ConfigParser()
    config.read(ROOT / "python" / ".coveragerc")
    lines = config["run"]["omit"].splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


class CoverageScopeTests(unittest.TestCase):
    def test_every_vendored_python_file_is_omitted(self):
        dirs, files = vendored_entries()
        patterns = omit_patterns()
        base = str(ROOT / "python" / "pyspark")
        paths = [f"{base}/{d}/module.py" for d in dirs]
        paths += [f"{base}/{f}" for f in files if f.endswith(".py")]
        measured = [p for p in paths if not any(fnmatch.fnmatch(p, pat) for pat in patterns)]
        self.assertEqual(
            measured,
            [],
            "vendored upstream files measured by python/.coveragerc; add them to its omit list",
        )

    def test_the_lists_are_read(self):
        dirs, files = vendored_entries()
        self.assertIn("cloudpickle", dirs)
        self.assertIn("taskcontext.py", files)
        self.assertIn("*/cloudpickle/*", omit_patterns())


if __name__ == "__main__":
    unittest.main()
