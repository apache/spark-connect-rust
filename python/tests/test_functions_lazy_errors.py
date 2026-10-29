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

"""Fresh-process coverage for functions' lazy error dependency."""
import subprocess
import sys
import textwrap

import pytest


def run_fresh(source):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_functions_and_valid_lambdas_do_not_import_errors():
    run_fresh("""
        import sys
        import pyspark
        assert "pyspark.errors" not in sys.modules
        from pyspark.sql import functions as F
        assert "pyspark.errors" not in sys.modules
        assert isinstance(F.col("a") + F.lit(1), pyspark.sql.Column)
        for f in (lambda x: x + 1, lambda x, i: x + i,
                  lambda x, /: x + 1, lambda x, i=0: x + i):
            assert isinstance(F.transform("a", f), pyspark.sql.Column)
        assert isinstance(F.aggregate("a", F.lit(0), lambda a, x: a + x,
                                      lambda x: x + 1), pyspark.sql.Column)
        assert isinstance(F.transform("a", lambda x: F.transform(x, lambda y: y + 1)),
                          pyspark.sql.Column)
        assert not any(n == "pyspark.errors" or n.startswith("pyspark.errors.")
                       for n in sys.modules)
    """)


@pytest.mark.parametrize("definition,condition,parameters,message", [
    ("def invalid(): return 1", "WRONG_NUM_ARGS_FOR_HIGHER_ORDER_FUNCTION",
     {"func_name": "invalid", "num_args": "0"},
     "Function `invalid` should take between 1 and 3 arguments, but the provided function takes 0."),
    ("def invalid(a, b, c, d): return a", "WRONG_NUM_ARGS_FOR_HIGHER_ORDER_FUNCTION",
     {"func_name": "invalid", "num_args": "4"},
     "Function `invalid` should take between 1 and 3 arguments, but the provided function takes 4."),
    ("def invalid(*x): return x", "UNSUPPORTED_PARAM_TYPE_FOR_HIGHER_ORDER_FUNCTION",
     {"func_name": "invalid"},
     "Function `invalid` should use only POSITIONAL or POSITIONAL OR KEYWORD arguments."),
    ("def invalid(*, x): return x", "UNSUPPORTED_PARAM_TYPE_FOR_HIGHER_ORDER_FUNCTION",
     {"func_name": "invalid"},
     "Function `invalid` should use only POSITIONAL or POSITIONAL OR KEYWORD arguments."),
    ("def invalid(**x): return x", "UNSUPPORTED_PARAM_TYPE_FOR_HIGHER_ORDER_FUNCTION",
     {"func_name": "invalid"},
     "Function `invalid` should use only POSITIONAL or POSITIONAL OR KEYWORD arguments."),
    ("def invalid(x): return 123", "HIGHER_ORDER_FUNCTION_SHOULD_RETURN_COLUMN",
     {"func_name": "invalid", "return_type": "int"},
     "Function `invalid` should return Column, got int."),
])
def test_invalid_lambda_preserves_canonical_error(definition, condition, parameters, message):
    run_fresh(f"""
        from pyspark.sql import functions as F
        {definition}
        try:
            F.transform("a", invalid)
        except Exception as error:
            from pyspark.errors import PySparkValueError
            assert type(error) is PySparkValueError
            assert error.getCondition() == {condition!r}
            assert error.getMessageParameters() == {parameters!r}
            assert str(error) == {('[' + condition + '] ' + message)!r}
        else:
            raise AssertionError("invalid callable was accepted")
    """)
