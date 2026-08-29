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

"""
Python package for statistical functions in MLlib.
"""

# Note: MLlib is RDD/SparkContext-based. In Connect-only mode, some modules
# require pyspark.core which is not available. We guard the imports so the
# package still loads.

try:
    from pyspark.mllib.stat._statistics import Statistics, MultivariateStatisticalSummary
except (ImportError, ModuleNotFoundError):
    # _statistics requires RDD support, not available in Connect-only mode
    Statistics = None
    MultivariateStatisticalSummary = None

try:
    from pyspark.mllib.stat.distribution import MultivariateGaussian
except (ImportError, ModuleNotFoundError):
    # distribution may have RDD dependencies
    MultivariateGaussian = None

try:
    from pyspark.mllib.stat.test import ChiSqTestResult, KolmogorovSmirnovTestResult
except (ImportError, ModuleNotFoundError):
    # test module may have RDD dependencies
    ChiSqTestResult = None
    KolmogorovSmirnovTestResult = None

try:
    from pyspark.mllib.stat.KernelDensity import KernelDensity
except (ImportError, ModuleNotFoundError):
    # KernelDensity may have RDD dependencies
    KernelDensity = None

__all__ = [
    "Statistics",
    "MultivariateStatisticalSummary",
    "ChiSqTestResult",
    "KolmogorovSmirnovTestResult",
    "MultivariateGaussian",
    "KernelDensity",
]
