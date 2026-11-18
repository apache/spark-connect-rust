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
"""Window / WindowSpec under the connect import path, as PySpark Connect's
``pyspark.sql.connect.window``: its ``Window`` is a class of its own, a subclass of
``pyspark.sql.window.Window``, and its ``WindowSpec`` the class of the window specifications."""
from pyspark import cloudpickle as _cloudpickle
from pyspark.sql.connect import _window_reduce
from pyspark.sql.window import Window as _ParentWindow
from pyspark.sql.window import WindowSpec

__all__ = ["Window", "WindowSpec"]


# As PySpark Connect's Window, a subclass of pyspark.sql.window.Window, of its static methods and
# attributes, which make the WindowSpecs of this module. No docstring, as PySpark's has none.
class Window(_ParentWindow):
    pass


_cloudpickle.register_pickle_by_value(_window_reduce)
