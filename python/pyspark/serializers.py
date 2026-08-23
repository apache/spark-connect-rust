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
Serializers for PySpark Spark Connect client.
Wraps cloudpickle for Python UDF serialization.
"""

import cloudpickle


class CloudPickleSerializer:
    """Wrapper around cloudpickle for serializing Python objects (especially UDFs)."""

    def dumps(self, obj):
        """Serialize an object using cloudpickle."""
        return cloudpickle.dumps(obj)

    def loads(self, data):
        """Deserialize an object using cloudpickle."""
        return cloudpickle.loads(data)
