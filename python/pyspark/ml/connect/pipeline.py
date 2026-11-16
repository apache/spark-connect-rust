"""Pipeline (pyspark.ml.connect.pipeline)."""

from pyspark._pyspark import Pipeline, PipelineModel

__all__ = ["Pipeline", "PipelineModel"]

# The public classes of the native classes (see pyspark.ml._native_classes).
from pyspark.ml._native_classes import publish as _publish  # noqa: E402

_publish(globals(), "pyspark.ml.pipeline")

# PySpark's own readers and writers of the Python format of the module, which its `__all__` does
# not name: over the param maps of the public classes, they save and load their objects too.
import pyspark.ml.pipeline as _pysparks  # noqa: E402

PipelineReader = _pysparks.PipelineReader
PipelineWriter = _pysparks.PipelineWriter
PipelineModelReader = _pysparks.PipelineModelReader
PipelineModelWriter = _pysparks.PipelineModelWriter
PipelineSharedReadWrite = _pysparks.PipelineSharedReadWrite
del _pysparks
