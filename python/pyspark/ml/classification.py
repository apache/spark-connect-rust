# SPDX-License-Identifier: Apache-2.0
"""Standard classification APIs, with LogisticRegression and LinearSVC delegated to Rust."""


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.ml._upstream.classification import *  # noqa: F403
    from pyspark.ml._upstream.classification import (
        ClassificationModel as ClassificationModel,
    )
    from pyspark.ml._upstream.classification import (
        Classifier as Classifier,
    )
    from pyspark.ml._upstream.classification import (
        OneVsRestModelReader as OneVsRestModelReader,
    )
    from pyspark.ml._upstream.classification import (
        OneVsRestModelWriter as OneVsRestModelWriter,
    )
    from pyspark.ml._upstream.classification import (
        OneVsRestReader as OneVsRestReader,
    )
    from pyspark.ml._upstream.classification import (
        OneVsRestWriter as OneVsRestWriter,
    )
    from pyspark.ml._upstream.classification import (
        ProbabilisticClassificationModel as ProbabilisticClassificationModel,
    )
    from pyspark.ml._upstream.classification import (
        ProbabilisticClassifier as ProbabilisticClassifier,
    )


def _load_classification():
    import sys
    from importlib.machinery import SourceFileLoader
    from pathlib import Path
    from types import SimpleNamespace

    from pyspark import _pyspark
    from pyspark.ml._native_classes import publish

    # Execute the unchanged upstream definitions under their public name. Their globals,
    # annotations and Python subclass hooks keep the same module as in PySpark.
    module = sys.modules[__name__]
    source = Path(__file__).with_name("_upstream") / "classification.py"
    SourceFileLoader(__name__, str(source)).exec_module(module)
    reference = SimpleNamespace(**vars(module))
    names = (
        "LinearSVC",
        "LinearSVCModel",
        "LinearSVCSummary",
        "LinearSVCTrainingSummary",
        "LogisticRegression",
        "LogisticRegressionModel",
        "LogisticRegressionSummary",
        "LogisticRegressionTrainingSummary",
        "BinaryLogisticRegressionSummary",
        "BinaryLogisticRegressionTrainingSummary",
    )
    native = {name: getattr(_pyspark, name) for name in names}
    publish(
        native, reference, public_module=__name__,
        eager_params=(
            "LogisticRegression", "LogisticRegressionModel", "LinearSVC", "LinearSVCModel",
        ),
    )
    vars(module).update(native)


_load_classification()
del _load_classification
