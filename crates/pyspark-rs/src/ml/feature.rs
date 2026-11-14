// SPDX-License-Identifier: Apache-2.0
//! Native bindings for standard pyspark.ml.feature stages.
use super::*;
use pyo3::sync::PyOnceLock;
use spark_connect::ml as core;

/// The transform of a feature stage, after the check of its `params` argument (see
/// [`check_params`]).
fn transform_native<T>(
    slf: &Bound<'_, T>,
    dataset: &Bound<'_, PyAny>,
    params: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyDataFrame>
where
    T: Native + PyClass,
    T::Core: Transformer,
{
    check_params(params)?;
    let input = dataset_of(dataset, "transform")?;
    let mut transformer = snapshot(slf)?;
    let output = slf
        .py()
        .detach(|| transformer.transform(&input).map_err(Box::new))
        .to_pyerr()?;
    Ok(PyDataFrame::new(output))
}
// inspect cannot parse a textual NaN default; construct a Signature with the real float. A method's
// signature (`with_self`) starts with `self`.
fn signature(py: Python<'_>, defaults: &str, with_self: bool) -> PyResult<Py<PyAny>> {
    let inspect = py.import("inspect")?;
    let parameter = inspect.getattr("Parameter")?;
    let values = py.import("json")?.call_method1("loads", (defaults,))?;
    let parameters = PyList::empty(py);
    if with_self {
        parameters
            .append(parameter.call1(("self", parameter.getattr("POSITIONAL_OR_KEYWORD")?))?)?;
    }
    for (key, value) in values.cast::<PyDict>()?.iter() {
        let kw = PyDict::new(py);
        kw.set_item(
            "default",
            if value.is_instance_of::<PyDict>() {
                py.import("builtins")?
                    .getattr("float")?
                    .call1((value.get_item("float")?,))?
            } else {
                value
            },
        )?;
        parameters.append(parameter.call((key, parameter.getattr("KEYWORD_ONLY")?), Some(&kw))?)?;
    }
    Ok(inspect.getattr("Signature")?.call1((parameters,))?.unbind())
}

/// The `setParams` of a feature stage, a Python function that calls its native `_setParams`, so
/// that `inspect.signature` gives the keyword-only params of PySpark's `setParams` with their real
/// defaults, as the constructor's `__signature__` does: a text signature cannot hold a NaN default.
fn set_params_function(py: Python<'_>, class: &str, defaults: &str) -> PyResult<Py<PyAny>> {
    static FACTORY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let factory = FACTORY.get_or_try_init(py, || -> PyResult<Py<PyAny>> {
        let module = PyModule::from_code(
            py,
            SET_PARAMS_SOURCE,
            c"pyspark/_ml_set_params.py",
            c"pyspark._ml_set_params",
        )?;
        Ok(module.getattr("make_set_params")?.unbind())
    })?;
    Ok(factory
        .bind(py)
        .call1((class, signature(py, defaults, true)?))?
        .unbind())
}

const SET_PARAMS_SOURCE: &std::ffi::CStr = c"def make_set_params(class_name, signature):
    def setParams(self, *args, **kwargs):
        return self._setParams(*args, **kwargs)

    setParams.__qualname__ = class_name + '.setParams'
    setParams.__module__ = 'pyspark.ml.connect.feature'
    setParams.__signature__ = signature
    return setParams
";

/// A transformer of `pyspark.ml.feature` (`$wrapper`, over the core `$core`): PySpark's Params API,
/// persistence and `copy` of [`native_pymethods!`] over the params `$rows` (of
/// `scripts/gen_native_params_table.py`), its constructor and `setParams`, its setters
/// `$setters` (`(set_input_col, "setInputCol", "inputCol", Str)`), and its `transform`. Its own
/// defaults (`OwnDefaults`) are what a load keeps of the uid it was made of, of an outputCol.
macro_rules! feature_transformer {
    ($wrapper:ident, $core:ident, $name:literal, $jvm:literal, $defaults:literal,
     [$($row:tt)*],
     [$(($param:literal, $kind:ident)),* $(,)?],
     [$(($setter:ident, $pysetter:literal, $wire:literal, $setter_kind:ident)),* $(,)?]
     $(, { $($extra:tt)* })?) => {
        #[pyclass(name = $name, module = "pyspark.ml.connect.feature")]
        pub struct $wrapper {
            inner: core::$core,
            unsendable: Vec<&'static str>,
            raw: Vec<(&'static str, Py<PyAny>)>,
            own_defaults: params_api::OwnDefaults,
        }
        impl Native for $wrapper {
            type Core = core::$core;
            fn inner(&self) -> &Self::Core { &self.inner }
            fn params(&self) -> &Params { self.inner.params() }
            fn params_mut(&mut self) -> &mut Params { self.inner.params_mut() }
            fn unsendable(&self) -> &[&'static str] { &self.unsendable }
            fn unsendable_mut(&mut self) -> &mut Vec<&'static str> { &mut self.unsendable }
            fn raw_params_mut(&mut self) -> Option<&mut Vec<(&'static str, Py<PyAny>)>> { Some(&mut self.raw) }
            fn raw_params(&self) -> Option<&[(&'static str, Py<PyAny>)]> { Some(&self.raw) }
            fn own_defaults(&self) -> Option<&params_api::OwnDefaults> { Some(&self.own_defaults) }
            fn own_defaults_mut(&mut self) -> Option<&mut params_api::OwnDefaults> { Some(&mut self.own_defaults) }
            fn snapshot_core(&self, py: Python<'_>) -> PyResult<Self::Core> {
                let mut inner = self.inner.clone();
                for (name, value) in &self.raw {
                    let column = crate::functions::to_value_column(value.bind(py))?;
                    let spark_connect::expression::ExpressionNode::Literal(lit) = column.expression().node() else {
                        return Err(crate::column::raise_pyspark(py, "PySparkTypeError", "UNSUPPORTED_DATA_TYPE", &[("data_type", &value.bind(py).get_type().name()?.to_string())]));
                    };
                    *inner.params_mut() = std::mem::take(inner.params_mut()).set_param_literal(name, lit.to_literal_proto());
                }
                Ok(inner)
            }
        }
        native_pymethods! {
            $wrapper;
            jvm: $jvm, op: Transformer;
            params: [$($row)*];
            #[new]
            #[pyo3(signature = (*args, **kwargs))]
            fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
                let mut this = Self { inner: core::$core::new(), unsendable: Vec::new(), raw: Vec::new(), own_defaults: Vec::new() };
                init_params(&mut this, $name, &[], &[$(($param, Kind::$kind)),*], args, kwargs)?;
                Ok(this)
            }
            #[classattr]
            fn __signature__(py: Python<'_>) -> PyResult<Py<PyAny>> {
                signature(py, $defaults, false)
            }
            // PySpark's setParams, keyword-only, which sets the params that it is given and
            // returns the object (see set_params_function and set_params_of).
            #[classattr]
            fn setParams(py: Python<'_>) -> PyResult<Py<PyAny>> {
                set_params_function(py, $name, $defaults)
            }
            #[pyo3(name = "_setParams", signature = (*args, **kwargs))]
            fn set_params_native<'py>(
                slf: Bound<'py, Self>,
                args: &Bound<'py, PyTuple>,
                kwargs: Option<&Bound<'py, PyDict>>,
            ) -> PyResult<Bound<'py, Self>> {
                set_params_of(slf, $name, &[$(($param, Kind::$kind)),*], args, kwargs)
            }
            $(
                #[pyo3(name = $pysetter)]
                fn $setter<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> { set_param(slf, $wire, Kind::$setter_kind, value) }
            )*
            /// The transform of `dataset`, as `Transformer.transform`: of a copy of the stage of a
            /// non-empty param map `params`, which the copy applies.
            #[pyo3(signature = (dataset, params=None))]
            fn transform<'py>(
                slf: &Bound<'py, Self>,
                dataset: &Bound<'py, PyAny>,
                params: Option<&Bound<'py, PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                let map = params.filter(|p| p.cast::<PyDict>().is_ok_and(|d| !d.is_empty()));
                if let Some(map) = map {
                    let copy = params_api::copy(slf, Some(map))?;
                    return Ok(copy.as_any().call_method1("transform", (dataset,))?.unbind());
                }
                Ok(Py::new(slf.py(), transform_native(slf, dataset, params)?)?.into_any())
            }
            $($($extra)*)?
        }

        impl params_api::NativeInstance for $wrapper {
            fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
                Py::new(py, Self {
                    inner: self.inner.clone(),
                    unsendable: self.unsendable.clone(),
                    raw: self.raw.iter().map(|(name, value)| (*name, value.clone_ref(py))).collect(),
                    own_defaults: params_api::clone_own_defaults(py, &self.own_defaults),
                })
            }

            fn uid(&self) -> String {
                Transformer::operator(&self.inner).uid.clone()
            }

            fn set_uid(&mut self, uid: &str) {
                Transformer::operator_mut(&mut self.inner).uid = uid.to_string();
            }
        }
    };
}

/// An estimator of `pyspark.ml.feature` (`$wrapper`, over the core `$core`), with the same surface
/// as [`feature_transformer!`]: PySpark's Params API, persistence and `copy` of
/// [`native_pymethods!`], its constructor, `setParams`, its setters `$setters`, and `fitMultiple`.
/// Its own defaults (`OwnDefaults`) are what a load keeps of the uid it was made of.
///
/// The first arm fits the typed model `feature_api::$model`, with the defaults of the estimator
/// copied to it (`fit_native_model`). The second takes the `fit` of its own, in `{ $($fit)* }`, for
/// a class whose fit gives another type (QuantileDiscretizer's `Bucketizer`).
macro_rules! feature_estimator {
    ($wrapper:ident, $core:ident, $model:ident, $name:literal, $jvm:literal, $defaults:literal,
     [$($row:tt)*],
     [$(($param:literal, $kind:ident)),* $(,)?],
     [$(($setter:ident, $pysetter:literal, $wire:literal, $setter_kind:ident)),* $(,)?]) => {
        feature_estimator!($wrapper, $core, $name, $jvm, $defaults, [$($row)*],
            [$(($param, $kind)),*], [$(($setter, $pysetter, $wire, $setter_kind)),*], {
            #[pyo3(signature = (dataset, params=None))]
            fn fit(
                slf: &Bound<'_, Self>,
                dataset: &Bound<'_, PyAny>,
                params: Option<&Bound<'_, PyAny>>,
            ) -> PyResult<Fitted<feature_api::$model>> {
                fit_with(slf, dataset, params, || {
                    fit_native_model(slf, dataset, |py, model, df| {
                        feature_api::$model::fitted(py, model, df, object::SummaryDataset::None)
                    })
                })
            }
        });
    };
    ($wrapper:ident, $core:ident, $name:literal, $jvm:literal, $defaults:literal,
     [$($row:tt)*],
     [$(($param:literal, $kind:ident)),* $(,)?],
     [$(($setter:ident, $pysetter:literal, $wire:literal, $setter_kind:ident)),* $(,)?],
     { $($fit:tt)* }) => {
        #[pyclass(name = $name, module = "pyspark.ml.connect.feature")]
        pub struct $wrapper {
            inner: core::$core,
            unsendable: Vec<&'static str>,
            raw: Vec<(&'static str, Py<PyAny>)>,
            own_defaults: params_api::OwnDefaults,
        }
        impl Native for $wrapper {
            type Core = core::$core;
            fn inner(&self) -> &Self::Core { &self.inner }
            fn params(&self) -> &Params { self.inner.params() }
            fn params_mut(&mut self) -> &mut Params { self.inner.params_mut() }
            fn unsendable(&self) -> &[&'static str] { &self.unsendable }
            fn unsendable_mut(&mut self) -> &mut Vec<&'static str> { &mut self.unsendable }
            fn raw_params_mut(&mut self) -> Option<&mut Vec<(&'static str, Py<PyAny>)>> { Some(&mut self.raw) }
            fn raw_params(&self) -> Option<&[(&'static str, Py<PyAny>)]> { Some(&self.raw) }
            fn own_defaults(&self) -> Option<&params_api::OwnDefaults> { Some(&self.own_defaults) }
            fn own_defaults_mut(&mut self) -> Option<&mut params_api::OwnDefaults> { Some(&mut self.own_defaults) }
            fn snapshot_core(&self, py: Python<'_>) -> PyResult<Self::Core> {
                let mut inner = self.inner.clone();
                for (name, value) in &self.raw {
                    let column = crate::functions::to_value_column(value.bind(py))?;
                    let spark_connect::expression::ExpressionNode::Literal(lit) = column.expression().node() else {
                        return Err(crate::column::raise_pyspark(py, "PySparkTypeError", "UNSUPPORTED_DATA_TYPE", &[("data_type", &value.bind(py).get_type().name()?.to_string())]));
                    };
                    *inner.params_mut() = std::mem::take(inner.params_mut()).set_param_literal(name, lit.to_literal_proto());
                }
                Ok(inner)
            }
        }
        native_pymethods! {
            $wrapper;
            jvm: $jvm, op: Estimator;
            params: [$($row)*];
            #[new]
            #[pyo3(signature = (*args, **kwargs))]
            fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
                let mut this = Self { inner: core::$core::new(), unsendable: Vec::new(), raw: Vec::new(), own_defaults: Vec::new() };
                init_params(&mut this, $name, &[], &[$(($param, Kind::$kind)),*], args, kwargs)?;
                Ok(this)
            }
            #[classattr]
            fn __signature__(py: Python<'_>) -> PyResult<Py<PyAny>> {
                signature(py, $defaults, false)
            }
            // PySpark's setParams, keyword-only, which sets the params that it is given and
            // returns the object (see set_params_function and set_params_of).
            #[classattr]
            fn setParams(py: Python<'_>) -> PyResult<Py<PyAny>> {
                set_params_function(py, $name, $defaults)
            }
            #[pyo3(name = "_setParams", signature = (*args, **kwargs))]
            fn set_params_native<'py>(
                slf: Bound<'py, Self>,
                args: &Bound<'py, PyTuple>,
                kwargs: Option<&Bound<'py, PyDict>>,
            ) -> PyResult<Bound<'py, Self>> {
                set_params_of(slf, $name, &[$(($param, Kind::$kind)),*], args, kwargs)
            }
            $(
                #[pyo3(name = $pysetter)]
                fn $setter<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> { set_param(slf, $wire, Kind::$setter_kind, value) }
            )*
            $($fit)*
            /// The fits of a copy of the estimator, one per param map, as `Estimator.fitMultiple`, of
            /// PySpark's keywords.
            #[pyo3(name = "fitMultiple")]
            #[allow(non_snake_case)]
            fn fit_multiple<'py>(
                slf: &Bound<'py, Self>,
                dataset: &Bound<'py, PyAny>,
                paramMaps: &Bound<'py, PyAny>,
            ) -> PyResult<params_api::PyFitMultipleIterator> {
                params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
            }
        }

        impl params_api::NativeInstance for $wrapper {
            fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
                Py::new(py, Self {
                    inner: self.inner.clone(),
                    unsendable: self.unsendable.clone(),
                    raw: self.raw.iter().map(|(name, value)| (*name, value.clone_ref(py))).collect(),
                    own_defaults: params_api::clone_own_defaults(py, &self.own_defaults),
                })
            }

            fn uid(&self) -> String {
                Estimator::operator(&self.inner).uid.clone()
            }

            fn set_uid(&mut self, uid: &str) {
                Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
            }
        }
    };
}

feature_transformer!(PyBinarizer,
Binarizer,
"Binarizer",
"org.apache.spark.ml.feature.Binarizer",
"{\"threshold\": 0.0, \"inputCol\": null, \"outputCol\": null, \"thresholds\": null, \"inputCols\": null, \"outputCols\": null}",
[
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
    "outputCols" => ("getOutputCols", get_output_cols, param_output_cols),
    "threshold" => ("getThreshold", get_threshold, param_threshold),
    "thresholds" => ("getThresholds", get_thresholds, param_thresholds),
],
[("threshold", Float), ("inputCol", Str), ("outputCol", Str), ("thresholds", FloatList), ("inputCols", StrList), ("outputCols", StrList)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_input_cols, "setInputCols", "inputCols", StrList),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_output_cols, "setOutputCols", "outputCols", StrList),
    (set_threshold, "setThreshold", "threshold", Float),
    (set_thresholds, "setThresholds", "thresholds", FloatList),
]);

feature_estimator!(PyBucketedRandomProjectionLSH, BucketedRandomProjectionLSH, PyBucketedRandomProjectionLSHModel, "BucketedRandomProjectionLSH", "org.apache.spark.ml.feature.BucketedRandomProjectionLSH", "{\"inputCol\": null, \"outputCol\": null, \"seed\": null, \"numHashTables\": 1, \"bucketLength\": null}",
[
        "bucketLength" => ("getBucketLength", get_bucket_length, param_bucket_length),
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "numHashTables" => ("getNumHashTables", get_num_hash_tables, param_num_hash_tables),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "seed" => ("getSeed", get_seed, param_seed),
],
[("inputCol", Str), ("outputCol", Str), ("seed", Int), ("numHashTables", Int), ("bucketLength", Float)],
[
    (set_bucket_length, "setBucketLength", "bucketLength", Float),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_num_hash_tables, "setNumHashTables", "numHashTables", Int),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_seed, "setSeed", "seed", Int),
]);

feature_transformer!(PyBucketizer,
Bucketizer,
"Bucketizer",
"org.apache.spark.ml.feature.Bucketizer",
"{\"splits\": null, \"inputCol\": null, \"outputCol\": null, \"handleInvalid\": \"error\", \"splitsArray\": null, \"inputCols\": null, \"outputCols\": null}",
[
    "handleInvalid" => ("getHandleInvalid", get_handle_invalid, param_handle_invalid),
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
    "outputCols" => ("getOutputCols", get_output_cols, param_output_cols),
    "splits" => ("getSplits", get_splits, param_splits),
    "splitsArray" => ("getSplitsArray", get_splits_array, param_splits_array),
],
[("splits", FloatList), ("inputCol", Str), ("outputCol", Str), ("handleInvalid", Str), ("splitsArray", FloatListList), ("inputCols", StrList), ("outputCols", StrList)],
[
    (set_handle_invalid, "setHandleInvalid", "handleInvalid", Str),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_input_cols, "setInputCols", "inputCols", StrList),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_output_cols, "setOutputCols", "outputCols", StrList),
    (set_splits, "setSplits", "splits", FloatList),
    (set_splits_array, "setSplitsArray", "splitsArray", FloatListList),
]);

feature_estimator!(PyChiSqSelector, ChiSqSelector, PyChiSqSelectorModel, "ChiSqSelector", "org.apache.spark.ml.feature.ChiSqSelector", "{\"numTopFeatures\": 50, \"featuresCol\": \"features\", \"outputCol\": null, \"labelCol\": \"label\", \"selectorType\": \"numTopFeatures\", \"percentile\": 0.1, \"fpr\": 0.05, \"fdr\": 0.05, \"fwe\": 0.05}",
[
        "fdr" => ("getFdr", get_fdr, param_fdr),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "fpr" => ("getFpr", get_fpr, param_fpr),
        "fwe" => ("getFwe", get_fwe, param_fwe),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "numTopFeatures" => ("getNumTopFeatures", get_num_top_features, param_num_top_features),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "percentile" => ("getPercentile", get_percentile, param_percentile),
        "selectorType" => ("getSelectorType", get_selector_type, param_selector_type),
],
[("numTopFeatures", Int), ("featuresCol", Str), ("outputCol", Str), ("labelCol", Str), ("selectorType", Str), ("percentile", Float), ("fpr", Float), ("fdr", Float), ("fwe", Float)],
[
    (set_fdr, "setFdr", "fdr", Float),
    (set_features_col, "setFeaturesCol", "featuresCol", Str),
    (set_fpr, "setFpr", "fpr", Float),
    (set_fwe, "setFwe", "fwe", Float),
    (set_label_col, "setLabelCol", "labelCol", Str),
    (set_num_top_features, "setNumTopFeatures", "numTopFeatures", Int),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_percentile, "setPercentile", "percentile", Float),
    (set_selector_type, "setSelectorType", "selectorType", Str),
]);

feature_estimator!(PyCountVectorizer, CountVectorizer, PyCountVectorizerModel, "CountVectorizer", "org.apache.spark.ml.feature.CountVectorizer", "{\"minTF\": 1.0, \"minDF\": 1.0, \"maxDF\": 9223372036854775807, \"vocabSize\": 262144, \"binary\": false, \"inputCol\": null, \"outputCol\": null}",
[
        "binary" => ("getBinary", get_binary, param_binary),
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "maxDF" => ("getMaxDF", get_max_df, param_max_df),
        "minDF" => ("getMinDF", get_min_df, param_min_df),
        "minTF" => ("getMinTF", get_min_tf, param_min_tf),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "vocabSize" => ("getVocabSize", get_vocab_size, param_vocab_size),
],
[("minTF", Float), ("minDF", Float), ("maxDF", Float), ("vocabSize", Int), ("binary", Bool), ("inputCol", Str), ("outputCol", Str)],
[
    (set_binary, "setBinary", "binary", Bool),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_max_df, "setMaxDF", "maxDF", Float),
    (set_min_df, "setMinDF", "minDF", Float),
    (set_min_tf, "setMinTF", "minTF", Float),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_vocab_size, "setVocabSize", "vocabSize", Int),
]);

feature_transformer!(PyDCT,
DCT,
"DCT",
"org.apache.spark.ml.feature.DCT",
"{\"inverse\": false, \"inputCol\": null, \"outputCol\": null}",
[
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "inverse" => ("getInverse", get_inverse, param_inverse),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[("inverse", Bool), ("inputCol", Str), ("outputCol", Str)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_inverse, "setInverse", "inverse", Bool),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_transformer!(PyElementwiseProduct,
ElementwiseProduct,
"ElementwiseProduct",
"org.apache.spark.ml.feature.ElementwiseProduct",
"{\"scalingVec\": null, \"inputCol\": null, \"outputCol\": null}",
[
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
    "scalingVec" => ("getScalingVec", get_scaling_vec, param_scaling_vec),
],
[
    ("scalingVec", Vector),
    ("inputCol", Str),
    ("outputCol", Str)
],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_scaling_vec, "setScalingVec", "scalingVec", Vector),
]);

feature_transformer!(PyFeatureHasher,
FeatureHasher,
"FeatureHasher",
"org.apache.spark.ml.feature.FeatureHasher",
"{\"numFeatures\": 262144, \"inputCols\": null, \"outputCol\": null, \"categoricalCols\": null}",
[
    "categoricalCols" => ("getCategoricalCols", get_categorical_cols, param_categorical_cols),
    "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
    "numFeatures" => ("getNumFeatures", get_num_features, param_num_features),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[("numFeatures", Int), ("inputCols", StrList), ("outputCol", Str), ("categoricalCols", StrList)],
[
    (set_categorical_cols, "setCategoricalCols", "categoricalCols", StrList),
    (set_input_cols, "setInputCols", "inputCols", StrList),
    (set_num_features, "setNumFeatures", "numFeatures", Int),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_transformer!(PyHashingTF,
HashingTF,
"HashingTF",
"org.apache.spark.ml.feature.HashingTF",
"{\"numFeatures\": 262144, \"binary\": false, \"inputCol\": null, \"outputCol\": null}",
[
    "binary" => ("getBinary", get_binary, param_binary),
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "numFeatures" => ("getNumFeatures", get_num_features, param_num_features),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[
    ("numFeatures", Int),
    ("binary", Bool),
    ("inputCol", Str),
    ("outputCol", Str)
],
[
    (set_binary, "setBinary", "binary", Bool),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_num_features, "setNumFeatures", "numFeatures", Int),
    (set_output_col, "setOutputCol", "outputCol", Str),
], {
    /// PySpark's `HashingTF.indexOf`, which begins with `JavaWrapper._transfer_params_to_java`,
    /// whose import of `pyspark.core.context` fails over Spark Connect, as the official one's
    /// does (there is no JVM to index the term with).
    #[pyo3(name = "indexOf", signature = (term), text_signature = "($self, term)")]
    #[allow(non_snake_case, unused_variables)]
    fn index_of(slf: &Bound<'_, Self>, term: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.py().import("pyspark.core.context")?;
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "HashingTF.indexOf needs the JVM of PySpark's classic API",
        ))
    }
});

feature_estimator!(PyIDF, IDF, PyIDFModel, "IDF", "org.apache.spark.ml.feature.IDF", "{\"minDocFreq\": 0, \"inputCol\": null, \"outputCol\": null}",
[
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "minDocFreq" => ("getMinDocFreq", get_min_doc_freq, param_min_doc_freq),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[("minDocFreq", Int), ("inputCol", Str), ("outputCol", Str)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_min_doc_freq, "setMinDocFreq", "minDocFreq", Int),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_estimator!(PyImputer, Imputer, PyImputerModel, "Imputer", "org.apache.spark.ml.feature.Imputer", "{\"strategy\": \"mean\", \"missingValue\": {\"float\": \"nan\"}, \"inputCols\": null, \"outputCols\": null, \"inputCol\": null, \"outputCol\": null, \"relativeError\": 0.001}",
[
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
        "missingValue" => ("getMissingValue", get_missing_value, param_missing_value),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "outputCols" => ("getOutputCols", get_output_cols, param_output_cols),
        "relativeError" => ("getRelativeError", get_relative_error, param_relative_error),
        "strategy" => ("getStrategy", get_strategy, param_strategy),
],
[("strategy", Str), ("missingValue", Float), ("inputCols", StrList), ("outputCols", StrList), ("inputCol", Str), ("outputCol", Str), ("relativeError", Float)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_input_cols, "setInputCols", "inputCols", StrList),
    (set_missing_value, "setMissingValue", "missingValue", Float),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_output_cols, "setOutputCols", "outputCols", StrList),
    (set_relative_error, "setRelativeError", "relativeError", Float),
    (set_strategy, "setStrategy", "strategy", Str),
]);

feature_transformer!(PyIndexToString,
IndexToString,
"IndexToString",
"org.apache.spark.ml.feature.IndexToString",
"{\"inputCol\": null, \"outputCol\": null, \"labels\": null}",
[
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "labels" => ("getLabels", get_labels, param_labels),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[("inputCol", Str), ("outputCol", Str), ("labels", StrList)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_labels, "setLabels", "labels", StrList),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_transformer!(PyInteraction,
Interaction,
"Interaction",
"org.apache.spark.ml.feature.Interaction",
"{\"inputCols\": null, \"outputCol\": null}",
[
    "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[("inputCols", StrList), ("outputCol", Str)],
[
    (set_input_cols, "setInputCols", "inputCols", StrList),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_estimator!(PyMaxAbsScaler, MaxAbsScaler, PyMaxAbsScalerModel, "MaxAbsScaler", "org.apache.spark.ml.feature.MaxAbsScaler", "{\"inputCol\": null, \"outputCol\": null}",
[
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[("inputCol", Str), ("outputCol", Str)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_estimator!(PyMinHashLSH, MinHashLSH, PyMinHashLSHModel, "MinHashLSH", "org.apache.spark.ml.feature.MinHashLSH", "{\"inputCol\": null, \"outputCol\": null, \"seed\": null, \"numHashTables\": 1}",
[
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "numHashTables" => ("getNumHashTables", get_num_hash_tables, param_num_hash_tables),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "seed" => ("getSeed", get_seed, param_seed),
],
[("inputCol", Str), ("outputCol", Str), ("seed", Int), ("numHashTables", Int)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_num_hash_tables, "setNumHashTables", "numHashTables", Int),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_seed, "setSeed", "seed", Int),
]);

feature_estimator!(PyMinMaxScaler, MinMaxScaler, PyMinMaxScalerModel, "MinMaxScaler", "org.apache.spark.ml.feature.MinMaxScaler", "{\"min\": 0.0, \"max\": 1.0, \"inputCol\": null, \"outputCol\": null}",
[
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "max" => ("getMax", get_max, param_max),
        "min" => ("getMin", get_min, param_min),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[("min", Float), ("max", Float), ("inputCol", Str), ("outputCol", Str)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_max, "setMax", "max", Float),
    (set_min, "setMin", "min", Float),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_transformer!(PyNGram,
NGram,
"NGram",
"org.apache.spark.ml.feature.NGram",
"{\"n\": 2, \"inputCol\": null, \"outputCol\": null}",
[
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "n" => ("getN", get_n, param_n),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[("n", Int), ("inputCol", Str), ("outputCol", Str)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_n, "setN", "n", Int),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_transformer!(PyNormalizer,
Normalizer,
"Normalizer",
"org.apache.spark.ml.feature.Normalizer",
"{\"p\": 2.0, \"inputCol\": null, \"outputCol\": null}",
[
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
    "p" => ("getP", get_p, param_p),
],
[("p", Float), ("inputCol", Str), ("outputCol", Str)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_p, "setP", "p", Float),
]);

feature_estimator!(PyOneHotEncoder, OneHotEncoder, PyOneHotEncoderModel, "OneHotEncoder", "org.apache.spark.ml.feature.OneHotEncoder", "{\"inputCols\": null, \"outputCols\": null, \"handleInvalid\": \"error\", \"dropLast\": true, \"inputCol\": null, \"outputCol\": null}",
[
        "dropLast" => ("getDropLast", get_drop_last, param_drop_last),
        "handleInvalid" => ("getHandleInvalid", get_handle_invalid, param_handle_invalid),
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "outputCols" => ("getOutputCols", get_output_cols, param_output_cols),
],
[("inputCols", StrList), ("outputCols", StrList), ("handleInvalid", Str), ("dropLast", Bool), ("inputCol", Str), ("outputCol", Str)],
[
    (set_drop_last, "setDropLast", "dropLast", Bool),
    (set_handle_invalid, "setHandleInvalid", "handleInvalid", Str),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_input_cols, "setInputCols", "inputCols", StrList),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_output_cols, "setOutputCols", "outputCols", StrList),
]);

feature_estimator!(PyPCA, PCA, PyPCAModel, "PCA", "org.apache.spark.ml.feature.PCA", "{\"k\": null, \"inputCol\": null, \"outputCol\": null}",
[
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "k" => ("getK", get_k, param_k),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[("k", Int), ("inputCol", Str), ("outputCol", Str)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_k, "setK", "k", Int),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_transformer!(PyPolynomialExpansion,
PolynomialExpansion,
"PolynomialExpansion",
"org.apache.spark.ml.feature.PolynomialExpansion",
"{\"degree\": 2, \"inputCol\": null, \"outputCol\": null}",
[
    "degree" => ("getDegree", get_degree, param_degree),
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[("degree", Int), ("inputCol", Str), ("outputCol", Str)],
[
    (set_degree, "setDegree", "degree", Int),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_estimator!(PyQuantileDiscretizer, QuantileDiscretizer, "QuantileDiscretizer", "org.apache.spark.ml.feature.QuantileDiscretizer", "{\"numBuckets\": 2, \"inputCol\": null, \"outputCol\": null, \"relativeError\": 0.001, \"handleInvalid\": \"error\", \"numBucketsArray\": null, \"inputCols\": null, \"outputCols\": null}",
[
        "handleInvalid" => ("getHandleInvalid", get_handle_invalid, param_handle_invalid),
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
        "numBuckets" => ("getNumBuckets", get_num_buckets, param_num_buckets),
        "numBucketsArray" => ("getNumBucketsArray", get_num_buckets_array, param_num_buckets_array),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "outputCols" => ("getOutputCols", get_output_cols, param_output_cols),
        "relativeError" => ("getRelativeError", get_relative_error, param_relative_error),
],
[("numBuckets", Int), ("inputCol", Str), ("outputCol", Str), ("relativeError", Float), ("handleInvalid", Str), ("numBucketsArray", IntList), ("inputCols", StrList), ("outputCols", StrList)],
[
    (set_handle_invalid, "setHandleInvalid", "handleInvalid", Str),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_input_cols, "setInputCols", "inputCols", StrList),
    (set_num_buckets, "setNumBuckets", "numBuckets", Int),
    (set_num_buckets_array, "setNumBucketsArray", "numBucketsArray", IntList),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_output_cols, "setOutputCols", "outputCols", StrList),
    (set_relative_error, "setRelativeError", "relativeError", Float),
],
{
    /// Fit the discretizer, as PySpark's `QuantileDiscretizer.fit`: of its `_create_model`, a
    /// `Bucketizer` (of a new uid) of the splits, `inputCol(s)`, `outputCol(s)` and `handleInvalid`
    /// of the discretizer (see `QuantileDiscretizer::fit_bucketizer`).
    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<PyBucketizer>> {
        fit_with(slf, dataset, params, || {
            let py = slf.py();
            let df = dataset_of(dataset, "fit")?;
            let mut estimator = snapshot(slf)?;
            let session = df.spark_session();
            let bucketizer = interruptible(py, &session, || {
                estimator.fit_bucketizer(&df).map_err(Box::new)
            })?;
            Ok(PyBucketizer {
                inner: bucketizer,
                unsendable: Vec::new(),
                raw: Vec::new(),
                own_defaults: Vec::new(),
            })
        })
    }
});

feature_estimator!(PyRobustScaler, RobustScaler, PyRobustScalerModel, "RobustScaler", "org.apache.spark.ml.feature.RobustScaler", "{\"lower\": 0.25, \"upper\": 0.75, \"withCentering\": false, \"withScaling\": true, \"inputCol\": null, \"outputCol\": null, \"relativeError\": 0.001}",
[
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "lower" => ("getLower", get_lower, param_lower),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "relativeError" => ("getRelativeError", get_relative_error, param_relative_error),
        "upper" => ("getUpper", get_upper, param_upper),
        "withCentering" => ("getWithCentering", get_with_centering, param_with_centering),
        "withScaling" => ("getWithScaling", get_with_scaling, param_with_scaling),
],
[("lower", Float), ("upper", Float), ("withCentering", Bool), ("withScaling", Bool), ("inputCol", Str), ("outputCol", Str), ("relativeError", Float)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_lower, "setLower", "lower", Float),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_relative_error, "setRelativeError", "relativeError", Float),
    (set_upper, "setUpper", "upper", Float),
    (set_with_centering, "setWithCentering", "withCentering", Bool),
    (set_with_scaling, "setWithScaling", "withScaling", Bool),
]);

feature_transformer!(PyRegexTokenizer,
RegexTokenizer,
"RegexTokenizer",
"org.apache.spark.ml.feature.RegexTokenizer",
"{\"minTokenLength\": 1, \"gaps\": true, \"pattern\": \"\\\\s+\", \"inputCol\": null, \"outputCol\": null, \"toLowercase\": true}",
[
    "gaps" => ("getGaps", get_gaps, param_gaps),
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "minTokenLength" => ("getMinTokenLength", get_min_token_length, param_min_token_length),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
    "pattern" => ("getPattern", get_pattern, param_pattern),
    "toLowercase" => ("getToLowercase", get_to_lowercase, param_to_lowercase),
],
[("minTokenLength", Int), ("gaps", Identity), ("pattern", Str), ("inputCol", Str), ("outputCol", Str), ("toLowercase", Bool)],
[
    (set_gaps, "setGaps", "gaps", Identity),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_min_token_length, "setMinTokenLength", "minTokenLength", Int),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_pattern, "setPattern", "pattern", Str),
    (set_to_lowercase, "setToLowercase", "toLowercase", Bool),
]);

feature_estimator!(PyRFormula, RFormula, PyRFormulaModel, "RFormula", "org.apache.spark.ml.feature.RFormula", "{\"formula\": null, \"featuresCol\": \"features\", \"labelCol\": \"label\", \"forceIndexLabel\": false, \"stringIndexerOrderType\": \"frequencyDesc\", \"handleInvalid\": \"error\"}",
[
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "forceIndexLabel" => ("getForceIndexLabel", get_force_index_label, param_force_index_label),
        "formula" => ("getFormula", get_formula, param_formula),
        "handleInvalid" => ("getHandleInvalid", get_handle_invalid, param_handle_invalid),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "stringIndexerOrderType" => (
            "getStringIndexerOrderType",
            get_string_indexer_order_type,
            param_string_indexer_order_type
        ),
],
[("formula", Str), ("featuresCol", Str), ("labelCol", Str), ("forceIndexLabel", Bool), ("stringIndexerOrderType", Str), ("handleInvalid", Str)],
[
    (set_features_col, "setFeaturesCol", "featuresCol", Str),
    (set_force_index_label, "setForceIndexLabel", "forceIndexLabel", Bool),
    (set_formula, "setFormula", "formula", Str),
    (set_handle_invalid, "setHandleInvalid", "handleInvalid", Str),
    (set_label_col, "setLabelCol", "labelCol", Str),
    (set_string_indexer_order_type, "setStringIndexerOrderType", "stringIndexerOrderType", Str),
]);

feature_transformer!(PySQLTransformer,
    SQLTransformer,
    "SQLTransformer",
    "org.apache.spark.ml.feature.SQLTransformer",
    "{\"statement\": null}",
    [
        "statement" => ("getStatement", get_statement, param_statement),
    ],
    [("statement", Str)],
    [(set_statement, "setStatement", "statement", Str),]);

feature_estimator!(PyStandardScaler, StandardScaler, PyStandardScalerModel, "StandardScaler", "org.apache.spark.ml.feature.StandardScaler", "{\"withMean\": false, \"withStd\": true, \"inputCol\": null, \"outputCol\": null}",
[
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "withMean" => ("getWithMean", get_with_mean, param_with_mean),
        "withStd" => ("getWithStd", get_with_std, param_with_std),
],
[("withMean", Bool), ("withStd", Bool), ("inputCol", Str), ("outputCol", Str)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_with_mean, "setWithMean", "withMean", Bool),
    (set_with_std, "setWithStd", "withStd", Bool),
]);

feature_transformer!(PyStopWordsRemover,
StopWordsRemover,
"StopWordsRemover",
"org.apache.spark.ml.feature.StopWordsRemover",
"{\"inputCol\": null, \"outputCol\": null, \"stopWords\": null, \"caseSensitive\": false, \"locale\": null, \"inputCols\": null, \"outputCols\": null}",
[
    "caseSensitive" => ("getCaseSensitive", get_case_sensitive, param_case_sensitive),
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
    "locale" => ("getLocale", get_locale, param_locale),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
    "outputCols" => ("getOutputCols", get_output_cols, param_output_cols),
    "stopWords" => ("getStopWords", get_stop_words, param_stop_words),
],
[("inputCol", Str), ("outputCol", Str), ("stopWords", StrList), ("caseSensitive", Bool), ("locale", Str), ("inputCols", StrList), ("outputCols", StrList)],
[
    (set_case_sensitive, "setCaseSensitive", "caseSensitive", Bool),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_input_cols, "setInputCols", "inputCols", StrList),
    (set_locale, "setLocale", "locale", Str),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_output_cols, "setOutputCols", "outputCols", StrList),
    (set_stop_words, "setStopWords", "stopWords", StrList),
],
{
    /// PySpark's `StopWordsRemover.loadDefaultStopWords`: the server's default stop words of
    /// `language`, a call of Spark Connect's ML helper on the active session (as
    /// `invoke_helper_attr`), which raises PySpark's bare AssertionError without one. The
    /// language is sent as the literal of its value, which the server checks, as PySpark's
    /// `serialize` sends an argument.
    #[staticmethod]
    #[pyo3(name = "loadDefaultStopWords")]
    fn load_default_stop_words(
        py: Python<'_>,
        language: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<String>> {
        let session = crate::session::active_session(py)
            .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err(()))?;
        let session = session.borrow(py).session.clone();
        let column = crate::functions::to_value_column(language)?;
        let spark_connect::expression::ExpressionNode::Literal(literal) =
            column.expression().node()
        else {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkTypeError",
                "UNSUPPORTED_DATA_TYPE",
                &[("data_type", &language.get_type().name()?.to_string())],
            ));
        };
        let literal = literal.to_literal_proto();
        interruptible(py, &session, || {
            core::ml_helper_attribute(
                &session,
                "stopWordsRemoverLoadDefaultStopWords",
                vec![literal],
            )
            .and_then(core::ml_string_array)
        })
    }
});

feature_estimator!(PyStringIndexer, StringIndexer, PyStringIndexerModel, "StringIndexer", "org.apache.spark.ml.feature.StringIndexer", "{\"inputCol\": null, \"outputCol\": null, \"inputCols\": null, \"outputCols\": null, \"handleInvalid\": \"error\", \"stringOrderType\": \"frequencyDesc\"}",
[
        "handleInvalid" => ("getHandleInvalid", get_handle_invalid, param_handle_invalid),
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "outputCols" => ("getOutputCols", get_output_cols, param_output_cols),
        "stringOrderType" => ("getStringOrderType", get_string_order_type, param_string_order_type),
],
[("inputCol", Str), ("outputCol", Str), ("inputCols", StrList), ("outputCols", StrList), ("handleInvalid", Str), ("stringOrderType", Str)],
[
    (set_handle_invalid, "setHandleInvalid", "handleInvalid", Str),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_input_cols, "setInputCols", "inputCols", StrList),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_output_cols, "setOutputCols", "outputCols", StrList),
    (set_string_order_type, "setStringOrderType", "stringOrderType", Str),
]);

feature_estimator!(PyTargetEncoder, TargetEncoder, PyTargetEncoderModel, "TargetEncoder", "org.apache.spark.ml.feature.TargetEncoder", "{\"inputCols\": null, \"outputCols\": null, \"labelCol\": \"label\", \"handleInvalid\": \"error\", \"targetType\": \"binary\", \"smoothing\": 0.0, \"inputCol\": null, \"outputCol\": null}",
[
        "handleInvalid" => ("getHandleInvalid", get_handle_invalid, param_handle_invalid),
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "outputCols" => ("getOutputCols", get_output_cols, param_output_cols),
        "smoothing" => ("getSmoothing", get_smoothing, param_smoothing),
        "targetType" => ("getTargetType", get_target_type, param_target_type),
],
[("inputCols", StrList), ("outputCols", StrList), ("labelCol", Str), ("handleInvalid", Str), ("targetType", Str), ("smoothing", Float), ("inputCol", Str), ("outputCol", Str)],
[
    (set_handle_invalid, "setHandleInvalid", "handleInvalid", Str),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_input_cols, "setInputCols", "inputCols", StrList),
    (set_label_col, "setLabelCol", "labelCol", Str),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_output_cols, "setOutputCols", "outputCols", StrList),
    (set_smoothing, "setSmoothing", "smoothing", Float),
    (set_target_type, "setTargetType", "targetType", Str),
]);

feature_transformer!(PyTokenizer,
Tokenizer,
"Tokenizer",
"org.apache.spark.ml.feature.Tokenizer",
"{\"inputCol\": null, \"outputCol\": null}",
[
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[("inputCol", Str), ("outputCol", Str)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_estimator!(PyUnivariateFeatureSelector, UnivariateFeatureSelector, PyUnivariateFeatureSelectorModel, "UnivariateFeatureSelector", "org.apache.spark.ml.feature.UnivariateFeatureSelector", "{\"featuresCol\": \"features\", \"outputCol\": null, \"labelCol\": \"label\", \"selectionMode\": \"numTopFeatures\"}",
[
        "featureType" => ("getFeatureType", get_feature_type, param_feature_type),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "labelType" => ("getLabelType", get_label_type, param_label_type),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "selectionMode" => ("getSelectionMode", get_selection_mode, param_selection_mode),
        "selectionThreshold" => (
            "getSelectionThreshold",
            get_selection_threshold,
            param_selection_threshold
        ),
],
[("featuresCol", Str), ("outputCol", Str), ("labelCol", Str), ("selectionMode", Str)],
[
    (set_feature_type, "setFeatureType", "featureType", Str),
    (set_features_col, "setFeaturesCol", "featuresCol", Str),
    (set_label_col, "setLabelCol", "labelCol", Str),
    (set_label_type, "setLabelType", "labelType", Str),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_selection_mode, "setSelectionMode", "selectionMode", Str),
    (set_selection_threshold, "setSelectionThreshold", "selectionThreshold", Float),
]);

feature_estimator!(PyVarianceThresholdSelector, VarianceThresholdSelector, PyVarianceThresholdSelectorModel, "VarianceThresholdSelector", "org.apache.spark.ml.feature.VarianceThresholdSelector", "{\"featuresCol\": \"features\", \"outputCol\": null, \"varianceThreshold\": 0.0}",
[
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "varianceThreshold" => (
            "getVarianceThreshold",
            get_variance_threshold,
            param_variance_threshold
        ),
],
[("featuresCol", Str), ("outputCol", Str), ("varianceThreshold", Float)],
[
    (set_features_col, "setFeaturesCol", "featuresCol", Str),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_variance_threshold, "setVarianceThreshold", "varianceThreshold", Float),
]);

feature_transformer!(PyVectorAssembler,
VectorAssembler,
"VectorAssembler",
"org.apache.spark.ml.feature.VectorAssembler",
"{\"inputCols\": null, \"outputCol\": null, \"handleInvalid\": \"error\"}",
[
    "handleInvalid" => ("getHandleInvalid", get_handle_invalid, param_handle_invalid),
    "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[
    ("inputCols", StrList),
    ("outputCol", Str),
    ("handleInvalid", Str)
],
[
    (set_handle_invalid, "setHandleInvalid", "handleInvalid", Str),
    (set_input_cols, "setInputCols", "inputCols", StrList),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_estimator!(PyVectorIndexer, VectorIndexer, PyVectorIndexerModel, "VectorIndexer", "org.apache.spark.ml.feature.VectorIndexer", "{\"maxCategories\": 20, \"inputCol\": null, \"outputCol\": null, \"handleInvalid\": \"error\"}",
[
        "handleInvalid" => ("getHandleInvalid", get_handle_invalid, param_handle_invalid),
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "maxCategories" => ("getMaxCategories", get_max_categories, param_max_categories),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[("maxCategories", Int), ("inputCol", Str), ("outputCol", Str), ("handleInvalid", Str)],
[
    (set_handle_invalid, "setHandleInvalid", "handleInvalid", Str),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_max_categories, "setMaxCategories", "maxCategories", Int),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_transformer!(PyVectorSizeHint,
VectorSizeHint,
"VectorSizeHint",
"org.apache.spark.ml.feature.VectorSizeHint",
"{\"inputCol\": null, \"size\": null, \"handleInvalid\": \"error\"}",
[
    "handleInvalid" => ("getHandleInvalid", get_handle_invalid, param_handle_invalid),
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "size" => ("getSize", get_size, param_size),
],
[("inputCol", Str), ("size", Int), ("handleInvalid", Str)],
[
    (set_handle_invalid, "setHandleInvalid", "handleInvalid", Str),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_size, "setSize", "size", Int),
]);

feature_transformer!(PyVectorSlicer,
VectorSlicer,
"VectorSlicer",
"org.apache.spark.ml.feature.VectorSlicer",
"{\"inputCol\": null, \"outputCol\": null, \"indices\": null, \"names\": null}",
[
    "indices" => ("getIndices", get_indices, param_indices),
    "inputCol" => ("getInputCol", get_input_col, param_input_col),
    "names" => ("getNames", get_names, param_names),
    "outputCol" => ("getOutputCol", get_output_col, param_output_col),
],
[
    ("inputCol", Str),
    ("outputCol", Str),
    ("indices", IntList),
    ("names", StrList)
],
[
    (set_indices, "setIndices", "indices", IntList),
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_names, "setNames", "names", StrList),
    (set_output_col, "setOutputCol", "outputCol", Str),
]);

feature_estimator!(PyWord2Vec, Word2Vec, PyWord2VecModel, "Word2Vec", "org.apache.spark.ml.feature.Word2Vec", "{\"vectorSize\": 100, \"minCount\": 5, \"numPartitions\": 1, \"stepSize\": 0.025, \"maxIter\": 1, \"seed\": null, \"inputCol\": null, \"outputCol\": null, \"windowSize\": 5, \"maxSentenceLength\": 1000}",
[
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "maxSentenceLength" => (
            "getMaxSentenceLength",
            get_max_sentence_length,
            param_max_sentence_length
        ),
        "minCount" => ("getMinCount", get_min_count, param_min_count),
        "numPartitions" => ("getNumPartitions", get_num_partitions, param_num_partitions),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "seed" => ("getSeed", get_seed, param_seed),
        "stepSize" => ("getStepSize", get_step_size, param_step_size),
        "vectorSize" => ("getVectorSize", get_vector_size, param_vector_size),
        "windowSize" => ("getWindowSize", get_window_size, param_window_size),
],
[("vectorSize", Int), ("minCount", Int), ("numPartitions", Int), ("stepSize", Float), ("maxIter", Int), ("seed", Int), ("inputCol", Str), ("outputCol", Str), ("windowSize", Int), ("maxSentenceLength", Int)],
[
    (set_input_col, "setInputCol", "inputCol", Str),
    (set_max_iter, "setMaxIter", "maxIter", Int),
    (set_max_sentence_length, "setMaxSentenceLength", "maxSentenceLength", Int),
    (set_min_count, "setMinCount", "minCount", Int),
    (set_num_partitions, "setNumPartitions", "numPartitions", Int),
    (set_output_col, "setOutputCol", "outputCol", Str),
    (set_seed, "setSeed", "seed", Int),
    (set_step_size, "setStepSize", "stepSize", Float),
    (set_vector_size, "setVectorSize", "vectorSize", Int),
    (set_window_size, "setWindowSize", "windowSize", Int),
]);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBinarizer>()?;
    m.add_class::<PyBucketedRandomProjectionLSH>()?;
    m.add_class::<PyBucketizer>()?;
    m.add_class::<PyChiSqSelector>()?;
    m.add_class::<PyCountVectorizer>()?;
    m.add_class::<PyDCT>()?;
    m.add_class::<PyElementwiseProduct>()?;
    m.add_class::<PyFeatureHasher>()?;
    m.add_class::<PyHashingTF>()?;
    m.add_class::<PyIDF>()?;
    m.add_class::<PyImputer>()?;
    m.add_class::<PyIndexToString>()?;
    m.add_class::<PyInteraction>()?;
    m.add_class::<PyMaxAbsScaler>()?;
    m.add_class::<PyMinHashLSH>()?;
    m.add_class::<PyMinMaxScaler>()?;
    m.add_class::<PyNGram>()?;
    m.add_class::<PyNormalizer>()?;
    m.add_class::<PyOneHotEncoder>()?;
    m.add_class::<PyPCA>()?;
    m.add_class::<PyPolynomialExpansion>()?;
    m.add_class::<PyQuantileDiscretizer>()?;
    m.add_class::<PyRobustScaler>()?;
    m.add_class::<PyRegexTokenizer>()?;
    m.add_class::<PyRFormula>()?;
    m.add_class::<PySQLTransformer>()?;
    m.add_class::<PyStandardScaler>()?;
    m.add_class::<PyStopWordsRemover>()?;
    m.add_class::<PyStringIndexer>()?;
    m.add_class::<PyTargetEncoder>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<PyUnivariateFeatureSelector>()?;
    m.add_class::<PyVarianceThresholdSelector>()?;
    m.add_class::<PyVectorAssembler>()?;
    m.add_class::<PyVectorIndexer>()?;
    m.add_class::<PyVectorSizeHint>()?;
    m.add_class::<PyVectorSlicer>()?;
    m.add_class::<PyWord2Vec>()?;
    feature_api::register(m)?;
    m.add(
        "_ml_feature_names",
        vec![
            "Binarizer",
            "BucketedRandomProjectionLSH",
            "Bucketizer",
            "ChiSqSelector",
            "CountVectorizer",
            "DCT",
            "ElementwiseProduct",
            "FeatureHasher",
            "HashingTF",
            "IDF",
            "Imputer",
            "IndexToString",
            "Interaction",
            "MaxAbsScaler",
            "MinHashLSH",
            "MinMaxScaler",
            "NGram",
            "Normalizer",
            "OneHotEncoder",
            "PCA",
            "PolynomialExpansion",
            "QuantileDiscretizer",
            "RobustScaler",
            "RegexTokenizer",
            "RFormula",
            "SQLTransformer",
            "StandardScaler",
            "StopWordsRemover",
            "StringIndexer",
            "TargetEncoder",
            "Tokenizer",
            "UnivariateFeatureSelector",
            "VarianceThresholdSelector",
            "VectorAssembler",
            "VectorIndexer",
            "VectorSizeHint",
            "VectorSlicer",
            "Word2Vec",
            "BucketedRandomProjectionLSHModel",
            "ChiSqSelectorModel",
            "CountVectorizerModel",
            "IDFModel",
            "ImputerModel",
            "MaxAbsScalerModel",
            "MinHashLSHModel",
            "MinMaxScalerModel",
            "OneHotEncoderModel",
            "PCAModel",
            "RobustScalerModel",
            "RFormulaModel",
            "StandardScalerModel",
            "StringIndexerModel",
            "TargetEncoderModel",
            "UnivariateFeatureSelectorModel",
            "VarianceThresholdSelectorModel",
            "VectorIndexerModel",
            "Word2VecModel",
        ],
    )?;
    Ok(())
}
