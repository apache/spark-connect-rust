// SPDX-License-Identifier: Apache-2.0
//! The typed models of the native feature estimators, as PySpark's `pyspark.ml.feature` has them
//! over Spark Connect (see [`super::object`]): their params, setters, members and class methods.
//! The names of the methods and arguments are PySpark's, so that the keywords match.
#![allow(non_snake_case)]
use super::*;
use pyo3::types::{PyDict, PyTuple};
use spark_connect::ml as core;
use spark_connect::ml::{Model, ReadResult, Transformer};

/// The active session of the Python of a class method that builds a model (PySpark's
/// `invoke_helper_attr` of a remote session), which raises the AssertionError of PySpark without
/// one.
fn helper_session(py: Python<'_>) -> PyResult<spark_connect::SparkSession> {
    let active = crate::session::active_session(py)
        .ok_or_else(|| pyo3::exceptions::PyAssertionError::new_err(()))?;
    let session = active.borrow(py).session.clone();
    Ok(session)
}

/// The read of a model that a class method built: its object, uid and params, as a loaded
/// model has them (see `ModelCore::loaded`).
fn built<M: Model>(model: &M) -> ReadResult {
    ReadResult {
        object: model.ml_ref().cloned(),
        uid: Transformer::operator(model).uid.clone(),
        params: Transformer::params(model).clone(),
    }
}

native_model! {
    PyBucketedRandomProjectionLSHModel = "BucketedRandomProjectionLSHModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.BucketedRandomProjectionLSHModel";
    params: [
        "bucketLength" => ("getBucketLength", get_bucket_length, param_bucket_length),
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "numHashTables" => ("getNumHashTables", get_num_hash_tables, param_num_hash_tables),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
    ];
    setters: [
        "setInputCol" => (set_input_col, "inputCol"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];

    /// `approxNearestNeighbors`, as PySpark's: the server's `approxNearestNeighbors` of the positional arguments only, as PySpark's
    /// `try_remote_attribute_relation` sends them over Spark Connect, which drops keyword arguments.
    #[pyo3(name = "approxNearestNeighbors", signature = (*args, **_kwargs), text_signature = "($self, dataset, key, numNearestNeighbors, distCol='distCol')")]
    fn approx_nearest_neighbors(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let target = slf.try_borrow()?.core.target()?;
        let args: Vec<Bound<'_, PyAny>> = args.iter().collect();
        object::attribute_relation_py(&target, "approxNearestNeighbors", &args)
    }

    /// `approxSimilarityJoin`, as PySpark's: the server's `approxSimilarityJoin` of the positional arguments only, as PySpark's
    /// `try_remote_attribute_relation` sends them over Spark Connect, which drops keyword arguments.
    #[pyo3(name = "approxSimilarityJoin", signature = (*args, **_kwargs), text_signature = "($self, datasetA, datasetB, threshold, distCol='distCol')")]
    fn approx_similarity_join(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let target = slf.try_borrow()?.core.target()?;
        let args: Vec<Bound<'_, PyAny>> = args.iter().collect();
        object::attribute_relation_py(&target, "approxSimilarityJoin", &args)
    }
}

native_model! {
    PyChiSqSelectorModel = "ChiSqSelectorModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.ChiSqSelectorModel";
    params: [
        "fdr" => ("getFdr", get_fdr, param_fdr),
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "fpr" => ("getFpr", get_fpr, param_fpr),
        "fwe" => ("getFwe", get_fwe, param_fwe),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "numTopFeatures" => ("getNumTopFeatures", get_num_top_features, param_num_top_features),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "percentile" => ("getPercentile", get_percentile, param_percentile),
        "selectorType" => ("getSelectorType", get_selector_type, param_selector_type),
    ];
    setters: [
        "setFeaturesCol" => (set_features_col, "featuresCol"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
        "selectedFeatures" => (selected_features, "selectedFeatures"),
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];
}

native_model! {
    PyCountVectorizerModel = "CountVectorizerModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.CountVectorizerModel";
    params: [
        "binary" => ("getBinary", get_binary, param_binary),
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "maxDF" => ("getMaxDF", get_max_df, param_max_df),
        "minDF" => ("getMinDF", get_min_df, param_min_df),
        "minTF" => ("getMinTF", get_min_tf, param_min_tf),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "vocabSize" => ("getVocabSize", get_vocab_size, param_vocab_size),
    ];
    setters: [
        "setBinary" => (set_binary, "binary"),
        "setInputCol" => (set_input_col, "inputCol"),
        "setMinTF" => (set_min_tf, "minTF"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
        "vocabulary" => (vocabulary, "vocabulary"),
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];

    /// `CountVectorizerModel.from_vocabulary`, as PySpark's `from_vocabulary`: the model that the
    /// server builds of `vocabulary` (of a new uid), which transforms `inputCol`, with `outputCol`,
    /// `minTF` and `binary` when they are given. An empty vocabulary raises PySpark's ValueError.
    #[classmethod]
    #[pyo3(signature = (vocabulary, inputCol, outputCol=None, minTF=None, binary=None))]
    fn from_vocabulary(
        cls: &Bound<'_, PyType>,
        vocabulary: Vec<String>,
        inputCol: String,
        outputCol: Option<String>,
        minTF: Option<f64>,
        binary: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let py = cls.py();
        let session = helper_session(py)?;
        let model = core::CountVectorizerModel::from_vocabulary(
            &session,
            vocabulary,
            &inputCol,
            outputCol.as_deref(),
            minTF,
            binary,
        )
        .to_pyerr()?;
        PyCountVectorizerModel::__from_read(py, built(&model))
    }
}

native_model! {
    PyIDFModel = "IDFModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.IDFModel";
    params: [
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "minDocFreq" => ("getMinDocFreq", get_min_doc_freq, param_min_doc_freq),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
    ];
    setters: [
        "setInputCol" => (set_input_col, "inputCol"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
        "docFreq" => (doc_freq, "docFreq"),
        "idf" => (idf, "idf"),
        "numDocs" => (num_docs, "numDocs"),
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];
}

native_model! {
    PyImputerModel = "ImputerModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.ImputerModel";
    params: [
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
        "missingValue" => ("getMissingValue", get_missing_value, param_missing_value),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "outputCols" => ("getOutputCols", get_output_cols, param_output_cols),
        "relativeError" => ("getRelativeError", get_relative_error, param_relative_error),
        "strategy" => ("getStrategy", get_strategy, param_strategy),
    ];
    setters: [
        "setInputCol" => (set_input_col, "inputCol"),
        "setInputCols" => (set_input_cols, "inputCols"),
        "setOutputCol" => (set_output_col, "outputCol"),
        "setOutputCols" => (set_output_cols, "outputCols"),
    ];
    properties: [
    ];
    methods: [];
    relations: [
        "surrogateDF" => (surrogate_df, "surrogateDF"),
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];
}

native_model! {
    PyMaxAbsScalerModel = "MaxAbsScalerModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.MaxAbsScalerModel";
    params: [
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
    ];
    setters: [
        "setInputCol" => (set_input_col, "inputCol"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
        "maxAbs" => (max_abs, "maxAbs"),
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];
}

native_model! {
    PyMinHashLSHModel = "MinHashLSHModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.MinHashLSHModel";
    params: [
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "numHashTables" => ("getNumHashTables", get_num_hash_tables, param_num_hash_tables),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
    ];
    setters: [
        "setInputCol" => (set_input_col, "inputCol"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];

    /// `approxNearestNeighbors`, as PySpark's: the server's `approxNearestNeighbors` of the positional arguments only, as PySpark's
    /// `try_remote_attribute_relation` sends them over Spark Connect, which drops keyword arguments.
    #[pyo3(name = "approxNearestNeighbors", signature = (*args, **_kwargs), text_signature = "($self, dataset, key, numNearestNeighbors, distCol='distCol')")]
    fn approx_nearest_neighbors(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let target = slf.try_borrow()?.core.target()?;
        let args: Vec<Bound<'_, PyAny>> = args.iter().collect();
        object::attribute_relation_py(&target, "approxNearestNeighbors", &args)
    }

    /// `approxSimilarityJoin`, as PySpark's: the server's `approxSimilarityJoin` of the positional arguments only, as PySpark's
    /// `try_remote_attribute_relation` sends them over Spark Connect, which drops keyword arguments.
    #[pyo3(name = "approxSimilarityJoin", signature = (*args, **_kwargs), text_signature = "($self, datasetA, datasetB, threshold, distCol='distCol')")]
    fn approx_similarity_join(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let target = slf.try_borrow()?.core.target()?;
        let args: Vec<Bound<'_, PyAny>> = args.iter().collect();
        object::attribute_relation_py(&target, "approxSimilarityJoin", &args)
    }
}

native_model! {
    PyMinMaxScalerModel = "MinMaxScalerModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.MinMaxScalerModel";
    params: [
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "max" => ("getMax", get_max, param_max),
        "min" => ("getMin", get_min, param_min),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
    ];
    setters: [
        "setInputCol" => (set_input_col, "inputCol"),
        "setMax" => (set_max, "max"),
        "setMin" => (set_min, "min"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
        "originalMax" => (original_max, "originalMax"),
        "originalMin" => (original_min, "originalMin"),
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];
}

native_model! {
    PyOneHotEncoderModel = "OneHotEncoderModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.OneHotEncoderModel";
    params: [
        "dropLast" => ("getDropLast", get_drop_last, param_drop_last),
        "handleInvalid" => ("getHandleInvalid", get_handle_invalid, param_handle_invalid),
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "outputCols" => ("getOutputCols", get_output_cols, param_output_cols),
    ];
    setters: [
        "setDropLast" => (set_drop_last, "dropLast"),
        "setHandleInvalid" => (set_handle_invalid, "handleInvalid"),
        "setInputCol" => (set_input_col, "inputCol"),
        "setInputCols" => (set_input_cols, "inputCols"),
        "setOutputCol" => (set_output_col, "outputCol"),
        "setOutputCols" => (set_output_cols, "outputCols"),
    ];
    properties: [
        "categorySizes" => (category_sizes, "categorySizes"),
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];
}

native_model! {
    PyPCAModel = "PCAModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.PCAModel";
    params: [
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "k" => ("getK", get_k, param_k),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
    ];
    setters: [
        "setInputCol" => (set_input_col, "inputCol"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
        "explainedVariance" => (explained_variance, "explainedVariance"),
        "pc" => (pc, "pc"),
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];
}

native_model! {
    PyRobustScalerModel = "RobustScalerModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.RobustScalerModel";
    params: [
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "lower" => ("getLower", get_lower, param_lower),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "relativeError" => ("getRelativeError", get_relative_error, param_relative_error),
        "upper" => ("getUpper", get_upper, param_upper),
        "withCentering" => ("getWithCentering", get_with_centering, param_with_centering),
        "withScaling" => ("getWithScaling", get_with_scaling, param_with_scaling),
    ];
    setters: [
        "setInputCol" => (set_input_col, "inputCol"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
        "median" => (median, "median"),
        "range" => (range, "range"),
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];
}

native_model! {
    PyRFormulaModel = "RFormulaModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.RFormulaModel";
    params: [
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
    ];
    setters: [];
    properties: [
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];
}

native_model! {
    PyStandardScalerModel = "StandardScalerModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.StandardScalerModel";
    params: [
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "withMean" => ("getWithMean", get_with_mean, param_with_mean),
        "withStd" => ("getWithStd", get_with_std, param_with_std),
    ];
    setters: [
        "setInputCol" => (set_input_col, "inputCol"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
        "mean" => (mean, "mean"),
        "std" => (std, "std"),
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];
}

native_model! {
    PyStringIndexerModel = "StringIndexerModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.StringIndexerModel";
    params: [
        "handleInvalid" => ("getHandleInvalid", get_handle_invalid, param_handle_invalid),
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "outputCols" => ("getOutputCols", get_output_cols, param_output_cols),
        "stringOrderType" => ("getStringOrderType", get_string_order_type, param_string_order_type),
    ];
    setters: [
        "setHandleInvalid" => (set_handle_invalid, "handleInvalid"),
        "setInputCol" => (set_input_col, "inputCol"),
        "setInputCols" => (set_input_cols, "inputCols"),
        "setOutputCol" => (set_output_col, "outputCol"),
        "setOutputCols" => (set_output_cols, "outputCols"),
    ];
    properties: [
        "labels" => (labels, "labels"),
        "labelsArray" => (labels_array, "labelsArray"),
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];

    /// `StringIndexerModel.from_labels`, as PySpark's: the model that the server builds of `labels`
    /// (of a new uid), which indexes `inputCol`, with `outputCol` and `handleInvalid` when given.
    #[classmethod]
    #[pyo3(signature = (labels, inputCol, outputCol=None, handleInvalid=None))]
    fn from_labels(
        cls: &Bound<'_, PyType>,
        labels: Vec<String>,
        inputCol: String,
        outputCol: Option<String>,
        handleInvalid: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let py = cls.py();
        let session = helper_session(py)?;
        let model = core::StringIndexerModel::from_labels(
            &session,
            labels,
            &inputCol,
            outputCol.as_deref(),
            handleInvalid.as_deref(),
        )
        .to_pyerr()?;
        PyStringIndexerModel::__from_read(py, built(&model))
    }

    /// `StringIndexerModel.from_arrays_of_labels`, as PySpark's: the model of the labels of each of
    /// `inputCols`, with `outputCols` and `handleInvalid` when given.
    #[classmethod]
    #[pyo3(signature = (arrayOfLabels, inputCols, outputCols=None, handleInvalid=None))]
    fn from_arrays_of_labels(
        cls: &Bound<'_, PyType>,
        arrayOfLabels: Vec<Vec<String>>,
        inputCols: Vec<String>,
        outputCols: Option<Vec<String>>,
        handleInvalid: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let py = cls.py();
        let session = helper_session(py)?;
        let model = core::StringIndexerModel::from_arrays_of_labels(
            &session,
            arrayOfLabels,
            inputCols,
            outputCols,
            handleInvalid.as_deref(),
        )
        .to_pyerr()?;
        PyStringIndexerModel::__from_read(py, built(&model))
    }
}

native_model! {
    PyTargetEncoderModel = "TargetEncoderModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.TargetEncoderModel";
    params: [
        "handleInvalid" => ("getHandleInvalid", get_handle_invalid, param_handle_invalid),
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "inputCols" => ("getInputCols", get_input_cols, param_input_cols),
        "labelCol" => ("getLabelCol", get_label_col, param_label_col),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "outputCols" => ("getOutputCols", get_output_cols, param_output_cols),
        "smoothing" => ("getSmoothing", get_smoothing, param_smoothing),
        "targetType" => ("getTargetType", get_target_type, param_target_type),
    ];
    setters: [
        "setHandleInvalid" => (set_handle_invalid, "handleInvalid"),
        "setInputCol" => (set_input_col, "inputCol"),
        "setInputCols" => (set_input_cols, "inputCols"),
        "setOutputCol" => (set_output_col, "outputCol"),
        "setOutputCols" => (set_output_cols, "outputCols"),
        "setSmoothing" => (set_smoothing, "smoothing"),
    ];
    properties: [
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];
}

native_model! {
    PyUnivariateFeatureSelectorModel = "UnivariateFeatureSelectorModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.UnivariateFeatureSelectorModel";
    params: [
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
    ];
    setters: [
        "setFeaturesCol" => (set_features_col, "featuresCol"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
        "selectedFeatures" => (selected_features, "selectedFeatures"),
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];
}

native_model! {
    PyVarianceThresholdSelectorModel = "VarianceThresholdSelectorModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.VarianceThresholdSelectorModel";
    params: [
        "featuresCol" => ("getFeaturesCol", get_features_col, param_features_col),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
        "varianceThreshold" => (
            "getVarianceThreshold",
            get_variance_threshold,
            param_variance_threshold
        ),
    ];
    setters: [
        "setFeaturesCol" => (set_features_col, "featuresCol"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
        "selectedFeatures" => (selected_features, "selectedFeatures"),
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];
}

native_model! {
    PyVectorIndexerModel = "VectorIndexerModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.VectorIndexerModel";
    params: [
        "handleInvalid" => ("getHandleInvalid", get_handle_invalid, param_handle_invalid),
        "inputCol" => ("getInputCol", get_input_col, param_input_col),
        "maxCategories" => ("getMaxCategories", get_max_categories, param_max_categories),
        "outputCol" => ("getOutputCol", get_output_col, param_output_col),
    ];
    setters: [
        "setInputCol" => (set_input_col, "inputCol"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
        "numFeatures" => (num_features, "numFeatures"),
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];

    /// `VectorIndexerModel.categoryMaps`, as PySpark's: the rows of the model's `categoryMapsDF`
    /// (an attribute relation, which Spark Connect allows, unlike `categoryMaps`), of each feature
    /// index, a dict from each original value to its category index, in the order of the rows.
    #[getter]
    #[pyo3(name = "categoryMaps")]
    fn category_maps_of(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let target = slf.try_borrow()?.core.target()?;
        let df = object::attribute_relation_py(&target, "categoryMapsDF", &[])?;
        let rows = Py::new(py, df)?.bind(py).call_method0("collect")?;
        let maps = PyDict::new(py);
        for row in rows.try_iter()? {
            let row = row?;
            let feature = row.getattr("featureIndex")?.extract::<i64>()?;
            let value = row.getattr("originalValue")?.extract::<f64>()?;
            let category = row.getattr("categoryIndex")?.extract::<i64>()?;
            let values = match maps.get_item(feature)? {
                Some(values) => values.cast_into::<PyDict>()?,
                None => {
                    let values = PyDict::new(py);
                    maps.set_item(feature, &values)?;
                    values
                }
            };
            values.set_item(value, category)?;
        }
        Ok(maps.into_any().unbind())
    }
}

native_model! {
    PyWord2VecModel = "Word2VecModel" in "pyspark.ml.connect.feature",
        jvm: "org.apache.spark.ml.feature.Word2VecModel";
    params: [
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
    ];
    setters: [
        "setInputCol" => (set_input_col, "inputCol"),
        "setOutputCol" => (set_output_col, "outputCol"),
    ];
    properties: [
    ];
    methods: [];
    relations: [
    ];
    relation_methods: [
    ];
    summary: None;
    cached: [];

    /// `findSynonyms`, as PySpark's: the server's `findSynonyms` of the positional arguments only, as PySpark's
    /// `try_remote_attribute_relation` sends them over Spark Connect, which drops keyword arguments.
    #[pyo3(name = "findSynonyms", signature = (*args, **_kwargs), text_signature = "($self, word, num)")]
    fn find_synonyms(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let target = slf.try_borrow()?.core.target()?;
        let args: Vec<Bound<'_, PyAny>> = args.iter().collect();
        object::attribute_relation_py(&target, "findSynonyms", &args)
    }

    /// `Word2VecModel.getVectors`: the DataFrame of each word and its vector, of the server's
    /// `getVectors` (an attribute relation).
    #[pyo3(name = "getVectors", text_signature = "($self)")]
    fn get_vectors(slf: &Bound<'_, Self>) -> PyResult<PyDataFrame> {
        let target = slf.try_borrow()?.core.target()?;
        object::attribute_relation_py(&target, "getVectors", &[])
    }

    /// `Word2VecModel.findSynonymsArray`, as PySpark's: the words and similarities of
    /// `findSynonyms`, as `(str, float)` tuples.
    #[pyo3(name = "findSynonymsArray", signature = (word, num), text_signature = "($self, word, num)")]
    fn find_synonyms_array(
        slf: &Bound<'_, Self>,
        word: &Bound<'_, PyAny>,
        num: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<(String, f64)>> {
        let rows = slf
            .call_method1("findSynonyms", (word, num))?
            .call_method0("collect")?;
        rows.try_iter()?
            .map(|row| {
                let row = row?;
                Ok((
                    row.getattr("word")?.str()?.extract::<String>()?,
                    row.getattr("similarity")?.extract::<f64>()?,
                ))
            })
            .collect()
    }
}

/// Register the typed feature models.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBucketedRandomProjectionLSHModel>()?;
    m.add_class::<PyChiSqSelectorModel>()?;
    m.add_class::<PyCountVectorizerModel>()?;
    m.add_class::<PyIDFModel>()?;
    m.add_class::<PyImputerModel>()?;
    m.add_class::<PyMaxAbsScalerModel>()?;
    m.add_class::<PyMinHashLSHModel>()?;
    m.add_class::<PyMinMaxScalerModel>()?;
    m.add_class::<PyOneHotEncoderModel>()?;
    m.add_class::<PyPCAModel>()?;
    m.add_class::<PyRobustScalerModel>()?;
    m.add_class::<PyRFormulaModel>()?;
    m.add_class::<PyStandardScalerModel>()?;
    m.add_class::<PyStringIndexerModel>()?;
    m.add_class::<PyTargetEncoderModel>()?;
    m.add_class::<PyUnivariateFeatureSelectorModel>()?;
    m.add_class::<PyVarianceThresholdSelectorModel>()?;
    m.add_class::<PyVectorIndexerModel>()?;
    m.add_class::<PyWord2VecModel>()?;
    Ok(())
}
