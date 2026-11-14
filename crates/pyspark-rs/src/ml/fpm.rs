// SPDX-License-Identifier: Apache-2.0
//! Native Python frequent-pattern mining APIs: the estimators FPGrowth and PrefixSpan, and the
//! typed model FPGrowthModel, as PySpark's `pyspark.ml.fpm` has them over Spark Connect.
use super::*;
use spark_connect::ml::{FPGrowth, PrefixSpan};

const FPGROWTH_PARAMS: &[(&str, Kind)] = &[
    ("itemsCol", Kind::Str),
    ("minConfidence", Kind::Float),
    ("minSupport", Kind::Float),
    ("numPartitions", Kind::Int),
    ("predictionCol", Kind::Str),
];

const PREFIX_SPAN_PARAMS: &[(&str, Kind)] = &[
    ("maxLocalProjDBSize", Kind::Int),
    ("maxPatternLength", Kind::Int),
    ("minSupport", Kind::Float),
    ("sequenceCol", Kind::Str),
];

/// FPGrowth, as PySpark's `FPGrowth` over Spark Connect: an estimator of its params.
#[pyclass(name = "FPGrowth", module = "pyspark.ml.connect.fpm")]
pub struct PyFPGrowth {
    inner: FPGrowth,
    unsendable: Vec<&'static str>,
}
impl Native for PyFPGrowth {
    type Core = FPGrowth;
    fn inner(&self) -> &Self::Core {
        &self.inner
    }
    fn params(&self) -> &Params {
        self.inner.params()
    }
    fn params_mut(&mut self) -> &mut Params {
        self.inner.params_mut()
    }
    fn unsendable(&self) -> &[&'static str] {
        &self.unsendable
    }
    fn unsendable_mut(&mut self) -> &mut Vec<&'static str> {
        &mut self.unsendable
    }
}

native_pymethods! {
    PyFPGrowth;
    jvm: "org.apache.spark.ml.fpm.FPGrowth", op: Estimator;
    params: [
        "itemsCol" => ("getItemsCol", get_items_col, param_items_col),
        "minConfidence" => ("getMinConfidence", get_min_confidence, param_min_confidence),
        "minSupport" => ("getMinSupport", get_min_support, param_min_support),
        "numPartitions" => ("getNumPartitions", get_num_partitions, param_num_partitions),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, minSupport=0.3, minConfidence=0.8, itemsCol='items', predictionCol='prediction', numPartitions=None)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: FPGrowth::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "FPGrowth", &[], FPGROWTH_PARAMS, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, minSupport=0.3, minConfidence=0.8, itemsCol='items', predictionCol='prediction', numPartitions=None)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "FPGrowth", FPGROWTH_PARAMS, args, kwargs)
    }
    #[pyo3(name = "setItemsCol")]
    fn set_items_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "itemsCol", Kind::Str, value)
    }
    #[pyo3(name = "setMinConfidence")]
    fn set_min_confidence<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minConfidence", Kind::Float, value)
    }
    #[pyo3(name = "setMinSupport")]
    fn set_min_support<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minSupport", Kind::Float, value)
    }
    #[pyo3(name = "setNumPartitions")]
    fn set_num_partitions<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "numPartitions", Kind::Int, value)
    }
    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }
    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<PyFPGrowthModel>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                PyFPGrowthModel::fitted(py, model, df, object::SummaryDataset::None)
            })
        })
    }

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

impl params_api::NativeInstance for PyFPGrowth {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        Estimator::operator(&self.inner).uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        Estimator::operator_mut(&mut self.inner).uid = uid.to_string();
    }
}

native_model! {
    PyFPGrowthModel = "FPGrowthModel" in "pyspark.ml.connect.fpm",
        jvm: "org.apache.spark.ml.fpm.FPGrowthModel";
    params: [
        "itemsCol" => ("getItemsCol", get_items_col, param_items_col),
        "minConfidence" => ("getMinConfidence", get_min_confidence, param_min_confidence),
        "minSupport" => ("getMinSupport", get_min_support, param_min_support),
        "numPartitions" => ("getNumPartitions", get_num_partitions, param_num_partitions),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
    ];
    setters: [
        "setItemsCol" => (set_items_col, "itemsCol"),
        "setMinConfidence" => (set_min_confidence, "minConfidence"),
        "setPredictionCol" => (set_prediction_col, "predictionCol"),
    ];
    properties: [];
    methods: [];
    relations: [
        "associationRules" => (association_rules, "associationRules"),
        "freqItemsets" => (freq_itemsets, "freqItemsets"),
    ];
    relation_methods: [];
    summary: None;
    cached: [];
}

/// Native PrefixSpan: it mines sequential patterns directly, with no model, and has no persistence
/// in Spark.
#[pyclass(name = "PrefixSpan", module = "pyspark.ml.connect.fpm")]
pub struct PyPrefixSpan {
    inner: PrefixSpan,
    unsendable: Vec<&'static str>,
    /// The uid of the object, as `Identifiable` makes it of the class name (PrefixSpan has no JVM
    /// operator to hold it).
    uid: String,
}
impl Native for PyPrefixSpan {
    type Core = PrefixSpan;
    fn inner(&self) -> &Self::Core {
        &self.inner
    }
    fn params(&self) -> &Params {
        self.inner.params()
    }
    fn params_mut(&mut self) -> &mut Params {
        self.inner.params_mut()
    }
    fn unsendable(&self) -> &[&'static str] {
        &self.unsendable
    }
    fn unsendable_mut(&mut self) -> &mut Vec<&'static str> {
        &mut self.unsendable
    }
}

impl params_api::NativeInstance for PyPrefixSpan {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        Py::new(
            py,
            Self {
                inner: self.inner.clone(),
                unsendable: self.unsendable.clone(),
                uid: self.uid.clone(),
            },
        )
    }

    fn uid(&self) -> String {
        self.uid.clone()
    }

    fn set_uid(&mut self, uid: &str) {
        self.uid = uid.to_string();
    }
}

native_pymethods! {
    PyPrefixSpan;
    persistence: python;
    params: [
        "maxLocalProjDBSize" => (
            "getMaxLocalProjDBSize",
            get_max_local_proj_dbsize,
            param_max_local_proj_dbsize
        ),
        "maxPatternLength" => (
            "getMaxPatternLength",
            get_max_pattern_length,
            param_max_pattern_length
        ),
        "minSupport" => ("getMinSupport", get_min_support, param_min_support),
        "sequenceCol" => ("getSequenceCol", get_sequence_col, param_sequence_col),
    ];
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, minSupport=0.1, maxPatternLength=10, maxLocalProjDBSize=32000000, sequenceCol='sequence')")]
    fn new(
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let mut this = Self {
            inner: PrefixSpan::new(),
            unsendable: Vec::new(),
            uid: params_api::random_uid(py, "PrefixSpan")?,
        };
        init_params(
            &mut this,
            "PrefixSpan",
            &[],
            PREFIX_SPAN_PARAMS,
            args,
            kwargs,
        )?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, minSupport=0.1, maxPatternLength=10, maxLocalProjDBSize=32000000, sequenceCol='sequence')"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "PrefixSpan", PREFIX_SPAN_PARAMS, args, kwargs)
    }
    #[pyo3(name = "setMaxLocalProjDBSize")]
    fn set_max_local_proj_db_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxLocalProjDBSize", Kind::Int, value)
    }
    #[pyo3(name = "setMaxPatternLength")]
    fn set_max_pattern_length<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxPatternLength", Kind::Int, value)
    }
    #[pyo3(name = "setMinSupport")]
    fn set_min_support<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "minSupport", Kind::Float, value)
    }
    #[pyo3(name = "setSequenceCol")]
    fn set_sequence_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "sequenceCol", Kind::Str, value)
    }
    /// A copy of the same uid and params, with `extra`, as `Params.copy`.
    #[pyo3(name = "copy", signature = (extra=None))]
    fn __params_copy<'py>(
        slf: &Bound<'py, Self>,
        extra: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        params_api::copy(slf, extra)
    }
    #[pyo3(name = "findFrequentSequentialPatterns")]
    fn find_frequent_sequential_patterns(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let input = helper_dataset_of(dataset, "findFrequentSequentialPatterns")?;
        let algorithm = snapshot(slf)?;
        Ok(PyDataFrame::new(
            algorithm
                .find_frequent_sequential_patterns(&input)
                .to_pyerr()?,
        ))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFPGrowth>()?;
    m.add_class::<PyFPGrowthModel>()?;
    m.add_class::<PyPrefixSpan>()?;
    Ok(())
}
