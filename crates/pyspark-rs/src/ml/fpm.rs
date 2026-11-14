// SPDX-License-Identifier: Apache-2.0
//! Native Python frequent-pattern mining APIs.
use super::*;
use spark_connect::ml::{FPGrowth, FPGrowthModel, PrefixSpan};

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
#[pymethods]
impl PyFPGrowth {
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
    ) -> PyResult<PyFPGrowthModel> {
        check_fit_params(params)?;
        let input = dataset_of(dataset, "fit", "select")?;
        let mut estimator = snapshot(slf)?;
        let session = input.spark_session();
        let model = interruptible(slf.py(), &session, || {
            estimator.fit(&input).map_err(Box::new)
        })?;
        Ok(PyFPGrowthModel {
            inner: model,
            unsendable: Vec::new(),
        })
    }
}

#[pyclass(name = "FPGrowthModel", module = "pyspark.ml.connect.fpm")]
pub struct PyFPGrowthModel {
    inner: FPGrowthModel,
    unsendable: Vec<&'static str>,
}
impl Native for PyFPGrowthModel {
    type Core = FPGrowthModel;
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
#[pymethods]
impl PyFPGrowthModel {
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
    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }
    #[pyo3(signature = (dataset, params=None))]
    fn transform(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        check_params(params)?;
        let input = dataset_of(dataset, "transform", "columns")?;
        let mut model = snapshot(slf)?;
        Ok(PyDataFrame::new(model.transform(&input).to_pyerr()?))
    }
    #[getter]
    #[pyo3(name = "freqItemsets")]
    fn freq_itemsets(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.inner.freq_itemsets().to_pyerr()?))
    }
    #[getter]
    #[pyo3(name = "associationRules")]
    fn association_rules(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.inner.association_rules().to_pyerr()?))
    }
}

#[pyclass(name = "PrefixSpan", module = "pyspark.ml.connect.fpm")]
pub struct PyPrefixSpan {
    inner: PrefixSpan,
    unsendable: Vec<&'static str>,
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
#[pymethods]
impl PyPrefixSpan {
    #[new]
    #[pyo3(signature = (*args, **kwargs), text_signature = "(*, minSupport=0.1, maxPatternLength=10, maxLocalProjDBSize=32000000, sequenceCol='sequence')")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: PrefixSpan::new(),
            unsendable: Vec::new(),
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
    #[pyo3(name = "findFrequentSequentialPatterns")]
    fn find_frequent_sequential_patterns(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let input = dataset_of(dataset, "findFrequentSequentialPatterns", "select")?;
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
