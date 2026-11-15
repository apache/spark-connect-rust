// SPDX-License-Identifier: Apache-2.0
//! Native ALS fitting, prediction, factors and recommendations.
use super::*;
use spark_connect::ml::ALS;

const ALS_PARAMS: &[(&str, Kind)] = &[
    ("rank", Kind::Int),
    ("maxIter", Kind::Int),
    ("regParam", Kind::Float),
    ("numUserBlocks", Kind::Int),
    ("numItemBlocks", Kind::Int),
    ("implicitPrefs", Kind::Bool),
    ("alpha", Kind::Float),
    ("userCol", Kind::Str),
    ("itemCol", Kind::Str),
    ("seed", Kind::Int),
    ("ratingCol", Kind::Str),
    ("nonnegative", Kind::Bool),
    ("checkpointInterval", Kind::Int),
    ("intermediateStorageLevel", Kind::Str),
    ("finalStorageLevel", Kind::Str),
    ("coldStartStrategy", Kind::Str),
    ("blockSize", Kind::Int),
];

/// ALS, as PySpark's `ALS` over Spark Connect: an estimator of its params.
#[pyclass(subclass, name = "ALS", module = "pyspark.ml.connect.recommendation")]
pub struct PyALS {
    inner: ALS,
    unsendable: Vec<&'static str>,
}
impl Native for PyALS {
    type Core = ALS;
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
    PyALS;
    jvm: "org.apache.spark.ml.recommendation.ALS", op: Estimator;
    params: [
        "alpha" => ("getAlpha", get_alpha, param_alpha),
        "blockSize" => ("getBlockSize", get_block_size, param_block_size),
        "checkpointInterval" => (
            "getCheckpointInterval",
            get_checkpoint_interval,
            param_checkpoint_interval
        ),
        "coldStartStrategy" => (
            "getColdStartStrategy",
            get_cold_start_strategy,
            param_cold_start_strategy
        ),
        "finalStorageLevel" => (
            "getFinalStorageLevel",
            get_final_storage_level,
            param_final_storage_level
        ),
        "implicitPrefs" => ("getImplicitPrefs", get_implicit_prefs, param_implicit_prefs),
        "intermediateStorageLevel" => (
            "getIntermediateStorageLevel",
            get_intermediate_storage_level,
            param_intermediate_storage_level
        ),
        "itemCol" => ("getItemCol", get_item_col, param_item_col),
        "maxIter" => ("getMaxIter", get_max_iter, param_max_iter),
        "nonnegative" => ("getNonnegative", get_nonnegative, param_nonnegative),
        "numItemBlocks" => ("getNumItemBlocks", get_num_item_blocks, param_num_item_blocks),
        "numUserBlocks" => ("getNumUserBlocks", get_num_user_blocks, param_num_user_blocks),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "rank" => ("getRank", get_rank, param_rank),
        "ratingCol" => ("getRatingCol", get_rating_col, param_rating_col),
        "regParam" => ("getRegParam", get_reg_param, param_reg_param),
        "seed" => ("getSeed", get_seed, param_seed),
        "userCol" => ("getUserCol", get_user_col, param_user_col),
    ];
    #[new]
    #[pyo3(signature=(*args,**kwargs),text_signature="(*, rank=10, maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10, implicitPrefs=False, alpha=1.0, userCol='user', itemCol='item', seed=None, ratingCol='rating', nonnegative=False, checkpointInterval=10, intermediateStorageLevel='MEMORY_AND_DISK', finalStorageLevel='MEMORY_AND_DISK', coldStartStrategy='nan', blockSize=4096)")]
    fn new(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut this = Self {
            inner: ALS::new(),
            unsendable: Vec::new(),
        };
        init_params(&mut this, "ALS", &[], ALS_PARAMS, args, kwargs)?;
        Ok(this)
    }
    #[pyo3(
        name = "setParams",
        signature = (*args, **kwargs),
        text_signature = "(self, *, rank=10, maxIter=10, regParam=0.1, numUserBlocks=10, numItemBlocks=10, implicitPrefs=False, alpha=1.0, userCol='user', itemCol='item', seed=None, ratingCol='rating', nonnegative=False, checkpointInterval=10, intermediateStorageLevel='MEMORY_AND_DISK', finalStorageLevel='MEMORY_AND_DISK', coldStartStrategy='nan', blockSize=4096)"
    )]
    fn set_params<'py>(
        slf: Bound<'py, Self>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, Self>> {
        set_params_of(slf, "ALS", ALS_PARAMS, args, kwargs)
    }
    #[pyo3(name = "setAlpha")]
    fn set_alpha<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "alpha", Kind::Float, value)
    }
    #[pyo3(name = "setBlockSize")]
    fn set_block_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "blockSize", Kind::Int, value)
    }
    #[pyo3(name = "setCheckpointInterval")]
    fn set_checkpoint_interval<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "checkpointInterval", Kind::Int, value)
    }
    #[pyo3(name = "setColdStartStrategy")]
    fn set_cold_start_strategy<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "coldStartStrategy", Kind::Str, value)
    }
    #[pyo3(name = "setFinalStorageLevel")]
    fn set_final_storage_level<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "finalStorageLevel", Kind::Str, value)
    }
    #[pyo3(name = "setImplicitPrefs")]
    fn set_implicit_prefs<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "implicitPrefs", Kind::Bool, value)
    }
    #[pyo3(name = "setIntermediateStorageLevel")]
    fn set_intermediate_storage_level<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "intermediateStorageLevel", Kind::Str, value)
    }
    #[pyo3(name = "setItemCol")]
    fn set_item_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "itemCol", Kind::Str, value)
    }
    #[pyo3(name = "setMaxIter")]
    fn set_max_iter<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "maxIter", Kind::Int, value)
    }
    #[pyo3(name = "setNonnegative")]
    fn set_nonnegative<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "nonnegative", Kind::Bool, value)
    }
    #[pyo3(name = "setNumItemBlocks")]
    fn set_num_item_blocks<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "numItemBlocks", Kind::Int, value)
    }
    #[pyo3(name = "setNumUserBlocks")]
    fn set_num_user_blocks<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "numUserBlocks", Kind::Int, value)
    }
    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }
    #[pyo3(name = "setRank")]
    fn set_rank<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "rank", Kind::Int, value)
    }
    #[pyo3(name = "setRatingCol")]
    fn set_rating_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "ratingCol", Kind::Str, value)
    }
    #[pyo3(name = "setRegParam")]
    fn set_reg_param<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "regParam", Kind::Float, value)
    }
    #[pyo3(name = "setSeed")]
    fn set_seed<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "seed", Kind::Int, value)
    }
    #[pyo3(name = "setUserCol")]
    fn set_user_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "userCol", Kind::Str, value)
    }
    /// Set both the user and the item block counts, as `ALS.setNumBlocks`.
    #[pyo3(name = "setNumBlocks")]
    fn set_num_blocks<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let users = convert("numUserBlocks", Kind::Int, value)?;
        let items = convert("numItemBlocks", Kind::Int, value)?;
        {
            let mut this = slf.try_borrow_mut()?;
            apply(&mut *this, "numUserBlocks", users);
            apply(&mut *this, "numItemBlocks", items);
        }
        Ok(slf)
    }
    #[pyo3(signature = (dataset, params=None))]
    fn fit(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Fitted<Py<PyALSModel>>> {
        fit_with(slf, dataset, params, || {
            fit_native_model(slf, dataset, |py, model, df| {
                PyALSModel::fitted(py, model, df, object::SummaryDataset::None)
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
    ) -> PyResult<Py<params_api::PyFitMultipleIterator>> {
        params_api::fit_multiple(slf.as_any(), dataset, paramMaps)
    }
}

impl params_api::NativeInstance for PyALS {
    fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
        crate::ml::object::new_py(
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
    PyALSModel = "ALSModel" in "pyspark.ml.connect.recommendation",
        jvm: "org.apache.spark.ml.recommendation.ALSModel";
    params: [
        "blockSize" => ("getBlockSize", get_block_size, param_block_size),
        "coldStartStrategy" => (
            "getColdStartStrategy",
            get_cold_start_strategy,
            param_cold_start_strategy
        ),
        "itemCol" => ("getItemCol", get_item_col, param_item_col),
        "predictionCol" => ("getPredictionCol", get_prediction_col, param_prediction_col),
        "userCol" => ("getUserCol", get_user_col, param_user_col),
    ];
    setters: [
        "setBlockSize" => (set_block_size, "blockSize"),
        "setColdStartStrategy" => (set_cold_start_strategy, "coldStartStrategy"),
        "setItemCol" => (set_item_col, "itemCol"),
        "setPredictionCol" => (set_prediction_col, "predictionCol"),
        "setUserCol" => (set_user_col, "userCol"),
    ];
    // The rank of the factors, which PySpark's `rank` reads of the server on each access.
    properties: [
        "rank" => (rank, "rank"),
    ];
    methods: [];
    relations: [
        "itemFactors" => (item_factors, "itemFactors"),
        "userFactors" => (user_factors, "userFactors"),
    ];
    relation_methods: [];
    summary: None;
    cached: [];

    /// `recommendForAllUsers`, as PySpark's (see `recommend`).
    #[pyo3(
        name = "recommendForAllUsers",
        signature = (*args, **_kwargs),
        text_signature = "($self, numItems)"
    )]
    fn recommend_for_all_users(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        recommend(slf, "recommendForAllUsers", args)
    }

    /// `recommendForAllItems`, as PySpark's (see `recommend`).
    #[pyo3(
        name = "recommendForAllItems",
        signature = (*args, **_kwargs),
        text_signature = "($self, numUsers)"
    )]
    fn recommend_for_all_items(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        recommend(slf, "recommendForAllItems", args)
    }

    /// `recommendForUserSubset`, as PySpark's (see `recommend`).
    #[pyo3(
        name = "recommendForUserSubset",
        signature = (*args, **_kwargs),
        text_signature = "($self, dataset, numItems)"
    )]
    fn recommend_for_user_subset(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        recommend(slf, "recommendForUserSubset", args)
    }

    /// `recommendForItemSubset`, as PySpark's (see `recommend`).
    #[pyo3(
        name = "recommendForItemSubset",
        signature = (*args, **_kwargs),
        text_signature = "($self, dataset, numUsers)"
    )]
    fn recommend_for_item_subset(
        slf: &Bound<'_, Self>,
        args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        recommend(slf, "recommendForItemSubset", args)
    }
}

/// The recommendations of `method` of the model `slf`, as PySpark's ALSModel methods, which
/// `try_remote_attribute_relation` sends over Spark Connect: the server's `method` of the
/// positional arguments only, sent as PySpark's `serialize` sends them (see
/// `object::fetch_arg_of`). So keyword arguments are dropped, no arity is checked (a wrong one,
/// like a dataset that is not a DataFrame, fails on the server when the result is analyzed),
/// and a model of no server-side object raises PySpark's AssertionError.
fn recommend(
    slf: &Bound<'_, PyALSModel>,
    method: &str,
    args: &Bound<'_, PyTuple>,
) -> PyResult<PyDataFrame> {
    let target = slf.try_borrow()?.core.target()?;
    let args: Vec<Bound<'_, PyAny>> = args.iter().collect();
    crate::ml::object::attribute_relation_py(&target, method, &args)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyALS>()?;
    m.add_class::<PyALSModel>()?;
    Ok(())
}
