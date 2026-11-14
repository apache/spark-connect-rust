// SPDX-License-Identifier: Apache-2.0
//! Native ALS fitting, prediction, factors and recommendations.
use super::*;
use pyo3::IntoPyObjectExt;
use spark_connect::ml::ALS;
use spark_connect_proto as proto;

/// The literal of a recommendation count, as PySpark's serialization of the argument,
/// `LiteralExpression._from_value`, makes it: a Column is not a value it takes, not even a Column
/// of a literal, so it raises the `UNSUPPORTED_DATA_TYPE` of `Column`, as another value that is
/// not a literal does.
fn count_literal(value: &Bound<'_, PyAny>) -> PyResult<proto::expression::Literal> {
    let column = match value.is_instance_of::<crate::column::PyColumn>() {
        true => None,
        false => Some(crate::functions::to_value_column(value)?),
    };
    match column.as_ref().map(|column| column.expression().node()) {
        Some(spark_connect::expression::ExpressionNode::Literal(literal)) => {
            Ok(literal.to_literal_proto())
        }
        _ => Err(crate::column::raise_pyspark(
            value.py(),
            "PySparkTypeError",
            "UNSUPPORTED_DATA_TYPE",
            &[("data_type", &value.get_type().name()?.to_string())],
        )),
    }
}

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
#[pyclass(name = "ALS", module = "pyspark.ml.connect.recommendation")]
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
    ) -> PyResult<Fitted<PyALSModel>> {
        fit_with(slf, dataset, params, || {
            let py = slf.py();
            let df = dataset_of(dataset, "fit")?;
            let mut estimator = snapshot(slf)?;
            // As `fit_native_model`, which the core model's rank does not fit: the fitted core
            // model is kept concrete here, for its rank.
            let defaults = params_api::fit_defaults(slf)?;
            let session = df.spark_session();
            let fitted = interruptible(py, &session, || estimator.fit(&df).map_err(Box::new))?;
            let rank = fitted.rank();
            let mut model = PyALSModel::fitted(py, Box::new(fitted), &df, object::SummaryDataset::None)?;
            // The rank of the fit, kept on the model for its `rank` (see `PyALSModel::rank`).
            model.core.cached(py, "rank", || rank.into_py_any(py))?;
            params_api::copy_defaults(py, defaults, &mut model)?;
            Ok(model)
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

impl params_api::NativeInstance for PyALS {
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
    properties: [];
    methods: [];
    relations: [
        "itemFactors" => (item_factors, "itemFactors"),
        "userFactors" => (user_factors, "userFactors"),
    ];
    relation_methods: [];
    summary: None;
    cached: [];

    /// The rank of the factors, as `ALSModel.rank`: of the fit, which the fit keeps in the cache of
    /// the model (no request), or, of a loaded model, of the server's `rank`, read once.
    #[getter]
    fn rank(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        let this = slf.try_borrow()?;
        this.core.cached(py, "rank", || {
            let target = this.core.target()?;
            let out = object::fetch_py(py, &target, "rank", &[])?;
            object::fetch_output_to_py(py, out)
        })
    }

    /// The top `numItems` items of every user, as `ALSModel.recommendForAllUsers`. The count
    /// is sent as PySpark's serialization sends it, a literal (see [`count_literal`]).
    #[allow(non_snake_case)]
    #[pyo3(name = "recommendForAllUsers", signature = (numItems))]
    fn recommend_for_all_users(
        slf: &Bound<'_, Self>,
        numItems: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        count_literal(numItems)?;
        let target = slf.try_borrow()?.core.target()?;
        let count = numItems.clone();
        object::attribute_relation_py(&target, "recommendForAllUsers", &[count])
    }

    /// The top `numUsers` users of every item, as `ALSModel.recommendForAllItems`.
    #[allow(non_snake_case)]
    #[pyo3(name = "recommendForAllItems", signature = (numUsers))]
    fn recommend_for_all_items(
        slf: &Bound<'_, Self>,
        numUsers: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        count_literal(numUsers)?;
        let target = slf.try_borrow()?.core.target()?;
        let count = numUsers.clone();
        object::attribute_relation_py(&target, "recommendForAllItems", &[count])
    }

    /// The top `numItems` items of each user of `dataset`, as
    /// `ALSModel.recommendForUserSubset`, of the dataset as PySpark sends it.
    #[allow(non_snake_case)]
    #[pyo3(name = "recommendForUserSubset", signature = (dataset, numItems))]
    fn recommend_for_user_subset(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        numItems: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        helper_dataset_of(dataset, "recommendForUserSubset")?;
        count_literal(numItems)?;
        let target = slf.try_borrow()?.core.target()?;
        object::attribute_relation_py(
            &target,
            "recommendForUserSubset",
            &[dataset.clone(), numItems.clone()],
        )
    }

    /// The top `numUsers` users of each item of `dataset`, as
    /// `ALSModel.recommendForItemSubset`, of the dataset as PySpark sends it.
    #[allow(non_snake_case)]
    #[pyo3(name = "recommendForItemSubset", signature = (dataset, numUsers))]
    fn recommend_for_item_subset(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        numUsers: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        helper_dataset_of(dataset, "recommendForItemSubset")?;
        count_literal(numUsers)?;
        let target = slf.try_borrow()?.core.target()?;
        object::attribute_relation_py(
            &target,
            "recommendForItemSubset",
            &[dataset.clone(), numUsers.clone()],
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyALS>()?;
    m.add_class::<PyALSModel>()?;
    Ok(())
}
