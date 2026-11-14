// SPDX-License-Identifier: Apache-2.0
//! Native ALS fitting, prediction, factors and recommendations.
use super::*;
use spark_connect::ml::{ALSModel, ALS};
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
#[pymethods]
impl PyALS {
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
    ) -> PyResult<PyALSModel> {
        check_fit_params(params)?;
        let input = dataset_of(dataset, "fit")?;
        let mut estimator = snapshot(slf)?;
        let session = input.spark_session();
        let inner = interruptible(slf.py(), &session, || {
            estimator.fit(&input).map_err(Box::new)
        })?;
        Ok(PyALSModel {
            inner,
            unsendable: Vec::new(),
        })
    }
}

#[pyclass(name = "ALSModel", module = "pyspark.ml.connect.recommendation")]
pub struct PyALSModel {
    inner: ALSModel,
    unsendable: Vec<&'static str>,
}
impl Native for PyALSModel {
    type Core = ALSModel;
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
impl PyALSModel {
    #[pyo3(name = "setBlockSize")]
    fn set_block_size<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "blockSize", Kind::Int, value)
    }
    #[pyo3(name = "setColdStartStrategy")]
    fn set_cold_start_strategy<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "coldStartStrategy", Kind::Str, value)
    }
    #[pyo3(name = "setItemCol")]
    fn set_item_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "itemCol", Kind::Str, value)
    }
    #[pyo3(name = "setPredictionCol")]
    fn set_prediction_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "predictionCol", Kind::Str, value)
    }
    #[pyo3(name = "setUserCol")]
    fn set_user_col<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        set_param(slf, "userCol", Kind::Str, value)
    }
    #[pyo3(signature = (dataset, params=None))]
    fn transform(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        check_params(params)?;
        let input = dataset_of(dataset, "transform")?;
        let mut model = snapshot(slf)?;
        Ok(PyDataFrame::new(model.transform(&input).to_pyerr()?))
    }
    #[getter]
    fn rank(&self) -> i32 {
        self.inner.rank()
    }
    #[getter]
    #[pyo3(name = "userFactors")]
    fn user_factors(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.inner.user_factors().to_pyerr()?))
    }
    #[getter]
    #[pyo3(name = "itemFactors")]
    fn item_factors(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.inner.item_factors().to_pyerr()?))
    }
    #[allow(non_snake_case)]
    #[pyo3(name="recommendForAllUsers",signature=(numItems))]
    fn recommend_for_all_users(
        slf: &Bound<'_, Self>,
        numItems: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let count = count_literal(numItems)?;
        let model = slf.try_borrow()?.inner.clone();
        Ok(PyDataFrame::new(
            model.recommend_with_literal(true, None, count).to_pyerr()?,
        ))
    }
    #[allow(non_snake_case)]
    #[pyo3(name="recommendForAllItems",signature=(numUsers))]
    fn recommend_for_all_items(
        slf: &Bound<'_, Self>,
        numUsers: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let count = count_literal(numUsers)?;
        let model = slf.try_borrow()?.inner.clone();
        Ok(PyDataFrame::new(
            model
                .recommend_with_literal(false, None, count)
                .to_pyerr()?,
        ))
    }
    #[allow(non_snake_case)]
    #[pyo3(name="recommendForUserSubset",signature=(dataset, numItems))]
    fn recommend_for_user_subset(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        numItems: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let input = helper_dataset_of(dataset, "recommendForUserSubset")?;
        let count = count_literal(numItems)?;
        let model = slf.try_borrow()?.inner.clone();
        Ok(PyDataFrame::new(
            model
                .recommend_with_literal(true, Some(&input), count)
                .to_pyerr()?,
        ))
    }
    #[allow(non_snake_case)]
    #[pyo3(name="recommendForItemSubset",signature=(dataset, numUsers))]
    fn recommend_for_item_subset(
        slf: &Bound<'_, Self>,
        dataset: &Bound<'_, PyAny>,
        numUsers: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        let input = helper_dataset_of(dataset, "recommendForItemSubset")?;
        let count = count_literal(numUsers)?;
        let model = slf.try_borrow()?.inner.clone();
        Ok(PyDataFrame::new(
            model
                .recommend_with_literal(false, Some(&input), count)
                .to_pyerr()?,
        ))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyALS>()?;
    m.add_class::<PyALSModel>()?;
    Ok(())
}
