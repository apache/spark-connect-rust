// SPDX-License-Identifier: Apache-2.0
//! The macros of the native ML classes: the `#[pymethods]` of a class with PySpark's Params API,
//! the getter and the Param of each of its params, `copy`, and its persistence (`write`, `save`,
//! `read`, `load`), around the methods of the class itself.
//!
//! PyO3 takes one `#[pymethods]` block of a class (without its `multiple-pymethods`), so the macro
//! emits the whole block: the methods of the class, given as they are, and the generated ones.

/// The `#[pymethods]` of a native estimator, transformer or evaluator `$ty`, of the JVM class
/// `$jvm` of the operator type `$op` (`Estimator`, `Transformer` or `Evaluator`): its own methods
/// `$body`, the Params API of PySpark's `Params`, a getter and a Param per param of `$params`
/// (`"maxIter" => ("getMaxIter", get_max_iter, param_max_iter)`), `copy` (of
/// [`Duplicate`](super::Duplicate)), `__repr__` (the uid, as `Identifiable.__repr__`), and
/// `write`, `save`, `read` and `load` (as `RemoteMLWriter` and `RemoteMLReader`).
macro_rules! native_pymethods {
    (
        $ty:ty;
        jvm: $jvm:literal, op: $op:ident;
        params: [ $( $param:literal => ($getter_py:literal, $getter:ident, $attr:ident) ),* $(,)? ];
        $($body:tt)*
    ) => {
        #[pymethods]
        impl $ty {
            $($body)*

            /// The uid, as `Identifiable.uid`.
            #[getter]
            fn uid(&self) -> String {
                $crate::ml::params_api::NativeInstance::uid(self)
            }

            /// The Param objects, as `Params.params`.
            #[getter]
            fn params<'py>(slf: &Bound<'py, Self>) -> PyResult<Vec<Bound<'py, PyAny>>> {
                $crate::ml::params_api::params(slf.as_any())
            }

            /// `getParam(paramName)`, of PySpark's keyword.
            #[pyo3(name = "getParam")]
            #[allow(non_snake_case)]
            fn __params_get_param<'py>(
                slf: &Bound<'py, Self>,
                paramName: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                $crate::ml::params_api::get_param(slf.as_any(), paramName)
            }

            /// `hasParam(paramName)`, of PySpark's keyword.
            #[pyo3(name = "hasParam")]
            #[allow(non_snake_case)]
            fn __params_has_param(slf: &Bound<'_, Self>, paramName: &Bound<'_, PyAny>) -> PyResult<bool> {
                $crate::ml::params_api::has_param(slf.as_any(), paramName)
            }

            #[pyo3(name = "isSet")]
            fn __params_is_set(slf: &Bound<'_, Self>, param: &Bound<'_, PyAny>) -> PyResult<bool> {
                $crate::ml::params_api::is_set(slf, param)
            }

            #[pyo3(name = "isDefined")]
            fn __params_is_defined(slf: &Bound<'_, Self>, param: &Bound<'_, PyAny>) -> PyResult<bool> {
                $crate::ml::params_api::is_defined(slf, param)
            }

            #[pyo3(name = "hasDefault")]
            fn __params_has_default(slf: &Bound<'_, Self>, param: &Bound<'_, PyAny>) -> PyResult<bool> {
                $crate::ml::params_api::has_default(slf, param)
            }

            #[pyo3(name = "getOrDefault")]
            fn __params_get_or_default(
                slf: &Bound<'_, Self>,
                param: &Bound<'_, PyAny>,
            ) -> PyResult<Py<PyAny>> {
                $crate::ml::params_api::get_or_default(slf, param)
            }

            #[pyo3(name = "explainParam")]
            fn __params_explain_param(slf: &Bound<'_, Self>, param: &Bound<'_, PyAny>) -> PyResult<String> {
                $crate::ml::params_api::explain_param(slf, param)
            }

            #[pyo3(name = "explainParams")]
            fn __params_explain_params(slf: &Bound<'_, Self>) -> PyResult<String> {
                $crate::ml::params_api::explain_params(slf)
            }

            #[pyo3(name = "extractParamMap", signature = (extra=None))]
            fn __params_extract_param_map<'py>(
                slf: &Bound<'py, Self>,
                extra: Option<&Bound<'py, PyAny>>,
            ) -> PyResult<Bound<'py, PyDict>> {
                $crate::ml::params_api::extract_param_map(slf, extra)
            }

            #[pyo3(name = "set")]
            fn __params_set(
                slf: &Bound<'_, Self>,
                param: &Bound<'_, PyAny>,
                value: &Bound<'_, PyAny>,
            ) -> PyResult<()> {
                $crate::ml::params_api::set(slf, param, value)
            }

            /// As `JavaParams.clear` over Spark Connect, which does nothing (`try_remote_intercept`).
            #[pyo3(name = "clear")]
            fn __params_clear(&self, param: &Bound<'_, PyAny>) {
                let _ = param;
            }

            /// A copy of the same uid and params, with `extra`, as `Params.copy`.
            #[pyo3(name = "copy", signature = (extra=None))]
            fn __params_copy<'py>(
                slf: &Bound<'py, Self>,
                extra: Option<&Bound<'py, PyAny>>,
            ) -> PyResult<Bound<'py, Self>> {
                $crate::ml::params_api::copy(slf, extra)
            }

            /// The uid, as `Identifiable.__repr__`.
            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                slf.getattr("uid")?.extract()
            }

            /// The writer of the operator, as `JavaMLWritable.write` (`RemoteMLWriter`).
            fn write(slf: &Bound<'_, Self>) -> PyResult<$crate::ml::object::PyMLWriter> {
                $crate::ml::params_api::operator_writer(
                    slf,
                    $jvm,
                    spark_connect::ml::OperatorType::$op,
                )
            }

            /// Save to `path`, as `MLWritable.save`: `self.write().save(path)`.
            fn save(slf: &Bound<'_, Self>, path: &str) -> PyResult<()> {
                slf.call_method0("write")?.call_method1("save", (path,))?;
                Ok(())
            }

            /// The reader of the class, as `JavaMLReadable.read` (`RemoteMLReader`).
            #[classmethod]
            fn read(cls: &Bound<'_, pyo3::types::PyType>) -> $crate::ml::object::PyMLReader {
                let _ = cls;
                $crate::ml::object::PyMLReader::new(
                    $jvm,
                    spark_connect::ml::OperatorType::$op,
                    <$ty>::__from_read,
                )
            }

            /// Load from `path`, as `MLReadable.load`: `cls.read().load(path)`.
            #[classmethod]
            fn load(cls: &Bound<'_, pyo3::types::PyType>, path: &str) -> PyResult<Py<PyAny>> {
                Ok(cls.call_method0("read")?.call_method1("load", (path,))?.unbind())
            }

            $(
                #[pyo3(name = $getter_py)]
                fn $getter(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                    $crate::ml::params_api::getter(slf, $param)
                }

                #[getter]
                #[pyo3(name = $param)]
                fn $attr<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                    $crate::ml::params_api::param(slf.as_any(), $param)
                }
            )*
        }

        impl $ty {
            /// The instance of what a `Read` returns, as `RemoteMLReader.loadInstance`: the class's
            /// default instance, of the uid and the params that the server read.
            fn __from_read<'py>(
                py: Python<'py>,
                read: spark_connect::ml::ReadResult,
            ) -> PyResult<Py<PyAny>> {
                $crate::ml::params_api::from_read::<$ty>(py, read)
            }
        }
    };
}

/// A native model class `$ty` of PySpark's model class `$name` (of the module `$module`) and of the
/// JVM class `$jvm`, over a [`ModelCore`](super::object::ModelCore): the Params API, a getter and a
/// Param per param of `$params`, the setters of `$setters` (`"setFeaturesCol" => (set_features_col,
/// "featuresCol")`), the members of the server's model object (a property of `$properties`, a method
/// of `$methods` of its typed arguments, and a DataFrame of `$relations`, each as the method of the
/// object that PySpark's `_call_java` or attribute relation reads), `transform`, `copy`,
/// `__repr__` (the server's `toString`, as `JavaModel.__repr__`), `write`, `save`, `read` and
/// `load`, and the class's own methods `$body`.
macro_rules! native_model {
    (
        $ty:ident = $name:literal in $module:literal, jvm: $jvm:literal;
        params: [ $( $param:literal => ($getter_py:literal, $getter:ident, $attr:ident) ),* $(,)? ];
        setters: [ $( $setter_py:literal => ($setter:ident, $set_param:literal) ),* $(,)? ];
        properties: [ $( $prop_py:literal => ($prop:ident, $prop_method:literal) ),* $(,)? ];
        methods: [ $( $m_py:literal => ($m:ident, $m_method:literal, ( $( $arg:ident : $argty:ty $(= $default:tt)? ),* )) ),* $(,)? ];
        relations: [ $( $r_py:literal => ($r:ident, $r_method:literal) ),* $(,)? ];
        $($body:tt)*
    ) => {
        #[pyclass(name = $name, module = $module)]
        pub struct $ty {
            core: $crate::ml::object::ModelCore,
        }

        impl $crate::ml::params_api::ParamsHolder for $ty {
            fn set_value(&self, py: Python<'_>, name: &str) -> PyResult<Option<Py<PyAny>>> {
                $crate::ml::params_api::literal_value(py, self.core.model.params(), name)
            }

            fn put(
                &mut self,
                py: Python<'_>,
                name: &'static str,
                value: $crate::ml::Converted,
            ) -> PyResult<()> {
                let params = self.core.model.params().clone();
                *self.core.model.params_mut() =
                    $crate::ml::object::put_model_param(py, params, name, value)?;
                Ok(())
            }

            fn own_default(&self, py: Python<'_>, name: &str) -> Option<Py<PyAny>> {
                $crate::ml::params_api::own_default_of(py, &self.core.own_defaults, name)
            }

            fn put_own_default(&mut self, name: &'static str, value: Py<PyAny>) -> bool {
                $crate::ml::params_api::put_own_default_in(&mut self.core.own_defaults, name, value);
                true
            }
        }

        impl $crate::ml::params_api::NativeInstance for $ty {
            fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
                Py::new(py, Self { core: self.core.duplicate(py) })
            }

            fn uid(&self) -> String {
                self.core.uid()
            }

            fn set_uid(&mut self, uid: &str) {
                self.core.model.operator_mut().uid = uid.to_string();
            }
        }

        impl $ty {
            /// The model of a fit, of the training summary of the summary dataset `summary` (see
            /// `ModelCore::fitted`).
            #[allow(dead_code)]
            pub(crate) fn fitted(
                py: Python<'_>,
                model: Box<dyn Model>,
                df: &DataFrame,
                summary: $crate::ml::object::SummaryDataset,
            ) -> PyResult<Self> {
                Ok(Self { core: $crate::ml::object::ModelCore::fitted(py, model, df, summary)? })
            }

            fn __from_read<'py>(
                py: Python<'py>,
                read: spark_connect::ml::ReadResult,
            ) -> PyResult<Py<PyAny>> {
                let core = $crate::ml::object::ModelCore::loaded(py, $jvm, $name, read)?;
                Ok(Py::new(py, Self { core })?.into_any())
            }
        }

        #[pymethods]
        impl $ty {
            $($body)*

            #[getter]
            fn uid(&self) -> String {
                self.core.uid()
            }

            /// The transform of `dataset`, as `Transformer.transform`: of a copy of the model of a
            /// non-empty param map `params`.
            #[pyo3(signature = (dataset, params=None))]
            fn transform<'py>(
                slf: &Bound<'py, Self>,
                dataset: &Bound<'py, PyAny>,
                params: Option<&Bound<'py, PyAny>>,
            ) -> PyResult<Py<PyAny>> {
                if let Some(map) = params.filter(|p| p.cast::<PyDict>().is_ok_and(|d| !d.is_empty())) {
                    let copy = $crate::ml::params_api::copy(slf, Some(map))?;
                    return Ok(copy.call_method1("transform", (dataset,))?.unbind());
                }
                $crate::ml::check_params(params)?;
                let input = $crate::ml::dataset_of(dataset, "transform")?;
                let df = slf.try_borrow()?.core.transform(slf.py(), &input)?;
                Ok(Py::new(slf.py(), PyDataFrame::new(df))?.into_any())
            }

            #[getter]
            fn params<'py>(slf: &Bound<'py, Self>) -> PyResult<Vec<Bound<'py, PyAny>>> {
                $crate::ml::params_api::params(slf.as_any())
            }

            #[pyo3(name = "getParam")]
            #[allow(non_snake_case)]
            fn __params_get_param<'py>(slf: &Bound<'py, Self>, paramName: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                $crate::ml::params_api::get_param(slf.as_any(), paramName)
            }

            #[pyo3(name = "hasParam")]
            #[allow(non_snake_case)]
            fn __params_has_param(slf: &Bound<'_, Self>, paramName: &Bound<'_, PyAny>) -> PyResult<bool> {
                $crate::ml::params_api::has_param(slf.as_any(), paramName)
            }

            #[pyo3(name = "isSet")]
            fn __params_is_set(slf: &Bound<'_, Self>, param: &Bound<'_, PyAny>) -> PyResult<bool> {
                $crate::ml::params_api::is_set(slf, param)
            }

            #[pyo3(name = "isDefined")]
            fn __params_is_defined(slf: &Bound<'_, Self>, param: &Bound<'_, PyAny>) -> PyResult<bool> {
                $crate::ml::params_api::is_defined(slf, param)
            }

            #[pyo3(name = "hasDefault")]
            fn __params_has_default(slf: &Bound<'_, Self>, param: &Bound<'_, PyAny>) -> PyResult<bool> {
                $crate::ml::params_api::has_default(slf, param)
            }

            #[pyo3(name = "getOrDefault")]
            fn __params_get_or_default(slf: &Bound<'_, Self>, param: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                $crate::ml::params_api::get_or_default(slf, param)
            }

            #[pyo3(name = "explainParam")]
            fn __params_explain_param(slf: &Bound<'_, Self>, param: &Bound<'_, PyAny>) -> PyResult<String> {
                $crate::ml::params_api::explain_param(slf, param)
            }

            #[pyo3(name = "explainParams")]
            fn __params_explain_params(slf: &Bound<'_, Self>) -> PyResult<String> {
                $crate::ml::params_api::explain_params(slf)
            }

            #[pyo3(name = "extractParamMap", signature = (extra=None))]
            fn __params_extract_param_map<'py>(slf: &Bound<'py, Self>, extra: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyDict>> {
                $crate::ml::params_api::extract_param_map(slf, extra)
            }

            #[pyo3(name = "set")]
            fn __params_set(slf: &Bound<'_, Self>, param: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
                $crate::ml::params_api::set(slf, param, value)
            }

            /// As `JavaParams.clear` over Spark Connect, which does nothing.
            #[pyo3(name = "clear")]
            fn __params_clear(&self, param: &Bound<'_, PyAny>) {
                let _ = param;
            }

            #[pyo3(name = "copy", signature = (extra=None))]
            fn __params_copy<'py>(slf: &Bound<'py, Self>, extra: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, Self>> {
                $crate::ml::params_api::copy(slf, extra)
            }

            /// The server's description of the model, as `JavaModel.__repr__` (its `toString`).
            fn __repr__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                let target = slf.try_borrow()?.core.target()?;
                let out = $crate::ml::object::fetch_py(slf.py(), &target, "toString", &[])?;
                $crate::ml::object::fetch_output_to_py(slf.py(), out)
            }

            fn write(slf: &Bound<'_, Self>) -> PyResult<$crate::ml::object::PyMLWriter> {
                slf.try_borrow()?.core.writer(slf.py())
            }

            fn save(slf: &Bound<'_, Self>, path: &str) -> PyResult<()> {
                slf.call_method0("write")?.call_method1("save", (path,))?;
                Ok(())
            }

            #[classmethod]
            fn read(cls: &Bound<'_, pyo3::types::PyType>) -> $crate::ml::object::PyMLReader {
                let _ = cls;
                $crate::ml::object::PyMLReader::new($jvm, spark_connect::ml::OperatorType::Model, <$ty>::__from_read)
            }

            #[classmethod]
            fn load(cls: &Bound<'_, pyo3::types::PyType>, path: &str) -> PyResult<Py<PyAny>> {
                Ok(cls.call_method0("read")?.call_method1("load", (path,))?.unbind())
            }

            $(
                #[pyo3(name = $getter_py)]
                fn $getter(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                    $crate::ml::params_api::getter(slf, $param)
                }

                #[getter]
                #[pyo3(name = $param)]
                fn $attr<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                    $crate::ml::params_api::param(slf.as_any(), $param)
                }
            )*

            $(
                #[pyo3(name = $setter_py)]
                fn $setter<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
                    $crate::ml::params_api::set_by_name(&slf, $set_param, value)?;
                    Ok(slf)
                }
            )*

            $(
                #[getter]
                #[pyo3(name = $prop_py)]
                fn $prop(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                    let target = slf.try_borrow()?.core.target()?;
                    let out = $crate::ml::object::fetch_py(slf.py(), &target, $prop_method, &[])?;
                    $crate::ml::object::fetch_output_to_py(slf.py(), out)
                }
            )*

            $(
                #[pyo3(name = $m_py, signature = ( $( $arg $(= $default)? ),* ))]
                fn $m<'py>(slf: &Bound<'py, Self>, $( $arg: $argty ),*) -> PyResult<Py<PyAny>> {
                    use pyo3::IntoPyObjectExt;
                    let py = slf.py();
                    let target = slf.try_borrow()?.core.target()?;
                    let args = vec![$( $arg.into_bound_py_any(py)? ),*];
                    let out = $crate::ml::object::fetch_py(py, &target, $m_method, &args)?;
                    $crate::ml::object::fetch_output_to_py(py, out)
                }
            )*

            $(
                #[getter]
                #[pyo3(name = $r_py)]
                fn $r(slf: &Bound<'_, Self>) -> PyResult<PyDataFrame> {
                    let target = slf.try_borrow()?.core.target()?;
                    $crate::ml::object::attribute_relation_py(&target, $r_method, &[])
                }
            )*
        }
    };
}

/// A summary class `$ty` of PySpark's summary class `$name` (of the module `$module`), over an
/// [`MlTarget`](spark_connect::ml::MlTarget) of the summary's server-side object: a property of
/// `$properties`, a method of `$methods` of its typed arguments, and a DataFrame of `$relations`
/// (of the summary dataset of a training summary), each as the method of the object that PySpark's
/// `_call_java` or attribute relation reads.
macro_rules! native_summary {
    (
        $ty:ident = $name:literal in $module:literal;
        properties: [ $( $prop_py:literal => ($prop:ident, $prop_method:literal) ),* $(,)? ];
        methods: [ $( $m_py:literal => ($m:ident, $m_method:literal, ( $( $arg:ident : $argty:ty $(= $default:tt)? ),* )) ),* $(,)? ];
        relations: [ $( $r_py:literal => ($r:ident, $r_method:literal) ),* $(,)? ];
    ) => {
        #[pyclass(name = $name, module = $module)]
        pub struct $ty {
            target: spark_connect::ml::MlTarget,
        }

        impl From<spark_connect::ml::MlTarget> for $ty {
            fn from(target: spark_connect::ml::MlTarget) -> Self {
                Self { target }
            }
        }

        #[pymethods]
        impl $ty {
            $(
                #[getter]
                #[pyo3(name = $prop_py)]
                fn $prop(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                    let out = $crate::ml::object::fetch_py(py, &self.target, $prop_method, &[])?;
                    $crate::ml::object::fetch_output_to_py(py, out)
                }
            )*

            $(
                #[pyo3(name = $m_py, signature = ( $( $arg $(= $default)? ),* ))]
                fn $m(&self, py: Python<'_>, $( $arg: $argty ),*) -> PyResult<Py<PyAny>> {
                    use pyo3::IntoPyObjectExt;
                    let args = vec![$( $arg.into_bound_py_any(py)? ),*];
                    let out = $crate::ml::object::fetch_py(py, &self.target, $m_method, &args)?;
                    $crate::ml::object::fetch_output_to_py(py, out)
                }
            )*

            $(
                #[getter]
                #[pyo3(name = $r_py)]
                fn $r(&self) -> PyResult<PyDataFrame> {
                    $crate::ml::object::attribute_relation_py(&self.target, $r_method, &[])
                }
            )*
        }
    };
}
