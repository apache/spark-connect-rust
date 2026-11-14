// SPDX-License-Identifier: Apache-2.0
//! The macros of the native ML classes: the `#[pymethods]` of a class with PySpark's Params API,
//! the getter and the Param of each of its params, `copy`, and its persistence (`write`, `save`,
//! `read`, `load`), around the methods of the class itself.
//!
//! PyO3 takes one `#[pymethods]` block of a class (without its `multiple-pymethods`), so each
//! macro emits the whole block, through [`params_api_block!`]: the methods of the class, given as
//! they are, and the generated ones.
//!
//! The arguments of the methods and DataFrames of a model or a summary are the Python objects as
//! given, which a fetch sends as PySpark's `_call_java` does. Each is written with its default of
//! PySpark, in Python (`(value)`, `(beta = 1.0)`, `(residualsType = "deviance")`), and a method of
//! a default gives its text signature too (`"($self, beta=1.0)"`), which PyO3 cannot render of a
//! default that it evaluates.

/// The `#[pymethods]` block of the class `$ty`: the methods `$methods`, the Params API of PySpark's
/// `Params` (`params`, `getParam`, `hasParam`, `isSet`, `isDefined`, `hasDefault`, `getOrDefault`,
/// `explainParam(s)`, `extractParamMap`, `set` and `clear`), and a getter and a Param per param of
/// `$params` (`"maxIter" => ("getMaxIter", get_max_iter, param_max_iter)`).
macro_rules! params_api_block {
    (
        $ty:ty;
        params: [
            $( $param:literal => ($getter_py:literal, $getter:ident, $attr:ident) ),* $(,)?
        ];
        params_only: [ $( $po:literal => $po_attr:ident ),* $(,)? ];
        { $($methods:tt)* }
    ) => {
        #[pymethods]
        impl $ty {
            $($methods)*

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
            fn __params_has_param(
                slf: &Bound<'_, Self>,
                paramName: &Bound<'_, PyAny>,
            ) -> PyResult<bool> {
                $crate::ml::params_api::has_param(slf.as_any(), paramName)
            }

            #[pyo3(name = "isSet")]
            fn __params_is_set(slf: &Bound<'_, Self>, param: &Bound<'_, PyAny>) -> PyResult<bool> {
                $crate::ml::params_api::is_set(slf, param)
            }

            #[pyo3(name = "isDefined")]
            fn __params_is_defined(
                slf: &Bound<'_, Self>,
                param: &Bound<'_, PyAny>,
            ) -> PyResult<bool> {
                $crate::ml::params_api::is_defined(slf, param)
            }

            #[pyo3(name = "hasDefault")]
            fn __params_has_default(
                slf: &Bound<'_, Self>,
                param: &Bound<'_, PyAny>,
            ) -> PyResult<bool> {
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
            fn __params_explain_param(
                slf: &Bound<'_, Self>,
                param: &Bound<'_, PyAny>,
            ) -> PyResult<String> {
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

            /// As `JavaParams.clear` over Spark Connect, which does nothing
            /// (`try_remote_intercept`).
            #[pyo3(name = "clear")]
            fn __params_clear(&self, param: &Bound<'_, PyAny>) {
                let _ = param;
            }

            $(
                /// The Param attribute of a param of no getter of the class (as `numTrees` of
                /// a tree ensemble model, whose `getNumTrees` is a property).
                #[classattr]
                #[pyo3(name = $po)]
                fn $po_attr() -> $crate::ml::params_api::PyParamAttribute {
                    $crate::ml::params_api::PyParamAttribute::new($po)
                }
            )*

            $(
                #[pyo3(name = $getter_py)]
                fn $getter(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                    $crate::ml::params_api::getter(slf, $param)
                }

                /// The Param attribute, of the class's Param and of the instance's (see
                /// `PyParamAttribute`).
                #[classattr]
                #[pyo3(name = $param)]
                fn $attr() -> $crate::ml::params_api::PyParamAttribute {
                    $crate::ml::params_api::PyParamAttribute::new($param)
                }
            )*
        }
    };
}

/// The `#[pymethods]` of a native estimator, transformer or evaluator `$ty`: its own methods
/// `$body`, the Params API and the params of [`params_api_block!`], the uid, and `__repr__` (the
/// uid, as `Identifiable.__repr__`).
///
/// Of the JVM class `$jvm` of the operator type `$op` (`Estimator`, `Transformer` or `Evaluator`),
/// it also has `copy` (of `NativeInstance::duplicate`), and `write`, `save`, `read` and `load`, as
/// `RemoteMLWriter` and `RemoteMLReader`. Of `persistence: python`, the class has its own `copy`
/// and persistence, in PySpark's Python format (a Pipeline or a validator).
macro_rules! native_pymethods {
    (
        $ty:ty;
        jvm: $jvm:literal, op: $op:ident;
        params: [
            $( $param:literal => ($getter_py:literal, $getter:ident, $attr:ident) ),* $(,)?
        ];
        $( params_only: [ $( $po:literal => $po_attr:ident ),* $(,)? ]; )?
        $($body:tt)*
    ) => {
        params_api_block! {
            $ty;
            params: [ $( $param => ($getter_py, $getter, $attr) ),* ];
            params_only: [ $( $( $po => $po_attr ),* )? ];
            {
                $($body)*

                /// The uid, as `Identifiable.uid`.
                #[getter]
                fn uid(&self) -> String {
                    $crate::ml::params_api::NativeInstance::uid(self)
                }

                /// The uid, as `Identifiable.__repr__`.
                fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                    slf.getattr("uid")?.extract()
                }

                /// A copy of the same uid and params, with `extra`, as `Params.copy`.
                #[pyo3(name = "copy", signature = (extra=None))]
                fn __params_copy<'py>(
                    slf: &Bound<'py, Self>,
                    extra: Option<&Bound<'py, PyAny>>,
                ) -> PyResult<Bound<'py, Self>> {
                    $crate::ml::params_api::copy(slf, extra)
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
            }
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
    (
        $ty:ty;
        persistence: python;
        params: [
            $( $param:literal => ($getter_py:literal, $getter:ident, $attr:ident) ),* $(,)?
        ];
        $( params_only: [ $( $po:literal => $po_attr:ident ),* $(,)? ]; )?
        $($body:tt)*
    ) => {
        params_api_block! {
            $ty;
            params: [ $( $param => ($getter_py, $getter, $attr) ),* ];
            params_only: [ $( $( $po => $po_attr ),* )? ];
            {
                $($body)*

                /// The uid, as `Identifiable.uid`.
                #[getter]
                fn uid(&self) -> String {
                    $crate::ml::params_api::NativeInstance::uid(self)
                }

                /// The uid, as `Identifiable.__repr__`.
                fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                    slf.getattr("uid")?.extract()
                }
            }
        }
    };
}

/// A native model class `$ty` of PySpark's model class `$name` (of the module `$module`) and of
/// the JVM class `$jvm`, over a [`ModelCore`](super::object::ModelCore), with:
/// - the Params API and the params of [`params_api_block!`], and the setters of `$setters`
///   (`"setFeaturesCol" => (set_features_col, "featuresCol")`);
/// - the members of the server's model object, each as the method of the object that PySpark's
///   `_call_java` or attribute relation reads: a property of `$properties`, a method of
///   `$methods`, a DataFrame property of `$relations` and a DataFrame method of
///   `$relation_methods` (see the module doc of their arguments);
/// - its training summary object, which `$summary` (`Some(make)`, of a
///   [`SummaryMaker`](super::object::SummaryMaker), or `None`) makes at the fit, as
///   `try_remote_fit` makes `_summary`, and which the class's own `summary` returns by
///   `ModelCore::summary_object`;
/// - the `functools.cached_property` members of `$cached` (`"trees" => (trees, make)`, of a
///   [`CachedMaker`](super::object::CachedMaker)), which PySpark's `params` walk of `_resetUid`
///   reads at a fit (after the summary) and at a load, and which a copy shares, and the methods of
///   `functools.cache` of the optional `cached_methods` (`"toLocal" => (to_local, make)`), which a
///   copy does not share;
/// - `transform`, `copy`, `__repr__` (the server's `toString`, as `JavaModel.__repr__`), `write`,
///   `save`, `read` and `load`, `fitted` (of a fit), `of_object` (of an object that a fetch
///   returns), and the class's own methods `$body`.
macro_rules! native_model {
    (
        $ty:ident = $name:literal in $module:literal, jvm: $jvm:literal;
        params: [
            $( $param:literal => ($getter_py:literal, $getter:ident, $attr:ident) ),* $(,)?
        ];
        $( params_only: [ $( $po:literal => $po_attr:ident ),* $(,)? ]; )?
        setters: [ $( $setter_py:literal => ($setter:ident, $set_param:literal) ),* $(,)? ];
        properties: [ $( $prop_py:literal => ($prop:ident, $prop_method:literal) ),* $(,)? ];
        methods: [
            $(
                $m_py:literal => (
                    $m:ident, $m_method:literal, ( $( $arg:ident $(= $default:tt)? ),* )
                    $(, $m_sig:literal)?
                )
            ),* $(,)?
        ];
        relations: [ $( $r_py:literal => ($r:ident, $r_method:literal) ),* $(,)? ];
        relation_methods: [
            $(
                $rm_py:literal => (
                    $rm:ident, $rm_method:literal, ( $( $rarg:ident $(= $rdefault:tt)? ),* )
                    $(, $rm_sig:literal)?
                )
            ),* $(,)?
        ];
        summary: $summary:expr;
        cached: [ $( $c_py:literal => ($c:ident, $c_make:expr) ),* $(,)? ];
        $( cached_methods: [ $( $cm_py:literal => ($cm:ident, $cm_make:expr) ),* $(,)? ]; )?
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
                let defaults = &mut self.core.own_defaults;
                $crate::ml::params_api::put_own_default_in(defaults, name, value);
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
            /// `ModelCore::fitted`), with the summary object and the cached members that PySpark
            /// makes at the fit, in its order.
            #[allow(dead_code)]
            pub(crate) fn fitted(
                py: Python<'_>,
                model: Box<dyn Model>,
                df: &DataFrame,
                summary: $crate::ml::object::SummaryDataset,
            ) -> PyResult<Self> {
                let core = $crate::ml::object::ModelCore::fitted(py, model, df, summary, $summary)?;
                let this = Self { core };
                this.__prime(py)?;
                Ok(this)
            }

            /// The model of the server-side object `object` that a fetch returns (a tree of an
            /// ensemble, a model of a OneVsRestModel), as PySpark makes one of a bare object: of a
            /// new uid, without set params, a training summary or cached members.
            #[allow(dead_code)]
            pub(crate) fn of_object(
                py: Python<'_>,
                object: spark_connect::ml::MlRef,
            ) -> PyResult<Self> {
                let core = $crate::ml::object::ModelCore::of_object(py, $jvm, $name, object)?;
                Ok(Self { core })
            }

            /// Read the cached members, as the `params` walk of PySpark's `_resetUid` does.
            #[allow(clippy::unused_unit, unused_variables)]
            fn __prime(&self, py: Python<'_>) -> PyResult<()> {
                $( self.core.cached(py, $c_py, || ($c_make)(py, &self.core))?; )*
                Ok(())
            }

            fn __from_read<'py>(
                py: Python<'py>,
                read: spark_connect::ml::ReadResult,
            ) -> PyResult<Py<PyAny>> {
                let core = $crate::ml::object::ModelCore::loaded(py, $jvm, $name, read)?;
                let this = Self { core };
                this.__prime(py)?;
                Ok(Py::new(py, this)?.into_any())
            }
        }

        params_api_block! {
            $ty;
            params: [ $( $param => ($getter_py, $getter, $attr) ),* ];
            params_only: [ $( $( $po => $po_attr ),* )? ];
            {
                $($body)*

                #[getter]
                fn uid(&self) -> String {
                    self.core.uid()
                }

                /// The transform of `dataset`, as `Transformer.transform`: of a copy of the model
                /// of a non-empty param map `params`.
                #[pyo3(signature = (dataset, params=None))]
                fn transform<'py>(
                    slf: &Bound<'py, Self>,
                    dataset: &Bound<'py, PyAny>,
                    params: Option<&Bound<'py, PyAny>>,
                ) -> PyResult<Py<PyAny>> {
                    let map = params.filter(|p| p.cast::<PyDict>().is_ok_and(|d| !d.is_empty()));
                    if let Some(map) = map {
                        let copy = $crate::ml::params_api::copy(slf, Some(map))?;
                        return Ok(copy.call_method1("transform", (dataset,))?.unbind());
                    }
                    $crate::ml::check_params(params)?;
                    let input = $crate::ml::dataset_of(dataset, "transform")?;
                    let df = slf.try_borrow()?.core.transform(slf.py(), &input)?;
                    Ok(Py::new(slf.py(), PyDataFrame::new(df))?.into_any())
                }

                #[pyo3(name = "copy", signature = (extra=None))]
                fn __params_copy<'py>(
                    slf: &Bound<'py, Self>,
                    extra: Option<&Bound<'py, PyAny>>,
                ) -> PyResult<Bound<'py, Self>> {
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
                    $crate::ml::object::PyMLReader::new(
                        $jvm,
                        spark_connect::ml::OperatorType::Model,
                        <$ty>::__from_read,
                    )
                }

                #[classmethod]
                fn load(cls: &Bound<'_, pyo3::types::PyType>, path: &str) -> PyResult<Py<PyAny>> {
                    Ok(cls.call_method0("read")?.call_method1("load", (path,))?.unbind())
                }

                $(
                    #[pyo3(name = $setter_py)]
                    fn $setter<'py>(
                        slf: Bound<'py, Self>,
                        value: &Bound<'py, PyAny>,
                    ) -> PyResult<Bound<'py, Self>> {
                        $crate::ml::params_api::set_by_name(&slf, $set_param, value)?;
                        Ok(slf)
                    }
                )*

                $(
                    #[getter]
                    #[pyo3(name = $prop_py)]
                    fn $prop(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                        let py = slf.py();
                        let target = slf.try_borrow()?.core.target()?;
                        let out = $crate::ml::object::fetch_py(py, &target, $prop_method, &[])?;
                        $crate::ml::object::fetch_output_to_py(py, out)
                    }
                )*

                $(
                    #[pyo3(
                        name = $m_py,
                        signature = (
                            $( $arg $(= $crate::ml::object::py_default(stringify!($default)))? ),*
                        )
                        $(, text_signature = $m_sig)?
                    )]
                    fn $m(slf: &Bound<'_, Self>, $( $arg: Py<PyAny> ),*) -> PyResult<Py<PyAny>> {
                        let py = slf.py();
                        let target = slf.try_borrow()?.core.target()?;
                        let args = vec![$( $arg.into_bound(py) ),*];
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

                $(
                    #[pyo3(
                        name = $rm_py,
                        signature = (
                            $( $rarg $(= $crate::ml::object::py_default(stringify!($rdefault)))? ),*
                        )
                        $(, text_signature = $rm_sig)?
                    )]
                    fn $rm(
                        slf: &Bound<'_, Self>,
                        $( $rarg: Py<PyAny> ),*
                    ) -> PyResult<PyDataFrame> {
                        let py = slf.py();
                        let target = slf.try_borrow()?.core.target()?;
                        let args = vec![$( $rarg.into_bound(py) ),*];
                        $crate::ml::object::attribute_relation_py(&target, $rm_method, &args)
                    }
                )*

                $(
                    #[getter]
                    #[pyo3(name = $c_py)]
                    fn $c(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                        let py = slf.py();
                        let this = slf.try_borrow()?;
                        this.core.cached(py, $c_py, || ($c_make)(py, &this.core))
                    }
                )*

                $( $(
                    /// A method of no arguments of `functools.cache` (`toLocal`): its value, made
                    /// once per object, which a copy does not share, as the cache of
                    /// `functools.cache` is of each object.
                    #[pyo3(name = $cm_py)]
                    fn $cm(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                        let py = slf.py();
                        let this = slf.try_borrow()?;
                        this.core.cached_method(py, $cm_py, || ($cm_make)(py, &this.core))
                    }
                )* )?
            }
        }
    };
}

/// A summary class `$ty` of PySpark's summary class `$name` (of the module `$module`), over an
/// [`MlTarget`](spark_connect::ml::MlTarget) of the summary's server-side object: a property of
/// `$properties`, a method of `$methods`, a DataFrame property of `$relations` and a DataFrame
/// method of `$relation_methods` (of the summary dataset of a training summary; see the module doc
/// of their arguments), each as the method of the object that PySpark's `_call_java` or attribute
/// relation reads, and the class's own methods `$body`.
macro_rules! native_summary {
    (
        $ty:ident = $name:literal in $module:literal;
        properties: [ $( $prop_py:literal => ($prop:ident, $prop_method:literal) ),* $(,)? ];
        methods: [
            $(
                $m_py:literal => (
                    $m:ident, $m_method:literal, ( $( $arg:ident $(= $default:tt)? ),* )
                    $(, $m_sig:literal)?
                )
            ),* $(,)?
        ];
        relations: [ $( $r_py:literal => ($r:ident, $r_method:literal) ),* $(,)? ];
        relation_methods: [
            $(
                $rm_py:literal => (
                    $rm:ident, $rm_method:literal, ( $( $rarg:ident $(= $rdefault:tt)? ),* )
                    $(, $rm_sig:literal)?
                )
            ),* $(,)?
        ];
        $($body:tt)*
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
            $($body)*

            $(
                #[getter]
                #[pyo3(name = $prop_py)]
                fn $prop(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
                    let out = $crate::ml::object::fetch_py(py, &self.target, $prop_method, &[])?;
                    $crate::ml::object::fetch_output_to_py(py, out)
                }
            )*

            $(
                #[pyo3(
                    name = $m_py,
                    signature = (
                        $( $arg $(= $crate::ml::object::py_default(stringify!($default)))? ),*
                    )
                    $(, text_signature = $m_sig)?
                )]
                fn $m(&self, py: Python<'_>, $( $arg: Py<PyAny> ),*) -> PyResult<Py<PyAny>> {
                    let args = vec![$( $arg.into_bound(py) ),*];
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

            $(
                #[pyo3(
                    name = $rm_py,
                    signature = (
                        $( $rarg $(= $crate::ml::object::py_default(stringify!($rdefault)))? ),*
                    )
                    $(, text_signature = $rm_sig)?
                )]
                fn $rm(&self, py: Python<'_>, $( $rarg: Py<PyAny> ),*) -> PyResult<PyDataFrame> {
                    let args = vec![$( $rarg.into_bound(py) ),*];
                    $crate::ml::object::attribute_relation_py(&self.target, $rm_method, &args)
                }
            )*
        }
    };
}
