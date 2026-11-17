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
//! given, which a fetch sends as PySpark's `_call_java` does. Each argument of a method is written
//! with its default of PySpark, in Python (`(value)`, `(beta = 1.0)`), and a method of a default
//! gives its text signature too (`"($self, beta=1.0)"`), which PyO3 cannot render of a default
//! that it evaluates. A DataFrame method takes any arguments, as PySpark's
//! `try_remote_attribute_relation` does, and is written with the text signature of PySpark's
//! (`"($self, residualsType='deviance')"`).

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

            /// The Param objects, as `Params.params`, of its walk (`ParamsHolder::walk`).
            #[getter]
            fn params<'py>(slf: &Bound<'py, Self>) -> PyResult<Vec<Bound<'py, PyAny>>> {
                $crate::ml::params_api::ParamsHolder::walk(&*slf.try_borrow()?, slf.py())?;
                $crate::ml::params_api::params(slf.as_any())
            }

            /// The AttributeError of an attribute that the object does not have, as CPython's of
            /// PySpark's class (see `params_api::no_attribute`).
            fn __getattr__(slf: &Bound<'_, Self>, name: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                Err($crate::ml::params_api::no_attribute(slf.as_any(), name))
            }

            /// `copy.copy`, as of PySpark's object (see `params_api::shallow_copy`).
            fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                $crate::ml::params_api::shallow_copy(slf)
            }

            /// The state that `copy.deepcopy` and `pickle` take (see `params_api::get_state`).
            fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyDict>> {
                $crate::ml::params_api::get_state(slf)
            }

            /// The object of a state of `__getstate__`.
            fn __setstate__(slf: &Bound<'_, Self>, state: &Bound<'_, PyAny>) -> PyResult<()> {
                $crate::ml::params_api::set_state(slf, state)
            }

            /// The set values and the defaults of the params by name, of which the public class
            /// makes PySpark's param maps of the object (see `params_api::native_params`).
            #[pyo3(name = "_native_params")]
            fn __native_params<'py>(
                slf: &Bound<'py, Self>,
            ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
                $crate::ml::params_api::native_params(slf)
            }

            /// Make the params those of PySpark's param maps of the object, of their values by
            /// name (see `params_api::sync_params`).
            #[pyo3(name = "_native_sync")]
            fn __native_sync(
                slf: &Bound<'_, Self>,
                set: &Bound<'_, PyDict>,
                defaults: &Bound<'_, PyDict>,
            ) -> PyResult<()> {
                $crate::ml::params_api::sync_params(slf, set, defaults)
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
    // The arms of `params_only` come first, each of a section that is not optional: an optional
    // one before the body of any tokens is ambiguous ("local ambiguity"). The arms without it give
    // an empty one.
    (
        $ty:ty;
        jvm: $jvm:literal, op: $op:ident;
        params: [
            $( $param:literal => ($getter_py:literal, $getter:ident, $attr:ident) ),* $(,)?
        ];
        params_only: [ $( $po:literal => $po_attr:ident ),* $(,)? ];
        $($body:tt)*
    ) => {
        params_api_block! {
            $ty;
            params: [ $( $param => ($getter_py, $getter, $attr) ),* ];
            params_only: [ $( $po => $po_attr ),* ];
            {
                $($body)*

                /// The uid, as `Identifiable.uid`.
                #[getter]
                fn uid(&self) -> String {
                    $crate::ml::params_api::NativeInstance::uid(self)
                }

                /// Set the uid, as an assignment of PySpark's `uid` of the instance does (and its
                /// `_resetUid`).
                #[setter(uid)]
                fn __set_uid(&mut self, uid: String) {
                    $crate::ml::params_api::NativeInstance::set_uid(self, &uid)
                }

                /// The uid, as `Identifiable.__repr__`.
                fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                    slf.getattr("uid")?.extract()
                }

                /// As `JavaParams.clear` over Spark Connect, which does nothing: its
                /// `try_remote_intercept` takes any positional arguments and returns None.
                #[pyo3(name = "clear", signature = (*args), text_signature = "($self, param)")]
                fn __params_clear(&self, args: &Bound<'_, pyo3::types::PyTuple>) {
                    let _ = args;
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
                fn write(slf: &Bound<'_, Self>) -> PyResult<Py<$crate::ml::object::PyMLWriter>> {
                    let writer = $crate::ml::params_api::operator_writer(
                        slf,
                        $jvm,
                        spark_connect::ml::OperatorType::$op,
                    )?
                    .with_instance(slf.as_any());
                    $crate::ml::object::new_py(slf.py(), writer)
                }

                /// Save to `path`, as `MLWritable.save`: `self.write().save(path)`.
                fn save(slf: &Bound<'_, Self>, path: &Bound<'_, PyAny>) -> PyResult<()> {
                    slf.call_method0("write")?.call_method1("save", (path,))?;
                    Ok(())
                }

                /// The reader of the class, as `JavaMLReadable.read` (`RemoteMLReader`).
                #[classmethod]
                fn read(
                    cls: &Bound<'_, pyo3::types::PyType>,
                ) -> PyResult<Py<$crate::ml::object::PyMLReader>> {
                    let reader = $crate::ml::object::PyMLReader::of_class::<$ty>(
                        cls,
                        $jvm,
                        spark_connect::ml::OperatorType::$op,
                        <$ty>::__from_read,
                    )?;
                    $crate::ml::object::new_py(cls.py(), reader)
                }

                /// Load from `path`, as `MLReadable.load`: `cls.read().load(path)`.
                #[classmethod]
                fn load(
        cls: &Bound<'_, pyo3::types::PyType>,
        path: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
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
        jvm: $jvm:literal, op: $op:ident;
        params: [
            $( $param:literal => ($getter_py:literal, $getter:ident, $attr:ident) ),* $(,)?
        ];
        $($body:tt)*
    ) => {
        native_pymethods! {
            $ty;
            jvm: $jvm, op: $op;
            params: [ $( $param => ($getter_py, $getter, $attr) ),* ];
            params_only: [];
            $($body)*
        }
    };
    (
        $ty:ty;
        persistence: python;
        params: [
            $( $param:literal => ($getter_py:literal, $getter:ident, $attr:ident) ),* $(,)?
        ];
        params_only: [ $( $po:literal => $po_attr:ident ),* $(,)? ];
        $($body:tt)*
    ) => {
        params_api_block! {
            $ty;
            params: [ $( $param => ($getter_py, $getter, $attr) ),* ];
            params_only: [ $( $po => $po_attr ),* ];
            {
                $($body)*

                /// The uid, as `Identifiable.uid`.
                #[getter]
                fn uid(&self) -> String {
                    $crate::ml::params_api::NativeInstance::uid(self)
                }

                /// Set the uid, as an assignment of PySpark's `uid` of the instance does (and its
                /// `_resetUid`).
                #[setter(uid)]
                fn __set_uid(&mut self, uid: String) {
                    $crate::ml::params_api::NativeInstance::set_uid(self, &uid)
                }

                /// The uid, as `Identifiable.__repr__`.
                fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
                    slf.getattr("uid")?.extract()
                }

                /// As `Params.clear` of a Python class of PySpark (a Pipeline, a validator): the
                /// param, if set, is unset.
                #[pyo3(name = "clear")]
                fn __params_clear(slf: &Bound<'_, Self>, param: &Bound<'_, PyAny>) -> PyResult<()> {
                    $crate::ml::params_api::clear(slf, param)
                }
            }
        }
    };
    (
        $ty:ty;
        persistence: python;
        params: [
            $( $param:literal => ($getter_py:literal, $getter:ident, $attr:ident) ),* $(,)?
        ];
        $($body:tt)*
    ) => {
        native_pymethods! {
            $ty;
            persistence: python;
            params: [ $( $param => ($getter_py, $getter, $attr) ),* ];
            params_only: [];
            $($body)*
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
///   `functools.cache` of the optional `cached_methods` (`"toLocal" => (to_local, make)`, a section
///   before `cached`), which a copy does not share;
/// - `transform`, `copy`, `__repr__` (the server's `toString`, as `JavaModel.__repr__`), `write`,
///   `save`, `read` and `load`, `fitted` (of a fit), `of_constructor` (of an object that a fetch
///   returns, as PySpark's constructor makes the model of one), and the class's own methods
///   `$body`.
///
/// A model class constructs as PySpark's of no `java_model`: a model of no server-side object (see
/// `ModelCore::unbound`).
///
/// Of `base` before `jvm: $jvm`, it is the base class of model classes of PySpark's base
/// class `$name` (`LDAModel`, of `LocalLDAModel` and `DistributedLDAModel`): a
/// `#[pyclass(subclass)]` of all these members but the persistence, which PySpark's base class
/// does not have. A subclass (`#[pyclass(extends = $ty)]`, of no fields) has the persistence and
/// its own members, and makes its objects of the base's `__fitted_with` and `__loaded_with`, of a
/// `make` that makes an object of the subclass of a core, of which a copy is of the same class
/// (see `NativeInstance::duplicate`).
macro_rules! native_model {
    (
        $ty:ident = $name:literal in $module:literal, jvm: $jvm:literal;
        $($rest:tt)*
    ) => {
        native_model! {
            @emit $ty = $name in $module, jvm: $jvm;
            pyclass: [subclass];
            persistence: {
                fn write(slf: &Bound<'_, Self>) -> PyResult<Py<$crate::ml::object::PyMLWriter>> {
                    let writer = slf.try_borrow()?.core.writer(slf.py())?.with_instance(slf.as_any());
                    $crate::ml::object::new_py(slf.py(), writer)
                }

                fn save(slf: &Bound<'_, Self>, path: &Bound<'_, PyAny>) -> PyResult<()> {
                    slf.call_method0("write")?.call_method1("save", (path,))?;
                    Ok(())
                }

                #[classmethod]
                fn read(
                    cls: &Bound<'_, pyo3::types::PyType>,
                ) -> PyResult<Py<$crate::ml::object::PyMLReader>> {
                    let reader = $crate::ml::object::PyMLReader::of_class::<$ty>(
                        cls,
                        $jvm,
                        spark_connect::ml::OperatorType::Model,
                        <$ty>::__from_read,
                    )?;
                    $crate::ml::object::new_py(cls.py(), reader)
                }

                #[classmethod]
                fn load(
        cls: &Bound<'_, pyo3::types::PyType>,
        path: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
                    Ok(cls.call_method0("read")?.call_method1("load", (path,))?.unbind())
                }
            };
            $($rest)*
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
                Self::__fitted_with(py, model, df, summary, Self::__make)
            }

            /// The model of the server-side object that `object` gives of the uid of the model, as
            /// PySpark makes the model of a bare object (a tree of an ensemble,
            /// `DecisionTreeClassificationModel(RemoteModelRef(id))`) or of a class method
            /// (`StringIndexerModel.from_labels`, which gives the model of `StringIndexerModel()`
            /// the `_java_obj` of the object of its uid): of the constructor of its class, whose
            /// `JavaModel.__init__` makes its uid and its param maps (the defaults in the order in
            /// which it sets them), without a training summary or cached members, and then of the
            /// object (see `ModelCore::bind`).
            #[allow(dead_code)]
            pub(crate) fn of_constructor<'py>(
                py: Python<'py>,
                object: impl FnOnce(&str) -> PyResult<spark_connect::ml::MlRef>,
            ) -> PyResult<Bound<'py, Self>> {
                let class = $crate::ml::object::public_class::<Self>(py)
                    .unwrap_or_else(|| py.get_type::<Self>());
                let model = class.call0()?.cast_into::<Self>()?;
                let uid = model.try_borrow()?.core.uid();
                let object = object(&uid)?;
                model.try_borrow_mut()?.core.bind(object);
                Ok(model)
            }

            fn __from_read<'py>(
                py: Python<'py>,
                read: spark_connect::ml::ReadResult,
            ) -> PyResult<Py<PyAny>> {
                let this = Self::__loaded_with(py, $jvm, $name, read, Self::__make)?;
                Ok($crate::ml::object::new_object(py, this)?.into_any().unbind())
            }
        }
    };
    (
        $ty:ident = $name:literal in $module:literal, base, jvm: $jvm:literal;
        $($rest:tt)*
    ) => {
        native_model! {
            @emit $ty = $name in $module, jvm: $jvm;
            pyclass: [subclass];
            persistence: {};
            $($rest)*
        }
    };
    (
        @emit $ty:ident = $name:literal in $module:literal, jvm: $jvm:literal;
        pyclass: [ $($flag:ident)? ];
        persistence: { $($persistence:tt)* };
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
            $( $rm_py:literal => ($rm:ident, $rm_method:literal, $rm_sig:literal) ),* $(,)?
        ];
        summary: $summary:expr;
        // Before `cached`: an optional section before the body of any tokens is ambiguous.
        $( cached_methods: [ $( $cm_py:literal => ($cm:ident, $cm_make:expr) ),* $(,)? ]; )?
        cached: [ $( $c_py:literal => ($c:ident, $c_make:expr) ),* $(,)? ];
        $($body:tt)*
    ) => {
        #[pyclass($($flag,)? name = $name, module = $module)]
        pub struct $ty {
            core: $crate::ml::object::ModelCore,
            /// The object of the class of a core, of the class of this one (a subclass of a base
            /// class), as a copy makes it.
            make: fn(Python<'_>, $crate::ml::object::ModelCore) -> PyResult<Py<$ty>>,
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

            /// Unset the param `name`, as `del self._paramMap[param]` of PySpark's `_clear`.
            fn unset(&mut self, name: &'static str) -> PyResult<()> {
                let params = std::mem::take(self.core.model.params_mut())
                    .clear_param(&$crate::ml::params_api::unsendable_key(name))
                    .clear_param(name);
                *self.core.model.params_mut() = params;
                Ok(())
            }

            fn put_own_default(&mut self, name: &'static str, value: Py<PyAny>) -> bool {
                let defaults = &mut self.core.own_defaults;
                $crate::ml::params_api::put_own_default_in(defaults, name, value);
                true
            }

            /// The walk reads the cached members, made once (see `__prime`).
            fn walk(&self, py: Python<'_>) -> PyResult<()> {
                self.__prime(py)
            }
        }

        impl $crate::ml::params_api::NativeInstance for $ty {
            fn duplicate(&self, py: Python<'_>) -> PyResult<Py<Self>> {
                (self.make)(py, self.core.duplicate(py))
            }

            fn uid(&self) -> String {
                self.core.uid()
            }

            fn set_uid(&mut self, uid: &str) {
                self.core.model.operator_mut().uid = uid.to_string();
            }

            /// A model of a server-side object is not copied deep nor pickled, as PySpark's of a
            /// `RemoteModelRef`; one of no server-side object is.
            fn check_picklable(&self) -> PyResult<()> {
                if self.core.is_unbound() {
                    Ok(())
                } else {
                    Err($crate::ml::params_api::lock_error())
                }
            }
        }

        impl $ty {
            /// The model of a fit (see `fitted`), whose copies `make` makes.
            #[allow(dead_code)]
            fn __fitted_with(
                py: Python<'_>,
                model: Box<dyn Model>,
                df: &DataFrame,
                summary: $crate::ml::object::SummaryDataset,
                make: fn(Python<'_>, $crate::ml::object::ModelCore) -> PyResult<Py<$ty>>,
            ) -> PyResult<Self> {
                let core = $crate::ml::object::ModelCore::fitted(py, model, df, summary, $summary)?;
                let this = Self { core, make };
                this.__prime(py)?;
                Ok(this)
            }

            /// The model of what a `Read` of the JVM class `jvm` returns, of PySpark's class
            /// `name`, with its cached members, whose copies `make` makes.
            #[allow(dead_code)]
            fn __loaded_with(
                py: Python<'_>,
                jvm: &str,
                name: &str,
                read: spark_connect::ml::ReadResult,
                make: fn(Python<'_>, $crate::ml::object::ModelCore) -> PyResult<Py<$ty>>,
            ) -> PyResult<Self> {
                let core = $crate::ml::object::ModelCore::loaded(py, jvm, name, read)?;
                let this = Self { core, make };
                this.__prime(py)?;
                Ok(this)
            }

            /// Read the cached members, as the `params` walk of PySpark's `_resetUid` does.
            #[allow(clippy::unused_unit, unused_variables)]
            fn __prime(&self, py: Python<'_>) -> PyResult<()> {
                $( self.core.cached(py, $c_py, || ($c_make)(py, &self.core))?; )*
                Ok(())
            }

            /// The object of this class of the core `core`, as a copy makes it.
            fn __make(py: Python<'_>, core: $crate::ml::object::ModelCore) -> PyResult<Py<Self>> {
                let this = Self { core, make: Self::__make };
                Ok($crate::ml::object::new_object(py, this)?.unbind())
            }

            /// The core model of the object, of its server-side object and its params, which the
            /// composite of a loaded OneVsRestModel holds of its loaded models.
            #[allow(dead_code)]
            pub(crate) fn core_model(&self) -> Box<dyn Model> {
                self.core.model.clone_box()
            }
        }

        impl $crate::ml::object::ModelObject for $ty {
            /// The object of the model, of its class (the public class of its native class, or of
            /// a subclass of a base class), as a copy makes it.
            fn into_object(self, py: Python<'_>) -> PyResult<Py<Self>> {
                let make = self.make;
                make(py, self.core)
            }
        }

        params_api_block! {
            $ty;
            params: [ $( $param => ($getter_py, $getter, $attr) ),* ];
            params_only: [ $( $( $po => $po_attr ),* )? ];
            {
                $($body)*

                /// A model of no server-side object, as PySpark's `JavaModel(java_model=None)`
                /// (see `ModelCore::unbound`). Another `java_model` is the AssertionError of
                /// PySpark's over Spark Connect, which takes a `RemoteModelRef` only.
                #[new]
                #[pyo3(signature = (java_model=None))]
                fn __new__(py: Python<'_>, java_model: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
                    if java_model.is_some() {
                        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
                    }
                    let core = $crate::ml::object::ModelCore::unbound(py, $jvm, $name)?;
                    Ok(Self { core, make: Self::__make })
                }

                #[getter]
                fn uid(&self) -> String {
                    self.core.uid()
                }

                /// Set the uid, as an assignment of PySpark's `uid` of the instance does (and its
                /// `_resetUid`).
                #[setter(uid)]
                fn __set_uid(&mut self, uid: String) {
                    $crate::ml::params_api::NativeInstance::set_uid(self, &uid)
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

                /// As `JavaParams.clear` over Spark Connect, which does nothing: its
                /// `try_remote_intercept` takes any positional arguments and returns None.
                #[pyo3(name = "clear", signature = (*args), text_signature = "($self, param)")]
                fn __params_clear(&self, args: &Bound<'_, pyo3::types::PyTuple>) {
                    let _ = args;
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

                $($persistence)*

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
                    /// A DataFrame method, as PySpark's `try_remote_attribute_relation` sends it
                    /// over Spark Connect: the server's method of the positional arguments only.
                    /// So keyword arguments are dropped, and no arity is checked (a call of other
                    /// arguments than the server's method takes fails on the server when the
                    /// result is analyzed).
                    #[pyo3(name = $rm_py, signature = (*args, **_kwargs), text_signature = $rm_sig)]
                    fn $rm(
                        slf: &Bound<'_, Self>,
                        args: &Bound<'_, pyo3::types::PyTuple>,
                        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
                    ) -> PyResult<PyDataFrame> {
                        let target = slf.try_borrow()?.core.target()?;
                        let args: Vec<Bound<'_, PyAny>> = args.iter().collect();
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

/// A summary class `$ty` of PySpark's summary class `$name` (of the module `$module`), over the
/// server-side object of the summary (see [`SummaryObject`](crate::ml::object::SummaryObject)): a
/// property of `$properties`, a method of `$methods`, a DataFrame property of `$relations` and a
/// DataFrame method of `$relation_methods` (of the summary dataset of a training summary; see the
/// module doc of their arguments), each as the method of the object that PySpark's `_call_java` or
/// attribute relation reads, and the class's own methods `$body`.
///
/// A summary of `extends $base` is a subclass of the native summary `$base`, as PySpark's summary
/// class is of `$base`'s (a training summary of its summary, `BinaryLogisticRegressionSummary` of
/// `LogisticRegressionSummary`): it has the members of `$base`, and its own of the lists, and its
/// server-side object is `$base`'s.
///
/// Each class constructs as PySpark's `JavaWrapper.__init__(java_obj=None)`, of no server-side
/// object: the `__init__` of the public class keeps the `java_obj` in the object's attributes
/// (`_java_obj`), which the members read at each call, of PySpark's errors of a value that names no
/// object (see `SummaryObject::target`).
macro_rules! native_summary {
    (
        $ty:ident = $name:literal in $module:literal extends $base:ty;
        $($rest:tt)*
    ) => {
        #[pyclass(subclass, extends = $base, name = $name, module = $module)]
        pub struct $ty;

        impl $ty {
            /// The summary of the server-side object `target`.
            pub(crate) fn from(target: spark_connect::ml::MlTarget) -> pyo3::PyClassInitializer<Self> {
                pyo3::PyClassInitializer::<$base>::from(<$base>::from(target)).add_subclass(Self)
            }

            /// The summary of PySpark's constructor.
            pub(crate) fn given() -> pyo3::PyClassInitializer<Self> {
                pyo3::PyClassInitializer::<$base>::from(<$base>::given()).add_subclass(Self)
            }

            /// The server-side object of the summary `slf`, as its members that fetch read it
            /// (see `SummaryObject::target`).
            pub(crate) fn fetch_target(
                slf: &Bound<'_, Self>,
            ) -> PyResult<spark_connect::ml::MlTarget> {
                <$base>::fetch_target(slf.as_super())
            }

            /// The server-side object of the summary `slf`, as its DataFrame members read it (see
            /// `SummaryObject::relation_target`).
            #[allow(dead_code)] // Read by a subclass of DataFrame members of its own only.
            pub(crate) fn summary_target(
                slf: &Bound<'_, Self>,
            ) -> PyResult<spark_connect::ml::MlTarget> {
                <$base>::summary_target(slf.as_super())
            }
        }

        native_summary!(@members $ty; {
            /// PySpark's `JavaWrapper.__init__(java_obj=None)`, whose `java_obj` the `__init__` of
            /// the public class keeps in the object's attributes.
            #[new]
            #[pyo3(signature = (java_obj=None))]
            fn new(java_obj: Option<&Bound<'_, PyAny>>) -> pyo3::PyClassInitializer<Self> {
                let _ = java_obj;
                Self::given()
            }
        }; $($rest)*);
    };

    (
        $ty:ident = $name:literal in $module:literal;
        $($rest:tt)*
    ) => {
        #[pyclass(subclass, name = $name, module = $module)]
        pub struct $ty {
            object: $crate::ml::object::SummaryObject,
        }

        impl From<spark_connect::ml::MlTarget> for $ty {
            fn from(target: spark_connect::ml::MlTarget) -> Self {
                Self {
                    object: $crate::ml::object::SummaryObject::Target(Box::new(target)),
                }
            }
        }

        impl $ty {
            /// The summary of PySpark's constructor.
            pub(crate) fn given() -> Self {
                Self {
                    object: $crate::ml::object::SummaryObject::Given,
                }
            }

            /// The server-side object of the summary `slf`, as its members that fetch read it
            /// (see `SummaryObject::target`).
            pub(crate) fn fetch_target(
                slf: &Bound<'_, Self>,
            ) -> PyResult<spark_connect::ml::MlTarget> {
                let object = slf.try_borrow()?.object.clone_ref();
                object.target(slf.as_any())
            }

            /// The server-side object of the summary `slf`, as its DataFrame members read it (see
            /// `SummaryObject::relation_target`).
            pub(crate) fn summary_target(
                slf: &Bound<'_, Self>,
            ) -> PyResult<spark_connect::ml::MlTarget> {
                let object = slf.try_borrow()?.object.clone_ref();
                object.relation_target(slf.as_any())
            }
        }

        native_summary!(@members $ty; {
            /// PySpark's `JavaWrapper.__init__(java_obj=None)`, whose `java_obj` the `__init__` of
            /// the public class keeps in the object's attributes.
            #[new]
            #[pyo3(signature = (java_obj=None))]
            fn new(java_obj: Option<&Bound<'_, PyAny>>) -> Self {
                let _ = java_obj;
                Self::given()
            }

            /// The AttributeError of an attribute that the summary does not have, as CPython's of
            /// PySpark's class (see `params_api::no_attribute`). CPython calls `__getattr__` also
            /// when the getter of a member raised an AttributeError (of a summary of no server-side
            /// object, or of no `_summary_dataset`): of a member of the class, the error of its
            /// getter, which runs again (it raises before any request), as PySpark's class has no
            /// `__getattr__` to raise another.
            fn __getattr__(slf: &Bound<'_, Self>, name: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                let get = pyo3::intern!(slf.py(), "__get__");
                if let Some(member) = $crate::ml::object::class_member(slf.as_any(), name)? {
                    if member.hasattr(get)? {
                        return Ok(member.call_method1(get, (slf, slf.get_type()))?.unbind());
                    }
                }
                Err($crate::ml::params_api::no_attribute(slf.as_any(), name))
            }

            /// `copy.copy`, as of PySpark's summary: of the same server-side object, of the class
            /// and the attributes of the summary. The copy is made by the class's `__new__` (of no
            /// `java_obj`), as the native class of a subclass of this one (a training summary) is
            /// of another deallocator than this one's, of which CPython assigns no `__class__`.
            fn __copy__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
                let object = slf.try_borrow()?.object.clone_ref();
                let class = slf.get_type();
                let that = class.call_method1("__new__", (&class,))?;
                that.cast::<Self>()?.try_borrow_mut()?.object = object;
                $crate::ml::params_api::copy_class_and_attributes(slf.as_any(), &that)?;
                Ok(that)
            }

            /// `copy.deepcopy`, as of PySpark's summary: of a summary of a server-side object, the
            /// TypeError of the lock of the `RemoteModelRef` of its model, which PySpark's copies
            /// (the `_remote_model_obj` of a fit's summary, the `__source_transformer__` of an
            /// evaluation's); of a summary of PySpark's constructor, a summary of the class and of
            /// copies of the attributes, as of the reduce of a plain object (`__reduce_ex__`).
            fn __deepcopy__<'py>(
                slf: &Bound<'py, Self>,
                memo: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, PyAny>> {
                if let $crate::ml::object::SummaryObject::Target(_) = &slf.try_borrow()?.object {
                    return Err($crate::ml::params_api::lock_error());
                }
                let py = slf.py();
                let reduced = $crate::reduce::plain_reduce(slf.as_any())?;
                let mut args = vec![slf.clone().into_any(), memo.clone()];
                args.extend(reduced.iter());
                py.import("copy")?
                    .call_method1("_reconstruct", pyo3::types::PyTuple::new(py, args)?)
            }

            /// `pickle`, as of PySpark's summary: of a summary of a server-side object, the error
            /// of the first of its attributes that PySpark's does not pickle (see
            /// `object::summary_pickle_error`); of a summary of PySpark's constructor, the reduce
            /// of a plain object, of the class and its attributes (see `reduce::plain_reduce`).
            fn __reduce_ex__<'py>(
                slf: &Bound<'py, Self>,
                protocol: &Bound<'py, PyAny>,
            ) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
                if let $crate::ml::object::SummaryObject::Target(target) = &slf.try_borrow()?.object
                {
                    let py = slf.py();
                    return Err($crate::ml::object::summary_pickle_error(py, target, protocol));
                }
                $crate::reduce::plain_reduce(slf.as_any())
            }
        }; $($rest)*);
    };

    (
        @members $ty:ident; { $($own:tt)* };
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
            $( $rm_py:literal => ($rm:ident, $rm_method:literal, $rm_sig:literal) ),* $(,)?
        ];
        $($body:tt)*
    ) => {
        #[pymethods]
        impl $ty {
            $($own)*

            $($body)*

            $(
                #[getter]
                #[pyo3(name = $prop_py)]
                fn $prop(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                    let py = slf.py();
                    let target = Self::fetch_target(slf)?;
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
                    let target = Self::fetch_target(slf)?;
                    let args = vec![$( $arg.into_bound(py) ),*];
                    let out = $crate::ml::object::fetch_py(py, &target, $m_method, &args)?;
                    $crate::ml::object::fetch_output_to_py(py, out)
                }
            )*

            $(
                #[getter]
                #[pyo3(name = $r_py)]
                fn $r(slf: &Bound<'_, Self>) -> PyResult<PyDataFrame> {
                    let target = Self::summary_target(slf)?;
                    $crate::ml::object::attribute_relation_py(&target, $r_method, &[])
                }
            )*

            $(
                /// A DataFrame method, as PySpark's `try_remote_attribute_relation` sends it over
                /// Spark Connect: the server's method of the positional arguments only. So
                /// keyword arguments are dropped, and no arity is checked (a call of other
                /// arguments than the server's method takes fails on the server when the result
                /// is analyzed).
                #[pyo3(name = $rm_py, signature = (*args, **_kwargs), text_signature = $rm_sig)]
                fn $rm(
                    slf: &Bound<'_, Self>,
                    args: &Bound<'_, pyo3::types::PyTuple>,
                    _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
                ) -> PyResult<PyDataFrame> {
                    let target = Self::summary_target(slf)?;
                    let args: Vec<Bound<'_, PyAny>> = args.iter().collect();
                    $crate::ml::object::attribute_relation_py(&target, $rm_method, &args)
                }
            )*
        }
    };
}
