//! `copy.deepcopy` of the objects of the native classes of the data types, of no Python call of
//! each object of a type.
//!
//! PySpark's data types are plain classes, which `copy.deepcopy` copies of `object.__reduce_ex__(4)`
//! and `copy._reconstruct`: a new object of `cls.__new__(cls)`, kept in the memo, and then the
//! deep copy of its `__dict__`, which `__dict__.update` sets. The native classes give the same
//! reduce (see `types::type_reduce_ex`), of which Python's `copy` makes the same copies and the
//! same entries of the memo, but of a few calls of Python code for each object of a type, which a
//! struct of many fields holds many of. This copier, which `copy`'s dispatch of exact classes calls
//! for an object of a native class itself (`copy._deepcopy_dispatch`), so that a subclass keeps
//! Python's protocol, makes what `copy.deepcopy` makes of it, of the same objects in the same
//! order: the copies, the entries of the memo, of the objects that `copy` copies and of the list of
//! `_keep_alive` (`memo[id(memo)]`), and the levels of the recursion that its frames count. A class
//! of a Python hook of the protocol, a patch of a native class over one of its methods, or an
//! entry of `copyreg.dispatch_table`, as an object of such a hook of its own, is copied as
//! `copy.deepcopy` copies it, of those hooks.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyTuple, PyType};

/// The attributes of a class that `copy.deepcopy` of its objects, and the reduce of a data type,
/// read through the class: a patch of a native class over one, or of a class of its MRO, is a
/// hook of Python that the copier honours (see `Kept::plain`).
const CLASS_HOOKS: [&str; 11] = [
    "__deepcopy__",
    "__reduce_ex__",
    "__reduce__",
    "__getstate__",
    "__setstate__",
    "__new__",
    "__getattribute__",
    "__getattr__",
    "__slotnames__",
    "json",
    "jsonValue",
];

/// Of those, the ones that an object's own `__dict__` takes over the method of its class, which
/// `getattr` of the object reads first.
const OBJECT_HOOKS: [&str; 6] = [
    "__deepcopy__",
    "__reduce_ex__",
    "__reduce__",
    "__getstate__",
    "json",
    "jsonValue",
];

/// How `copy.deepcopy` copies an object of a native class, of no hook.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A type of PySpark's `DataTypeSingleton`, whose reduce is `_parse_datatype_json_string` of
    /// its JSON, which gives the one instance: the copy is the object, which `_reconstruct` keeps
    /// in the memo.
    Singleton,
    /// A type whose state is its `__dict__`, whose reduce is `copyreg.__newobj__` of its class and
    /// the `__dict__`.
    Dict,
}

/// A native class of the copier, of its kind and of what its MRO gives of each of `CLASS_HOOKS`
/// when it is registered: the classes of its MRO, of a view of the namespace of each, and the
/// attribute that the lookup of each name finds, or None of none.
struct KeptClass {
    class: Py<PyType>,
    kind: Kind,
    mro: Py<PyTuple>,
    namespaces: Vec<Py<PyAny>>,
    hooks: Vec<Option<Py<PyAny>>>,
}

/// What the copier reads of the modules `copy`, `copyreg` and `pyspark.sql.types`, as they are
/// when it is registered.
struct Kept {
    /// `copy._deepcopy_dispatch`, which `copy.deepcopy` reads at each call.
    dispatch: Py<PyDict>,
    /// The copiers of `copy` of atomic objects, of dicts and of lists.
    atomic: Py<PyAny>,
    dict: Py<PyAny>,
    list: Py<PyAny>,
    /// `copy.deepcopy`, `copy._reconstruct` and `copy.Error`.
    deepcopy: Py<PyAny>,
    reconstruct: Py<PyAny>,
    error: Py<PyAny>,
    /// `copyreg.dispatch_table` and `copyreg._slotnames`.
    copyreg_dispatch: Py<PyAny>,
    slotnames: Py<PyAny>,
    /// `pyspark.sql.types` and its `_parse_datatype_json_string`, which the reduce of a singleton
    /// names.
    types: Py<PyAny>,
    parse_json: Py<PyAny>,
    /// The copier, as the dispatch holds it.
    copier: Py<PyAny>,
    classes: Vec<KeptClass>,
}

static KEPT: PyOnceLock<Kept> = PyOnceLock::new();

/// The classes of the copier: the singletons, and the types whose state is their `__dict__`.
fn native_classes(py: Python<'_>) -> Vec<(Bound<'_, PyType>, Kind)> {
    use crate::types::*;
    fn singleton(class: Bound<'_, PyType>) -> (Bound<'_, PyType>, Kind) {
        (class, Kind::Singleton)
    }
    fn dict(class: Bound<'_, PyType>) -> (Bound<'_, PyType>, Kind) {
        (class, Kind::Dict)
    }
    vec![
        singleton(py.get_type::<PyNullType>()),
        singleton(py.get_type::<PyBooleanType>()),
        singleton(py.get_type::<PyByteType>()),
        singleton(py.get_type::<PyShortType>()),
        singleton(py.get_type::<PyIntegerType>()),
        singleton(py.get_type::<PyLongType>()),
        singleton(py.get_type::<PyFloatType>()),
        singleton(py.get_type::<PyDoubleType>()),
        singleton(py.get_type::<PyBinaryType>()),
        singleton(py.get_type::<PyDateType>()),
        singleton(py.get_type::<PyTimestampType>()),
        singleton(py.get_type::<PyTimestampNTZType>()),
        singleton(py.get_type::<PyCalendarIntervalType>()),
        dict(py.get_type::<PyStructType>()),
        dict(py.get_type::<PyStructField>()),
        dict(py.get_type::<PyArrayType>()),
        dict(py.get_type::<PyMapType>()),
        dict(py.get_type::<PyDecimalType>()),
        dict(py.get_type::<PyStringType>()),
        dict(py.get_type::<PyCharType>()),
        dict(py.get_type::<PyVarcharType>()),
        dict(py.get_type::<PyTimeType>()),
        dict(py.get_type::<PyDayTimeIntervalType>()),
        dict(py.get_type::<PyYearMonthIntervalType>()),
        dict(py.get_type::<PyGeometryType>()),
        dict(py.get_type::<PyGeographyType>()),
    ]
}

/// What the lookups of `CLASS_HOOKS` through the namespaces `namespaces` of the MRO of a class
/// find: the attribute of the first class that has it, or None.
fn hooks_of(py: Python<'_>, namespaces: &[Py<PyAny>]) -> PyResult<Vec<Option<Py<PyAny>>>> {
    CLASS_HOOKS
        .iter()
        .map(|name| {
            for namespace in namespaces {
                // A view of the namespace of a class, of no `AttributeError` of a name that it
                // has not, which a lookup through the class would raise.
                if let Some(found) = namespace
                    .bind(py)
                    .call_method1(pyo3::intern!(py, "get"), (*name,))
                    .map(|found| (!found.is_none()).then_some(found))?
                {
                    return Ok(Some(found.unbind()));
                }
            }
            Ok(None)
        })
        .collect()
}

/// Register the copier `copier` (`_type_deepcopy`) for the native classes of the data types in
/// `copy`'s dispatch of exact classes, of the `_parse_datatype_json_string` of `pyspark.sql.types`
/// as `parse_json`, which `pyspark.sql.types` calls once it is made.
#[pyfunction]
pub(crate) fn _register_type_deepcopy(
    py: Python<'_>,
    copier: Bound<'_, PyAny>,
    parse_json: Bound<'_, PyAny>,
) -> PyResult<()> {
    let copy = py.import("copy")?;
    let copyreg = py.import("copyreg")?;
    let dispatch = copy.getattr("_deepcopy_dispatch")?.cast_into::<PyDict>()?;
    let mut classes = Vec::new();
    for (class, kind) in native_classes(py) {
        let mro = class.getattr("__mro__")?.cast_into::<PyTuple>()?;
        // Of the native classes of the MRO: the namespace of `object`, its last class, takes no
        // attribute, so its lookups are those that it gave.
        let object = py.get_type::<PyAny>();
        let namespaces = mro
            .iter()
            .filter(|base| !base.is(&object))
            .map(|base| Ok(base.getattr("__dict__")?.unbind()))
            .collect::<PyResult<Vec<_>>>()?;
        let hooks = hooks_of(py, &namespaces)?;
        classes.push(KeptClass {
            class: class.clone().unbind(),
            kind,
            mro: mro.unbind(),
            namespaces,
            hooks,
        });
    }
    let kept = Kept {
        dispatch: dispatch.clone().unbind(),
        atomic: copy.getattr("_deepcopy_atomic")?.unbind(),
        dict: copy.getattr("_deepcopy_dict")?.unbind(),
        list: copy.getattr("_deepcopy_list")?.unbind(),
        deepcopy: copy.getattr("deepcopy")?.unbind(),
        reconstruct: copy.getattr("_reconstruct")?.unbind(),
        error: copy.getattr("Error")?.unbind(),
        copyreg_dispatch: copyreg.getattr("dispatch_table")?.unbind(),
        slotnames: copyreg.getattr("_slotnames")?.unbind(),
        types: py.import("pyspark.sql.types")?.into_any().unbind(),
        parse_json: parse_json.unbind(),
        copier: copier.clone().unbind(),
        classes,
    };
    let kept = KEPT.get_or_init(py, || kept);
    for class in &kept.classes {
        dispatch.set_item(class.class.bind(py), &copier)?;
    }
    Ok(())
}

/// The `id()` of `obj`, as `copy` keys its memo of it.
fn id_of<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // SAFETY: of a valid object of the GIL; `PyLong_FromVoidPtr` is what `id()` returns.
    unsafe {
        Bound::from_owned_ptr_or_err(obj.py(), pyo3::ffi::PyLong_FromVoidPtr(obj.as_ptr().cast()))
    }
}

/// A level of the recursion of the interpreter, as a frame of Python's `copy` counts one, of the
/// `RecursionError` of its limit, left when it is dropped.
struct Level;

impl Level {
    fn enter(py: Python<'_>) -> PyResult<Level> {
        // SAFETY: the GIL is held; a check that succeeds is left by the drop of the level.
        if unsafe { pyo3::ffi::Py_EnterRecursiveCall(c"".as_ptr()) } != 0 {
            return Err(PyErr::fetch(py));
        }
        Ok(Level)
    }
}

impl Level {
    /// Levels `count` of frames that a call of Python's `copy` makes and leaves, of the
    /// `RecursionError` of its limit, of none kept.
    fn transient(py: Python<'_>, count: usize) -> PyResult<()> {
        let mut levels = Vec::with_capacity(count);
        for _ in 0..count {
            levels.push(Level::enter(py)?);
        }
        Ok(())
    }
}

impl Drop for Level {
    fn drop(&mut self) {
        // SAFETY: of the check that `Level::enter` entered, of the thread that holds the GIL.
        unsafe { pyo3::ffi::Py_LeaveRecursiveCall() };
    }
}

/// A copy of `copy.deepcopy`, of the memo `memo`.
struct Copier<'py> {
    py: Python<'py>,
    kept: &'py Kept,
    memo: Bound<'py, PyAny>,
    /// The memo, of a dict itself, which its `get` and its items read as dict operations do.
    memo_dict: Option<Bound<'py, PyDict>>,
    /// `id(memo)`, the key of the list of `_keep_alive`.
    memo_id: Bound<'py, PyAny>,
    /// The kinds of the classes that this copy found of no hook, of the pointers of the classes.
    plain: Vec<(usize, Option<Kind>)>,
}

impl<'py> Copier<'py> {
    fn new(kept: &'py Kept, memo: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(Copier {
            py: memo.py(),
            kept,
            memo: memo.clone(),
            memo_dict: memo.cast_exact::<PyDict>().ok().cloned(),
            memo_id: id_of(memo)?,
            plain: Vec::new(),
        })
    }

    /// `memo.get(key, _nil)`, as `copy.deepcopy` reads it.
    fn memo_get(&self, key: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
        if let Some(memo) = &self.memo_dict {
            return memo.get_item(key);
        }
        let nil = PyList::empty(self.py);
        let found = self
            .memo
            .call_method1(pyo3::intern!(self.py, "get"), (key, &nil))?;
        Ok((!found.is(&nil)).then_some(found))
    }

    /// `memo[key] = value`.
    fn memo_set(&self, key: &Bound<'py, PyAny>, value: &Bound<'py, PyAny>) -> PyResult<()> {
        match &self.memo_dict {
            Some(memo) => memo.set_item(key, value),
            None => self.memo.set_item(key, value),
        }
    }

    /// As `copy._keep_alive(x, memo)`: `memo[id(memo)].append(x)`, of a new list `[x]` of none.
    fn keep_alive(&self, x: &Bound<'py, PyAny>) -> PyResult<()> {
        let kept = match &self.memo_dict {
            Some(memo) => memo.get_item(&self.memo_id)?,
            None => match self.memo.get_item(&self.memo_id) {
                Ok(kept) => Some(kept),
                Err(error) if error.is_instance_of::<pyo3::exceptions::PyKeyError>(self.py) => None,
                Err(error) => return Err(error),
            },
        };
        match kept {
            Some(kept) => match kept.cast_exact::<PyList>() {
                Ok(list) => list.append(x),
                Err(_) => kept
                    .call_method1(pyo3::intern!(self.py, "append"), (x,))
                    .map(drop),
            },
            None => self.memo_set(&self.memo_id, &PyList::new(self.py, [x])?.into_any()),
        }
    }

    /// The kind of the native class `class` of no hook now, of one check of each class in a copy:
    /// None of a class whose MRO, or whose lookup of one of `CLASS_HOOKS`, gives another than when
    /// the copier was registered, of an entry of `copyreg.dispatch_table`, of a singleton whose
    /// reduce would name another `_parse_datatype_json_string`, or of a class of `__slotnames__`.
    fn plain(&mut self, class: &Bound<'py, PyType>) -> PyResult<Option<Kind>> {
        let address = class.as_ptr() as usize;
        if let Some((_, kind)) = self.plain.iter().find(|(known, _)| *known == address) {
            return Ok(*kind);
        }
        let kind = self.plain_now(class)?;
        self.plain.push((address, kind));
        Ok(kind)
    }

    fn plain_now(&self, class: &Bound<'py, PyType>) -> PyResult<Option<Kind>> {
        let py = self.py;
        let Some(kept) = self
            .kept
            .classes
            .iter()
            .find(|kept| kept.class.bind(py).is(class))
        else {
            return Ok(None);
        };
        if !class
            .getattr(pyo3::intern!(py, "__mro__"))?
            .is(kept.mro.bind(py))
        {
            return Ok(None);
        }
        if !self
            .kept
            .copyreg_dispatch
            .bind(py)
            .call_method1(pyo3::intern!(py, "get"), (class,))?
            .is_none()
        {
            return Ok(None);
        }
        // As `object.__reduce_ex__` of a plain object reads its class's `copyreg._slotnames`,
        // which keeps them in its `__slotnames__` on the first call: of a class of none, the
        // empty list, of which the state of an object is its `__dict__` alone.
        if kept.kind == Kind::Dict {
            let names = self.kept.slotnames.bind(py).call1((class,))?;
            if names.len()? != 0 {
                return Ok(None);
            }
        }
        let hooks = hooks_of(py, &kept.namespaces)?;
        let same = hooks
            .iter()
            .zip(&kept.hooks)
            .enumerate()
            .all(|(index, (now, then))| {
                // `__slotnames__`, which the first reduce of a plain object sets of its class.
                CLASS_HOOKS[index] == "__slotnames__"
                    || match (now, then) {
                        (Some(now), Some(then)) => now.bind(py).is(then.bind(py)),
                        (None, None) => true,
                        _ => false,
                    }
            });
        if !same {
            return Ok(None);
        }
        if kept.kind == Kind::Singleton
            && !self
                .kept
                .types
                .bind(py)
                .getattr(pyo3::intern!(py, "_parse_datatype_json_string"))?
                .is(self.kept.parse_json.bind(py))
        {
            return Ok(None);
        }
        Ok(Some(kept.kind))
    }

    /// Whether the object `x` holds a hook of `OBJECT_HOOKS` in its own `__dict__`.
    fn object_hooks(&self, x: &Bound<'py, PyAny>) -> PyResult<bool> {
        let py = self.py;
        let dict = x.getattr(pyo3::intern!(py, "__dict__"))?;
        let Ok(dict) = dict.cast::<PyDict>() else {
            return Ok(true);
        };
        if dict.is_empty() {
            return Ok(false);
        }
        for name in OBJECT_HOOKS {
            if dict.contains(name)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// `copy.deepcopy(v, memo)`.
    fn deep(&mut self, v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        // The frame of `deepcopy`.
        let _level = Level::enter(py)?;
        {
            let key = id_of(v)?;
            if let Some(found) = self.memo_get(&key)? {
                return Ok(found);
            }
            let class = v.get_type();
            let copier = self.kept.dispatch.bind(py).get_item(&class)?;
            let copied = match copier {
                Some(copier) if copier.is(self.kept.atomic.bind(py)) => return Ok(v.clone()),
                // The copies that recurse, on a stack with room for a level.
                Some(copier)
                    if copier.is(self.kept.dict.bind(py)) && v.is_exact_instance_of::<PyDict>() =>
                {
                    crate::pbmsg::grow(|| self.deep_dict(v.cast::<PyDict>()?))?
                }
                Some(copier)
                    if copier.is(self.kept.list.bind(py)) && v.is_exact_instance_of::<PyList>() =>
                {
                    crate::pbmsg::grow(|| self.deep_list(v.cast::<PyList>()?))?
                }
                Some(copier) if copier.is(self.kept.copier.bind(py)) => {
                    crate::pbmsg::grow(|| self.copy_native(v))?
                }
                // Of another object, Python's `copy.deepcopy` itself, of its memo and its levels.
                _ => {
                    drop(_level);
                    return self.kept.deepcopy.bind(py).call1((v, &self.memo));
                }
            };
            if !copied.is(v) {
                self.memo_set(&key, &copied)?;
                self.keep_alive(v)?;
            }
            Ok(copied)
        }
    }

    /// `copy._deepcopy_dict(x, memo)`: of the copy of each value, then of its key.
    fn deep_dict(&mut self, x: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let _level = Level::enter(py)?;
        let y = PyDict::new(py);
        self.memo_set(&id_of(x.as_any())?, y.as_any())?;
        // Of `x.items()`, of the `RuntimeError` of a dict that a copy changes.
        for item in x.call_method0(pyo3::intern!(py, "items"))?.try_iter()? {
            let item = item?;
            let item = item.cast::<PyTuple>()?;
            let (key, value) = (item.get_item(0)?, item.get_item(1)?);
            let value = self.deep(&value)?;
            let key = self.deep(&key)?;
            y.set_item(key, value)?;
        }
        Ok(y.into_any())
    }

    /// `copy._deepcopy_list(x, memo)`, of its elements as its iterator gives them.
    fn deep_list(&mut self, x: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let _level = Level::enter(py)?;
        let y = PyList::empty(py);
        self.memo_set(&id_of(x.as_any())?, y.as_any())?;
        let mut index = 0;
        while index < x.len() {
            let element = x.get_item(index)?;
            index += 1;
            y.append(self.deep(&element)?)?;
        }
        Ok(y.into_any())
    }

    /// What `copy.deepcopy` does of the object `x` of a native class of the copier once its memo
    /// has none of it: of a class or an object of a hook of Python, `copy`'s protocol itself;
    /// else `_reconstruct` of the reduce of its kind.
    fn copy_native(&mut self, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let class = x.get_type();
        let kind = match self.plain(&class)? {
            Some(kind) if !self.object_hooks(x)? => kind,
            _ => return self.protocol(x),
        };
        match kind {
            // `_reconstruct(x, memo, _parse_datatype_json_string, (json,))`: the args copy as they
            // are, the call gives the one instance, which the memo keeps, of no state.
            Kind::Singleton => {
                let _level = Level::enter(py)?;
                // The frames of `_reconstruct` below it: of the copy of the args, and of the call
                // of `_parse_datatype_json_string`, whose `json.loads` is the deepest, of four.
                Level::transient(py, 4)?;
                self.memo_set(&id_of(x)?, x)?;
                Ok(x.clone())
            }
            // `_reconstruct(x, memo, copyreg.__newobj__, (cls,), state)`: the object of
            // `cls.__new__(cls)`, which the memo keeps, then the copy of the `__dict__`, of no
            // state of one that is empty, set by `__dict__.update` of the new object, which has no
            // `__setstate__`.
            Kind::Dict => {
                let _level = Level::enter(py)?;
                // The frames of `_reconstruct` below it: of the copy of the args, of three, and
                // of `copyreg.__newobj__`.
                Level::transient(py, 3)?;
                let new = class.getattr(pyo3::intern!(py, "__new__"))?;
                let y = new.call1((&class,))?;
                self.memo_set(&id_of(x)?, &y)?;
                let state = x.getattr(pyo3::intern!(py, "__dict__"))?;
                if state.len()? > 0 {
                    let state = self.deep(&state)?;
                    match y.getattr_opt(pyo3::intern!(py, "__setstate__"))? {
                        Some(setstate) => {
                            setstate.call1((state,))?;
                        }
                        None => {
                            let dict = y.getattr(pyo3::intern!(py, "__dict__"))?;
                            // `dict.update` of a dict, of `PyDict_Update`, as it merges one.
                            match (dict.cast::<PyDict>(), state.cast_exact::<PyDict>()) {
                                (Ok(dict), Ok(state)) => dict.update(state.as_mapping())?,
                                _ => {
                                    dict.call_method1(pyo3::intern!(py, "update"), (state,))?;
                                }
                            }
                        }
                    }
                }
                Ok(y)
            }
        }
    }

    /// What `copy.deepcopy(x, memo)` does of `x` past its dispatch of exact classes: its
    /// `__deepcopy__`, else `_reconstruct` of the reduce of `copyreg.dispatch_table`, of its
    /// `__reduce_ex__(4)` or of its `__reduce__`, of `copy.Error` of none, and `x` itself of a
    /// reduce of a str.
    fn protocol(&mut self, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let class = x.get_type();
        // `getattr(x, name, None)`, of which a None is none, as `copy.deepcopy` takes it.
        let hook =
            |name: &Bound<'py, pyo3::types::PyString>| -> PyResult<Option<Bound<'py, PyAny>>> {
                Ok(x.getattr_opt(name)?.filter(|found| !found.is_none()))
            };
        if let Some(copier) = hook(pyo3::intern!(py, "__deepcopy__"))? {
            return copier.call1((&self.memo,));
        }
        let reductor = self
            .kept
            .copyreg_dispatch
            .bind(py)
            .call_method1(pyo3::intern!(py, "get"), (&class,))?;
        let reduced = if !reductor.is_none() {
            reductor.call1((x,))?
        } else if let Some(reductor) = hook(pyo3::intern!(py, "__reduce_ex__"))? {
            reductor.call1((4,))?
        } else {
            match hook(pyo3::intern!(py, "__reduce__"))? {
                Some(reductor) if reductor.is_truthy()? => reductor.call0()?,
                _ => {
                    let message = format!("un(deep)copyable object of type {}", class.str()?);
                    return Err(PyErr::from_value(
                        self.kept.error.bind(py).call1((message,))?,
                    ));
                }
            }
        };
        if reduced.is_instance_of::<pyo3::types::PyString>() {
            return Ok(x.clone());
        }
        let mut args = vec![x.clone(), self.memo.clone()];
        args.extend(reduced.try_iter()?.collect::<PyResult<Vec<_>>>()?);
        self.kept
            .reconstruct
            .bind(py)
            .call1(PyTuple::new(py, args)?)
    }
}

/// `copy.deepcopy` of the object `x` of a native class of the data types, of the memo `memo`,
/// which `copy`'s dispatch of exact classes calls once the memo has none of `x` (see the module):
/// `copy.deepcopy` then keeps the copy in the memo, and `x` in the list of `_keep_alive`, as of
/// any copy.
#[pyfunction]
pub(crate) fn _type_deepcopy<'py>(
    x: &Bound<'py, PyAny>,
    memo: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = x.py();
    let Some(kept) = KEPT.get(py) else {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "the copier of the data types is not registered",
        ));
    };
    Copier::new(kept, memo)?.copy_native(x)
}
