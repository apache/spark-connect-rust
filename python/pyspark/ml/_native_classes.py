"""The public classes of the native ML classes.

The public class of a native class is a subclass of it and of PySpark's class of the same name,
which a module of `pyspark.ml` has: its objects are instances of every class of the MRO of PySpark's
class, as PySpark's objects are (`MLWritable`, `Identifiable`, `HasMaxIter`, `ClassificationModel`,
the summary bases, the Java wrappers, and PySpark's class itself), and the members of the native
class come first. The native module (`pyspark._pyspark`) makes every object of a native class of
its public class (`_register_public_class`), as the constructor of the public class does.

The public class of a native class is a subclass of the public class of each class of PySpark's of
the same module that is a base of PySpark's class (a training summary of its summary, and the
`LogisticRegressionTrainingSummary` of a `BinaryLogisticRegressionTrainingSummary`), so that
`isinstance` holds of the public classes as of PySpark's, and its metaclass is PySpark's.

A public class constructs, copies and fits as PySpark's class, so that a Python subclass of it
behaves as a subclass of PySpark's:
- Its `__init__` has the parameters of PySpark's `__init__` (of `keyword_only`, which sets
  `_input_kwargs`), and sets the params of its keywords by `setParams`, as PySpark's sets them. Its
  `__new__` binds a call as that `__init__` does, before the native constructor runs (which may
  read the server's defaults), and makes the object of the native constructor; it leaves the
  arguments of a subclass's own `__init__` to it, as `object.__new__` does, and gives an object of
  a subclass a uid of the subclass's name, as `Identifiable._randomUID` does.
- `fit`, `fitMultiple`, `transform`, `evaluate`, `save` and `load` are PySpark's own (of `Estimator`,
  `Transformer`, `Evaluator`, `MLWritable` and `MLReadable`), which run `_fit`, `_transform`,
  `_evaluate`, `copy`, `write` and `read` of the object: a subclass's `_fit` runs. `_fit`,
  `_transform` and `_evaluate` are the native class's, of no param map.
- `copy` is PySpark's own; `copy.copy`, `copy.deepcopy` and `pickle` take the object's reduce, as
  of PySpark's object, of no `__copy__`: an object of the class's `__new__` and the object's
  attributes, PySpark's state (of the objects that the native object holds of a composite model
  too), and of a model of a server-side object, the native copy of the same object, after a lock,
  at which a pickle and a deep copy fail as PySpark's do at the lock of its `RemoteModelRef`.

A public class of a native summary constructs of PySpark's own `JavaWrapper.__init__`, which keeps
the `java_obj` in the object's attributes (`_java_obj`), which the native members read, and has
properties and methods of PySpark's docstrings and signatures over the native members.

A public class of a native writer or reader (`RemoteMLWriter`, `RemoteMLReader`) is a plain class
of the attributes of PySpark's `__init__`, which PySpark's own members set and read, and of the
native `save` (`load`), which reads them at the save (load), as PySpark's does.

A public class of a native class of params keeps them in PySpark's param maps (`_paramMap`,
`_defaultParamMap`), in the object's attributes, and takes PySpark's own Python code over them (the
Params API, the getters and setters, `setParams`, a `keyword_only` constructor) as it is. Before a
native member reads the params (a fit, a transform, an evaluation, a writer, a model's member), the
native object's params are made those of the maps, and so are those of the objects that it holds
(`_sync`): the native object reads them of their native objects. The maps of an object that
PySpark's constructor did not make (a fit's model, a load, a native copy) are made of the native
object's values at their first read, as the other attributes that PySpark's constructor, fit or load
sets (its uid, its `_java_obj`, a model's `_summary`), by the `__getattr__` of the native class (see
`_lazy_attribute`), which are of the object, not of its class. The public class reaches the native
object by the functions of the native module (`_ml_params`, `_ml_sync`, `_ml_java_obj`, ...), of
no member of the class.
"""

import dis
import functools
import importlib
import inspect
import sys
import types
import warnings

from pyspark import _pyspark as _native
from pyspark import keyword_only

# The names of PySpark's code of its server-side object, which a native class's own member
# replaces: the members of PySpark's that read them are not PySpark's own Python code.
_SERVER_NAMES = frozenset(
    {
        "_call_java",
        "_java_obj",
        "_new_java_obj",
        "_create_from_java_class",
        "_from_java",
        "_to_java",
        # The training summary, which PySpark's fit sets of its server-side object (`_summary`,
        # which `summary` reads, and `super().summary` of a model's own `summary`): the native
        # model fetches its own.
        "summary",
        "_summary",
    }
)

# The code of the wrapper of PySpark's `keyword_only`, of every function that it wraps.
_KEYWORD_ONLY_CODE = keyword_only(lambda self: None).__code__


def _names_of(code):
    """The names that `code` reads, of its nested functions and comprehensions too."""
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= _names_of(const)
    return names


def _pysparks_own(member):
    """Whether PySpark's member `member` is its own Python code over the object's other members (the
    Params API, the param getters and setters, `setParams` of `keyword_only`, a property over a
    DataFrame member), which a public class takes as it is; rather than code of PySpark's
    server-side object (`_call_java`, `_java_obj`, a `try_remote_*` wrapper, a cached member),
    which the native class's own member replaces."""
    if isinstance(member, (functools.cached_property, staticmethod, classmethod)):
        return False
    function = member.fget if isinstance(member, property) else member
    if not isinstance(function, types.FunctionType):
        return False
    if function.__code__ is _KEYWORD_ONLY_CODE:
        function = function.__wrapped__
    elif hasattr(function, "__wrapped__"):
        return False
    return not (_SERVER_NAMES & _names_of(function.__code__))


# The attributes of PySpark's object that its constructor, fit or load sets, which an object of
# a public class that PySpark's constructor did not make (of a fit, a load, a native copy) makes at
# their first read, of its native object (see `_lazy_attribute`).
_LAZY_PARAMS = ("uid", "_paramMap", "_defaultParamMap", "_params")


def _lazy_attribute(obj, name):
    """The attribute `name` of `obj` that its attributes do not have yet, of PySpark's attributes
    that its constructor, fit or load sets, which the `__getattribute__` of the native class gives
    of a lookup that does not find it (it raises the AttributeError of another name), made in the
    object's attributes, where PySpark's sets them: its uid and its param maps, of its native
    values (see `_materialize`); of a model, its `_java_obj`, the `RemoteModelRef` of its
    server-side object, the one of the `_remote_model_obj` of its training summary and of its
    copies, as PySpark's fit and copies share it (None of a model of none), which its `_call_java`
    reads, made with the model (see `_ml_make_java_obj`); of another Java wrapper of
    PySpark's, its `_java_obj`, the name of its JVM class; and of a model of a training summary,
    its `_summary` (see `_training_summary`). The uid of an object of no param maps yet is the
    native object's, an attribute once the maps are made. An assignment or a deletion of the uid
    or of `_java_obj` is of the attributes, of CPython's own `__setattr__`, which the native object
    takes before its members, a copy, a pickle or its `__del__` read it (`apply_attributes` of the
    native classes), as PySpark's model is of its `_java_obj`; a deleted `_java_obj` raises
    PySpark's AttributeError."""
    if name == "uid":
        # The native object's, which the native code that makes the object may set after it makes
        # it, until the param maps are made (see `_materialize`), which make it an attribute too:
        # a read of the uid alone makes no param maps of the native values of the moment.
        return _native._ml_uid(obj)
    if name in _LAZY_PARAMS:
        _materialize(obj)
        try:
            return obj.__dict__[name]
        except KeyError:
            raise _no_attribute(obj, name) from None
    kind = _native._ml_kind(type(obj))
    if name == "_java_obj" and kind == "model":
        _native._ml_make_java_obj(obj)
        try:
            return obj.__dict__["_java_obj"]
        except KeyError:
            # A deleted one, as PySpark's object has none then.
            raise _no_attribute(obj, name) from None
    if name == "_java_obj" and kind == "params":
        jvm_class = _native._ml_jvm_class(type(obj))
        if jvm_class is not None and _is_java_wrapper(type(obj)):
            obj.__dict__["_java_obj"] = jvm_class
            return jvm_class
    if name == "_summary" and kind == "model" and _has_training_summary(type(obj)):
        return _training_summary(obj)
    raise _no_attribute(obj, name)


def _has_java_obj(cls):
    """Whether the objects of the public class `cls` have PySpark's `_java_obj` (see
    `_lazy_attribute`): a model, and another Java wrapper of PySpark's of a JVM class."""
    kind = _native._ml_kind(cls)
    return kind == "model" or (
        kind == "params" and _native._ml_jvm_class(cls) is not None and _is_java_wrapper(cls)
    )


def _dir(self):
    """`dir()` of an object of a public class, as of PySpark's object: of its attributes that it
    makes at their first read (see `_lazy_attribute`) too, its uid, its param maps and its
    `_java_obj`. A training summary is listed once its first read made it (`_summary`), which may
    request the server (as `numClasses` of its class)."""
    names = set(object.__dir__(self))
    names.update(_LAZY_PARAMS)
    if _has_java_obj(type(self)):
        try:
            _lazy_attribute(self, "_java_obj")
            names.add("_java_obj")
        except AttributeError:
            # A deleted one, which PySpark's object has not.
            pass
    return sorted(names)


def _materialize(obj):
    """Make PySpark's param maps of `obj` of its native values, if it has none, and its uid, of the
    native object's, attributes of the object, as of PySpark's, and its `_java_obj` (see
    `_lazy_attribute`), so that the attributes are PySpark's state of the object, which
    `__getstate__` gives and a subclass's own `__getstate__` takes of `__dict__`."""
    attributes = obj.__dict__
    if "_paramMap" in attributes and "_defaultParamMap" in attributes:
        return
    attributes.setdefault("uid", _native._ml_uid(obj))
    if "_java_obj" not in attributes and _has_java_obj(type(obj)):
        try:
            _lazy_attribute(obj, "_java_obj")
        except AttributeError:
            # A deleted one stays deleted.
            pass
    obj._copy_params()
    set_values, defaults = _native._ml_params(obj)
    attributes.setdefault("_paramMap", {getattr(obj, n): v for n, v in set_values.items()})
    attributes.setdefault("_defaultParamMap", {getattr(obj, n): v for n, v in defaults.items()})
    attributes.setdefault("_params", None)


# The native getters of the objects that a model holds of its own (of no param), whose native
# objects its native members read: a PipelineModel's stages, an OneVsRestModel's models, and a
# validator model's best model and sub-models.
_HELD = ("stages", "models", "bestModel", "subModels")


def _sync(obj, synced=None):
    """Make the native object's params those of PySpark's param maps of `obj`, before a native
    member reads them (a fit's request sends them), and those of each object of native params that
    `obj` holds, whose native object the native members of `obj` read: of the values of its params
    (an OneVsRest's classifier, a Pipeline's stages, a validator's estimator and evaluator and the
    values of its grids) and of its own objects (`_HELD`). `synced` has the ids of the objects
    synced of this sync."""
    if synced is None:
        synced = set()
    if id(obj) in synced:
        return
    synced.add(id(obj))
    set_values = {param.name: value for param, value in obj._paramMap.items()}
    defaults = {param.name: value for param, value in obj._defaultParamMap.items()}
    held = []
    for name in _HELD:
        getter = inspect.getattr_static(type(obj), name, None)
        if type(getter).__name__ == "getset_descriptor":
            held.append(getter.__get__(obj, type(obj)))
    for value in [*set_values.values(), *defaults.values(), *held]:
        _sync_values(value, synced)
    _native._ml_sync(obj, set_values, defaults)


def _sync_values(value, synced):
    """Sync (`_sync`) each object of native params of `value`, of its lists, tuples and dicts (a
    grid's param maps) too."""
    if isinstance(value, dict):
        for item in value.values():
            _sync_values(item, synced)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _sync_values(item, synced)
    elif not isinstance(value, type) and _has_params(type(value)):
        _sync(value, synced)


def _has_params(cls):
    """Whether `cls` is a native class of params (an estimator, a transformer, a model, an
    evaluator, a Pipeline, a validator), or a subclass of one, whose native object holds params of
    its own (see `_ml_kind` of the native module)."""
    return _native._ml_kind(cls) in ("params", "model")


def _native_writer(obj):
    """The writer of the native class of `obj`, of the params of its maps (and of the objects that
    it holds) as of now: a writer's `save` saves its `_instance` of it at the save, as PySpark's
    `RemoteMLWriter.save` reads the params of its instance then. None of an object of no native
    class, which PySpark's own `RemoteMLWriter.saveInstance` saves."""
    for cls in type(obj).__mro__:
        member = vars(cls).get("write")
        if type(member).__name__ == "method_descriptor":
            _sync(obj)
            return member(obj)
    return None


def _native_reader(clazz):
    """The reader of the native class of the class `clazz` (a native class, its public class, or a
    Python subclass of them), as its native `read()` of `clazz` gives it: a reader's `load` loads
    of it at the load, as PySpark's `RemoteMLReader.load` reads its `_clazz` then. None of another
    class or object, which PySpark's own `RemoteMLReader.loadInstance` loads."""
    if isinstance(clazz, type):
        for cls in clazz.__mro__:
            member = vars(cls).get("read")
            if type(member).__name__ == "classmethod_descriptor":
                return member.__get__(None, clazz)()
    return None


def _warn_as(function, message, category):
    """Warn `message` of `category` as PySpark's `function` warns it of its `warnings.warn(...)`
    call: of the location that `warnings.warn` gives it, the line of the call in the file of the
    function, and of the module of the function and its registry of the warnings that it showed.
    So the filters of the module (`filterwarnings(..., module="pyspark.ml.clustering")`) and the
    actions of a location (`"default"`, `"module"`) take it as they take PySpark's."""
    code = function.__code__
    starts = dict(dis.findlinestarts(code))
    line = code.co_firstlineno
    for instruction in dis.get_instructions(code):
        # The call of `warnings.warn` starts of the load of `warnings`, of the line that Python
        # attributes the call to.
        line = starts.get(instruction.offset) or line
        if instruction.opname == "LOAD_GLOBAL" and instruction.argval == "warnings":
            break
    else:
        raise ValueError(f"{function.__qualname__} calls no warnings.warn")
    module = sys.modules[function.__module__]
    registry = vars(module).setdefault("__warningregistry__", {})
    warnings.warn_explicit(message, category, code.co_filename, line, module.__name__, registry)


def _log_as(function, logger_name, message):
    """Log `message` at WARNING as PySpark's `function` logs it of its `<logger_name>.warning(...)`
    call, made by `function` or by a function nested in it (the `wrapped` of a decorator): by the
    `warning` of the logger that the global `logger_name` of the module of `function` is when it is
    called, as PySpark's call looks it up, from a frame at the location of that call, the line of
    the call in the file of the function, in the function that makes it. So the `pathname`,
    `lineno` and `funcName` of the record are those of PySpark's, and so is a traceback of an
    exception of the logging (of a filter that raises)."""
    _logging_call(function, logger_name)(function.__globals__[logger_name], message)


@functools.cache
def _logging_call(function, logger_name):
    """The function that calls `logger.warning(message)` from the line of the
    `<logger_name>.warning(...)` call of `function`, or of a function nested in it, in its file (see
    `_log_as`)."""
    codes = [function.__code__]
    while codes:
        code = codes.pop(0)
        starts = dict(dis.findlinestarts(code))
        line = code.co_firstlineno
        for instruction in dis.get_instructions(code):
            line = starts.get(instruction.offset) or line
            if instruction.opname == "LOAD_GLOBAL" and instruction.argval == logger_name:
                break
        else:
            codes.extend(const for const in code.co_consts if isinstance(const, types.CodeType))
            continue

        def call(logger, message):
            logger.warning(message)

        # The call is on the line after the `def` of `call`, so its frame is at `line`.
        names = {
            "co_filename": code.co_filename,
            "co_firstlineno": line - 1,
            "co_name": code.co_name,
        }
        if sys.version_info >= (3, 11):
            names["co_qualname"] = code.co_qualname
        return types.FunctionType(call.__code__.replace(**names), {})
    raise ValueError(f"{function.__qualname__} calls no {logger_name}.warning")


def _binding_of(function):
    """The signature that Python binds a call of `function` of: of the function itself, which is
    the wrapper of a decorator (`try_remote_attribute_relation` takes any arguments, and
    `try_remote_intercept` any positional ones), rather than the function that it wraps, whose
    signature `inspect.signature` gives; else of the function that a wrapper of no signature of its
    own wraps (`functools.cache`'s, which passes it its arguments); else of any arguments."""
    for follow_wrapped in (False, True):
        try:
            return inspect.signature(function, follow_wrapped=follow_wrapped)
        except (TypeError, ValueError):
            continue
    Parameter = inspect.Parameter
    return inspect.Signature(
        [
            Parameter("self", Parameter.POSITIONAL_ONLY),
            Parameter("args", Parameter.VAR_POSITIONAL),
            Parameter("kwargs", Parameter.VAR_KEYWORD),
        ]
    )


class _NativeMember:
    """The member `name` of the native class `native`, which a function of its public class runs
    (see `_synced`): a reference to it, which `pickle` and cloudpickle take of the names of the
    native class (of the native module) and of the member, where the member itself, a descriptor
    of the native class, pickles of that class, which pickles of its attributes (they do not
    pickle). So cloudpickle pickles a function of a public class that runs a native member (a
    method, a property, the class method `read`) by value, of this reference, as it pickles a
    closure or a `functools.partial` of it (the function of a UDF, of `foreachBatch`), where it
    pickles PySpark's by reference. A static method is of its function."""

    __slots__ = ("member", "name", "native", "static")

    def __init__(self, native, name):
        self.native, self.name = native, name
        member = inspect.getattr_static(native, name)
        self.static = isinstance(member, staticmethod)
        self.member = member.__func__ if self.static else member

    def __call__(self, *args, **kwargs):
        return self.member(*args, **kwargs)

    def get(self, obj, cls):
        """The member of `obj` of the class `cls`, as `__get__` of the descriptor gives it."""
        return self.member.__get__(obj, cls)

    def of(self, obj):
        """The member of `obj`, of its class, as an attribute of `obj` gives it (a getter's value)."""
        return self.member.__get__(obj, type(obj))

    def __reduce__(self):
        return (_native_member, (self.native.__name__, self.name))


def _native_member(native, name):
    """The `_NativeMember` of the member `name` of the native class `native` of the native module,
    of which a reference unpickles."""
    return _NativeMember(getattr(_native, native), name)


def _ensure_readwrite_classes():
    """Initialize the shared native reader/writer adapters before a standard-import operation.

    Their registration currently lives in the native package initializer. Import it lazily,
    after the standard ML modules have loaded, to avoid a classification import cycle.
    """
    importlib.import_module("pyspark.ml.connect.readwrite")


def _restore_readwrite(factory):
    """Rebuild a pickled native reader/writer after registering its public Python class."""
    _ensure_readwrite_classes()
    return factory()


def _synced(member, name, official, sync=True, prepare=None):
    """The member `name` of the public class of PySpark's class `official`, which runs the native
    method `member` of the native object's params synced first: a function of the parameters of
    PySpark's member of the name, as Python binds a call of it (`_binding_of`), which passes them
    to `member` as they are given, of the qualified name, module, docstring and signature of
    PySpark's member (of the class of PySpark's that has it, `JavaMLWritable.write`). So a call that
    does not bind raises PySpark's message (`JavaMLWritable.write() takes 1 positional argument but
    2 were given`). Of a name that PySpark's class does not have, it is the native member's.
    Without `sync`, it syncs no params, as of an object of none (a summary); of `prepare`, it runs
    `prepare` of the object in place of the sync (`_materialize`). Of a class method or a static
    method of PySpark's (see `_class_member`), it is of PySpark's function of it."""
    _, pysparks = _member_of(official, name)
    if isinstance(pysparks, (classmethod, staticmethod)):
        pysparks = pysparks.__func__
    if not callable(pysparks):
        # The native member's own, of its signature and docstring.
        pysparks = member.member if isinstance(member, _NativeMember) else member
    binding = _binding_of(pysparks)
    parameters = list(binding.parameters.values())
    arguments = []
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            arguments.append("*" + parameter.name)
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            arguments.append("**" + parameter.name)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            arguments.append(f"{parameter.name}={parameter.name}")
        else:
            arguments.append(parameter.name)
    # The object, of the first parameter, as every member of PySpark's has it (`self`).
    body = f"return __member({', '.join(arguments)})"
    if sync or prepare is not None:
        # `_sync` of the scope, or `prepare` in its place.
        body = f"_sync({parameters[0].name})\n{body}"
    qualname = getattr(pysparks, "__qualname__", None) or f"{official.__name__}.{name}"
    if name in ("write", "read"):
        body = f"_ensure_readwrite_classes()\n{body}"
    scope = {
        "_sync": prepare or _sync,
        "__member": member,
        "_ensure_readwrite_classes": _ensure_readwrite_classes,
    }
    return _function(pysparks, name, qualname, body, scope, binding=binding)


def _class_member(own, name, official):
    """The class method (`read`, `from_labels`) or static method (`loadDefaultStopWords`) `name` of
    the public class of PySpark's class `official`, over the native one `own` (the `_NativeMember`
    of a `classmethod_descriptor` or a `staticmethod`), as `_synced` makes a method of it, of no
    params to sync: of PySpark's qualified name, module, docstring and signature
    (`JavaMLReadable.read`, of the wrapper of `try_remote_read`), so that Python binds a call of it
    as it binds PySpark's function and raises PySpark's message of a call that does not bind
    (`JavaMLReadable.read() takes 1 positional argument but 2 were given`, of its `cls` too). A
    class method runs the native one of the class that it is called of (a subclass)."""
    if own.static:
        return staticmethod(_synced(own, name, official, sync=False))

    def of_class(cls, *args, **kwargs):
        return own.get(None, cls)(*args, **kwargs)

    return classmethod(_synced(of_class, name, official, sync=False))


def _synced_property(descriptor, official):
    """A property of the native getter `descriptor` (its `_NativeMember`), of the native object's
    params synced first, so that PySpark's `params` walk does not read it (it reads no property),
    as of PySpark's property `official` (see `_property_of`)."""
    return _property_of(descriptor, official, sync=True)


def _property_of(descriptor, official, sync=False):
    """A property of the native getter `descriptor` (its `_NativeMember`), of the native object's
    params synced first of `sync`, of the docstring of PySpark's property `official` and of a
    getter of PySpark's getter (`LinearSVCModel.coefficients`, `JavaPredictionModel.numFeatures`):
    of its name, qualified name, module, docstring, signature and annotations (see `_function`)."""
    getter = official.fget
    first = next(iter(inspect.signature(getter).parameters))
    body = f"return __member.of({first})"
    if sync:
        body = f"_sync({first})\n{body}"
    scope = {"_sync": _sync, "__member": descriptor}
    function = _function(getter, getter.__name__, getter.__qualname__, body, scope)
    return property(function, doc=official.__doc__)


def _function(official, name, qualname, body, scope, binding=None):
    """A function `name` of the parameters of the function `official` (their names, kinds and
    defaults), or of the signature `binding` if given, of the qualified name `qualname`, whose body
    is the lines of `body` over the names of `scope`: Python binds a call of it as it binds
    `official`, and raises PySpark's messages of a call that does not bind
    (`LinearSVC.__init__() got an unexpected keyword argument 'foo'`). Its module, docstring,
    signature and annotations are those of `official` (of this module and of `binding`, of a native
    `official` of none), and its globals those that `typing.get_type_hints` reads of PySpark's
    function (of the function that a decorator's wrapper wraps), which resolve the names of its
    annotations as of PySpark's (`"LinearSVCModel"`): the names of `scope` are of a closure."""
    try:
        signature = inspect.signature(official)
    except (TypeError, ValueError):
        if binding is None:
            raise
        signature = binding
    parts, defaults, star, positional_only = [], {}, False, False
    for index, parameter in enumerate((binding or signature).parameters.values()):
        kind = parameter.kind
        if positional_only and kind is not inspect.Parameter.POSITIONAL_ONLY:
            parts.append("/")
            positional_only = False
        text = parameter.name
        if parameter.default is not inspect.Parameter.empty:
            defaults[f"__default_{index}"] = parameter.default
            text += f"=__default_{index}"
        if kind is inspect.Parameter.POSITIONAL_ONLY:
            positional_only = True
        elif kind is inspect.Parameter.VAR_POSITIONAL:
            text, star = "*" + parameter.name, True
        elif kind is inspect.Parameter.KEYWORD_ONLY and not star:
            parts.append("*")
            star = True
        elif kind is inspect.Parameter.VAR_KEYWORD:
            text = "**" + parameter.name
        parts.append(text)
    if positional_only:
        parts.append("/")
    # The function of a closure of the names of `scope`, whose defaults are of the names of the
    # closure's parameters, at the `def`.
    free = [*scope, *defaults]
    lines = "".join(f"        {line}\n" for line in body.split("\n"))
    source = (
        f"def __closure({', '.join(free)}):\n"
        f"    def {name}({', '.join(parts)}):\n{lines}"
        f"    return {name}\n"
    )
    namespace = {"__name__": __name__}
    exec(source, namespace)  # noqa: S102
    function = namespace["__closure"](*scope.values(), *defaults.values())
    if sys.version_info >= (3, 11):
        function.__code__ = function.__code__.replace(co_qualname=qualname)
    own_globals = getattr(_unwrapped(official), "__globals__", None)
    if own_globals is not None:
        # Of no global name of its own: the code reads its parameters and the closure's names.
        kwdefaults = function.__kwdefaults__
        function = types.FunctionType(
            function.__code__, own_globals, name, function.__defaults__, function.__closure__
        )
        function.__kwdefaults__ = kwdefaults
    function.__qualname__ = qualname
    function.__module__ = getattr(official, "__module__", None) or __name__
    function.__doc__ = getattr(official, "__doc__", None)
    function.__annotations__ = dict(getattr(official, "__annotations__", None) or {})
    function.__signature__ = signature
    return function


def _unwrapped(function):
    """The function that the wrappers of the decorators of `function` wrap (`__wrapped__`), as
    `typing.get_type_hints` reads the globals of its annotations of it; `function` of none."""
    try:
        return inspect.unwrap(function)
    except ValueError:
        return function


def _member_of(official, name):
    """PySpark's member `name` of the class `official` and the class of its MRO that has it."""
    owner = next((k for k in official.__mro__ if name in vars(k)), None)
    return (owner, vars(owner)[name]) if owner is not None else (None, None)


def _dispatchers():
    """PySpark's members that run other members of the object, by the class of PySpark's that has
    them: a public class takes them as they are, of PySpark's signatures, messages and docstrings."""
    from pyspark.ml.base import Estimator, Transformer
    from pyspark.ml.evaluation import Evaluator
    from pyspark.ml.util import MLReadable, MLWritable

    return {
        (Estimator, "fit"),
        (Estimator, "fitMultiple"),
        (Transformer, "transform"),
        (Evaluator, "evaluate"),
        (MLWritable, "save"),
        (MLReadable, "load"),
    }


# The member of the object that each dispatcher runs, of which the native class's own member (of
# no param map) is the public class's: PySpark's `_fit` (`_transform`, `_evaluate`) of the native
# `fit` (`transform`, `evaluate`), of the native object's params synced first (see `_synced`).
_PRIMITIVES = {"fit": "_fit", "transform": "_transform", "evaluate": "_evaluate"}


# The native members that a public class keeps of the native class whatever PySpark's are: the
# persistence of the native writers and readers.
_NATIVE_ALWAYS = frozenset({"write", "read"})


def _is_java_wrapper(official):
    """Whether PySpark's class `official` is a Java wrapper (`JavaWrapper`), whose objects have the
    `_java_obj` of `JavaWrapper.__init__`."""
    from pyspark.ml.wrapper import JavaWrapper

    return issubclass(official, JavaWrapper)


def _del_of(official):
    """The `__del__` of the public class of PySpark's Java wrapper `official`, as PySpark's
    `JavaWrapper.__del__` (of `try_remote_del`), which releases a reference of the `RemoteModelRef`
    of the object's `_java_obj`: of a model of a `RemoteModelRef` that its constructor or an
    assignment of its `_java_obj` gave, the release of the native object (see `_ml_finalize` of the
    native module), whose error Python reports as unraisable in this `__del__`, as in PySpark's,
    and the AttributeError of a deleted `_java_obj`; of a model of a fit or a load, of its copies
    and of its training summary, the release of their hold of the model's server-side object, the
    last of which deletes it, as the last release of PySpark's `RemoteModelRef` does (see `Holders`
    of the native module), where the training summary's `_remote_model_obj`, which this client does
    not count (see `remote_model_ref` of the native module), is not released. A function of the name, qualified name, module, docstring, signature and annotations
    of PySpark's (see `_function`)."""
    _, pysparks = _member_of(official, "__del__")
    return _function(
        pysparks,
        "__del__",
        pysparks.__qualname__,
        "_finalize(self)",
        {"_finalize": _native._ml_finalize},
    )


def _is_summary(official):
    """Whether PySpark's class `official` is a summary: a Java wrapper of no params, whose
    constructor is `JavaWrapper.__init__(java_obj=None)`."""
    from pyspark.ml.wrapper import JavaWrapper

    return _member_of(official, "__init__")[1] is JavaWrapper.__init__


def _constructors(native, official, holder):
    """The `__new__` and `__init__` of the public class of `native` and PySpark's class
    `official`; `holder` has the public class, once made."""
    _, init = _member_of(official, "__init__")
    inner = getattr(init, "__wrapped__", init)
    signature = inspect.signature(inner)
    keyword = inner is not init and all(
        p.kind is inspect.Parameter.KEYWORD_ONLY for p in list(signature.parameters.values())[1:]
    )
    qualname = inner.__qualname__
    checker = _function(inner, "__init__", qualname, "pass", {})
    if keyword:
        # PySpark's own `__init__`: its `Params.__init__` makes the param maps and the Params of
        # the object, `Identifiable.__init__` its uid (of the class's name), its `_setDefault` the
        # defaults in its order, and its `setParams` (of `keyword_only`) the params of the keywords.
        __init__ = init

        def check(*args, **kwargs):
            if args:
                raise TypeError("Method __init__ forces keyword arguments.")
            checker(None, **kwargs)

    elif _is_summary(official) or "java_model" in signature.parameters:
        # PySpark's own `JavaWrapper.__init__` of a summary, which keeps its `java_obj` in the
        # object's attributes (`_java_obj`), which the native members read, as PySpark's do: so a
        # summary of a subclass reads the `java_obj` of its `super().__init__`, and a copy or a
        # pickle keeps it with the other attributes. And PySpark's own `JavaModel.__init__`, of no
        # server-side object or of the object of a `RemoteModelRef`, which the native constructor
        # takes (and raises the AssertionError of PySpark's of another `java_model`): its
        # `Params.__init__` makes the param maps and the Params of the object,
        # `Identifiable.__init__` its uid (of the class's name), `JavaWrapper.__init__` its
        # `_java_obj` (see `_lazy_attribute`), and the `_setDefault` of its classes the
        # defaults in their order.
        __init__ = init

        def check(*args, **kwargs):
            checker(None, *args, **kwargs)

    else:
        # The native constructor takes the objects of the arguments (a PipelineModel's `stages`, a
        # OneVsRestModel's `models`, a validator model's `bestModel`), which PySpark's `__init__`
        # sets as attributes of the instance that are the native class's own; the `__init__` makes
        # the param maps and the uid of the object (of PySpark's `Params.__init__`). Of a subclass
        # of an `__init__` of its own, whose `__new__` made the native object of none, the
        # `__init__` that it calls (`super().__init__(stages)`) is PySpark's, which sets them of
        # the native class's setters (`self.stages = stages`).
        parameters = list(signature.parameters.values())
        cell = {}
        __init__ = _function(
            inner,
            "__init__",
            qualname,
            f"_composite_init(_cell['init'], _official, {_forwarded(parameters)})",
            {"_composite_init": _composite_init, "_cell": cell, "_official": inner},
        )
        cell["init"] = __init__

        def check(*args, **kwargs):
            # Of no arguments, an object of none, as `object.__new__` makes PySpark's of
            # `copyreg.__newobj__`, whose state gives its attributes (`stages`, `models`,
            # `bestModel`, ...); the `__init__` of a call of none raises PySpark's TypeError.
            if args or kwargs:
                checker(None, *args, **kwargs)

    try:
        native_parameters = list(inspect.signature(native).parameters.values())
    except (TypeError, ValueError):
        no_arguments = False
    else:
        if native_parameters and native_parameters[0].kind is inspect.Parameter.POSITIONAL_ONLY:
            # The class of the native `__new__` (`(cls, /, stages=None)`), of no argument of a call.
            native_parameters = native_parameters[1:]
        no_arguments = all(
            p.default is not inspect.Parameter.empty
            or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in native_parameters
        )

    def __new__(cls, *args, **kwargs):
        if cls.__init__ is __init__:
            check(*args, **kwargs)
            obj = native.__new__(cls) if keyword else native.__new__(cls, *args, **kwargs)
        elif keyword or no_arguments:
            # The subclass's own `__init__` takes the arguments, as `object.__new__` leaves them.
            obj = native.__new__(cls)
        else:
            obj = native.__new__(cls, *args, **kwargs)
        return obj

    del holder
    return __new__, __init__


def _forwarded(parameters):
    """The arguments of a call of the parameters `parameters` of a function, which passes each
    parameter on as it is bound: of its position, `*args`, `name=name` and `**kwargs`."""
    parts = []
    for parameter in parameters:
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            parts.append("*" + parameter.name)
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            parts.append("**" + parameter.name)
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            parts.append(f"{parameter.name}={parameter.name}")
        else:
            parts.append(parameter.name)
    return ", ".join(parts)


def _composite_init(init, official, obj, *args, **kwargs):
    """The `__init__` `init` of the public class of a composite (a PipelineModel, an
    OneVsRestModel, a validator's model) of `obj`: of an object of a class of that `__init__`, whose
    `__new__` made the native object of the arguments, its param maps and its uid
    (`_materialize`); of a subclass of an `__init__` of its own, whose `__new__` made the native
    object of none (as `object.__new__` leaves the arguments to the `__init__`), PySpark's
    `__init__` `official`, which sets the objects of the arguments (`self.stages = stages`, of the
    native class's setters), the param maps and the uid."""
    if type(obj).__init__ is init:
        _materialize(obj)
    else:
        official(obj, *args, **kwargs)


def _is_remote_model_ref(value):
    """Whether `value` is a `RemoteModelRef` of `pyspark.ml.util`."""
    from pyspark.ml.util import RemoteModelRef

    return isinstance(value, RemoteModelRef)


def _no_attribute(obj, name):
    """Python's AttributeError of the attribute `name` that `obj` does not have, of its `name`
    and `obj` (which Python 3.10 and up read), as a missing attribute of PySpark's raises it."""
    error = AttributeError(f"'{type(obj).__name__}' object has no attribute '{name}'")
    error.name, error.obj = name, obj
    return error


def _training_summary(obj):
    """PySpark's `_summary` of a native model, which PySpark's fit sets: at its first read, the
    model's training summary (`summary`), in the object's attributes, of a model that has one
    (the native `hasSummary`); one of none (a load) has none, as PySpark's."""
    attributes = obj.__dict__
    if "_summary" in attributes:
        return attributes["_summary"]
    if not _native_member_value(obj, "hasSummary"):
        raise _no_attribute(obj, "_summary")
    summary = _native_member_value(obj, "summary")
    # The `summary()` method of a LinearSVCModel, an MLP's and an FM classifier's.
    summary = summary() if callable(summary) else summary
    attributes["_summary"] = summary
    return summary


def _native_member_value(obj, name):
    """The value of the member `name` of the native class of `obj` (its getter's, or its method),
    of the object, as the native class reads it, not of the public class (whose PySpark's own
    `hasSummary` and `summary` read `_summary`)."""
    for cls in type(obj).__mro__:
        member = vars(cls).get(name)
        if type(member).__name__ in ("getset_descriptor", "method_descriptor"):
            return member.__get__(obj, type(obj))
    raise _no_attribute(obj, name)


def _has_training_summary(official):
    """Whether PySpark's class `official` has a training summary (`HasTrainingSummary`)."""
    from pyspark.ml.util import HasTrainingSummary

    return issubclass(official, HasTrainingSummary)


def _public_class(native, official, parents, public_module=None, eager_types=()):
    """The public class of `native`, of PySpark's class `official` and of the public classes
    `parents` of its bases of the module (see `publish`)."""
    bases = (native,) + parents + (official,)
    # PySpark's metaclass (`ABCMeta` of the Params, `type` of the summaries).
    meta = type(official)
    if _is_readwrite(official):
        public = meta(native.__name__, bases, _readwrite_api(native, official))
        _native._register_public_class(native, public)
        return public
    holder = [None]
    __new__, __init__ = _constructors(native, official, holder)
    namespace = {
        "__module__": public_module or native.__module__,
        "__qualname__": native.__qualname__,
        "__doc__": official.__doc__ or native.__doc__,
        "__new__": __new__,
        "__init__": __init__,
    }
    if _is_java_wrapper(official):
        # The object of the server of PySpark's Java wrappers, which the native class holds
        # itself, so that `__del__` releases none. A model and a summary have PySpark's value of
        # it, which its `_call_java` reads. PySpark's classes of no Java wrapper (a Pipeline, a
        # OneVsRest, a validator and their models) have none, and no `__del__`.
        namespace["__del__"] = _del_of(official)
    _sign_constructor(__new__, official)
    if _has_params(native):
        _params_api(native, official, namespace)
        if native in eager_types:
            _eager_params_api(native, official, namespace, eager_types)
    elif _is_summary(official):
        _summary_api(native, official, namespace)
    public = meta(native.__name__, bases, namespace)
    holder[0] = public
    _native._register_public_class(native, public)
    return public


def _sign_constructor(__new__, official):
    """Give `__new__` of a public class PySpark's signature of the class `official`, of its
    constructor, of a first parameter of the class: `inspect.signature` of the class gives it of
    its own `__new__` (of a Python subclass, of the subclass's own `__init__`), as of PySpark's
    `__init__`, and the class has no `__signature__`, as PySpark's has none."""
    try:
        signature = inspect.signature(official)
    except (TypeError, ValueError):
        return
    first = inspect.Parameter("cls", inspect.Parameter.POSITIONAL_ONLY)
    __new__.__signature__ = signature.replace(parameters=[first, *signature.parameters.values()])


def _params_api(native, official, namespace):
    """The members of the public class of the native class `native` of params and PySpark's class
    `official`, in `namespace`: PySpark's own code of its param maps (the Params API, the getters
    and setters, `fit`, `copy`, `save`, `load`), and the native class's members of PySpark's
    server-side object (its fit, transform and evaluate, the model's members, the persistence), of
    the native object's params synced first."""
    # PySpark's attributes of the object that it makes at their first read, which `dir()` lists
    # (see `_lazy_attribute`).
    namespace["__dir__"] = _dir
    dispatchers = _dispatchers()
    for name in dir(native):
        if name.startswith("__") or name.startswith("_native") or name in _NATIVE_ALWAYS:
            continue
        owner, member = _member_of(official, name)
        # `copy` is PySpark's (`Params.copy`, `JavaParams.copy`) of `copy.copy`, which reads
        # `_java_obj` of a server-side object that a native object does not have (it is None).
        pysparks = name == "copy" or (owner, name) in dispatchers
        if member is not None and (pysparks or _pysparks_own(member)):
            # PySpark's own code over the object's other members (the Params API, the getters
            # and setters, `fit`, `copy`, `save`, `load`), of its param maps.
            namespace[name] = member
            continue
        kind = type(inspect.getattr_static(native, name)).__name__
        if kind == "getset_descriptor" and isinstance(member, property):
            # A property, as PySpark's (its `params` walk reads none), of PySpark's docstring; a
            # getter of PySpark's instance attribute (a PipelineModel's `stages`) or cached member
            # (`trees`) stays the native one, which the walk reads, as PySpark's reads them.
            namespace[name] = _synced_property(_NativeMember(native, name), member)
        elif kind == "method_descriptor":
            namespace[name] = _synced(_NativeMember(native, name), name, official)
        elif kind in ("classmethod_descriptor", "staticmethod") and isinstance(
            member, (classmethod, staticmethod)
        ):
            # A class or static method, as PySpark's (`from_labels`, `loadDefaultStopWords`).
            namespace[name] = _class_member(_NativeMember(native, name), name, official)
    for name, primitive in _PRIMITIVES.items():
        if hasattr(native, name) and (_member_of(official, name)[0], name) in dispatchers:
            namespace[primitive] = _synced(_NativeMember(native, name), primitive, official)
    if hasattr(native, "__getstate__"):
        # PySpark's state, the attributes of the object, of its param maps made first, which
        # `copy.copy`, `copy.deepcopy` and `pickle` take (see `_materialize`).
        namespace["__getstate__"] = _synced(
            _NativeMember(native, "__getstate__"), "__getstate__", official, prepare=_materialize
        )
    if hasattr(native, "write"):
        # The native writer, of the params of the object (and of the objects that it holds) as of
        # the call: it takes them at its making.
        namespace["write"] = _synced(_NativeMember(native, "write"), "write", official)
    if hasattr(native, "read") and isinstance(_member_of(official, "read")[1], classmethod):
        # The native reader, of PySpark's class method (`JavaMLReadable.read`, `Pipeline.read`).
        namespace["read"] = _class_member(_NativeMember(native, "read"), "read", official)
    _text_api(native, official, namespace)



def _eager_params_api(native, official, namespace, eager_types):
    """Sync opted-in classes after Python Params mutations, including copies and restoration.

    Keep Python's method signatures, conversions and return values. The operation-time sync
    remains necessary for direct dictionary writes and overrides that bypass these methods.
    """
    def sync(instance):
        if isinstance(instance, eager_types):
            attributes = vars(instance)
            # Constructors and custom pickle states can have incomplete Python Params state.
            if "_paramMap" in attributes and "_defaultParamMap" in attributes:
                _sync(instance)

    for name in ("set", "_set", "_setDefault", "clear", "_resetUid", "_copyValues", "copy",
                 "__setstate__"):
        if name == "__setstate__":
            member = _NativeMember(native, name)
            reference = member.member
        else:
            reference = _member_of(official, name)[1]
            member = namespace.get(name, reference)
        binding = _binding_of(reference)
        parameters = list(binding.parameters.values())
        call = f"__member({_forwarded(parameters)})"
        if name == "copy":
            body = f"result = {call}\n__sync(result)\nreturn result"
        else:
            target = "to" if name == "_copyValues" else parameters[0].name
            # A multi-parameter mutation can update earlier values before conversion fails.
            body = f"try:\n    return {call}\nfinally:\n    __sync({target})"
        namespace[name] = _function(
            reference, name, reference.__qualname__, body,
            {"__member": member, "__sync": sync}, binding=binding,
        )


def _text_api(native, official, namespace):
    """The `__repr__` and `__str__` of the public class of the native class `native` and PySpark's
    class `official`, in `namespace`: the native class's own (the uid, `Identifiable.__repr__`, or
    the server's description of a model, `JavaModel.__repr__`), of PySpark's function of the name
    (see `_synced`), of no params to sync, as PySpark's reads none."""
    for name in ("__repr__", "__str__"):
        own = inspect.getattr_static(native, name)
        if own is inspect.getattr_static(object, name):
            continue
        if isinstance(_member_of(official, name)[1], types.FunctionType):
            namespace[name] = _synced(_NativeMember(native, name), name, official, sync=False)


def _is_readwrite(official):
    """Whether PySpark's class `official` is a writer or a reader (`RemoteMLWriter`,
    `RemoteMLReader`): a plain Python class of `BaseReadWrite`, whose `__init__` sets its state in
    the object's attributes."""
    from pyspark.ml.util import BaseReadWrite

    return issubclass(official, BaseReadWrite)


def _readwrite_api(native, official):
    """The namespace of the public class of the native writer or reader `native` and PySpark's
    class `official` (see `_is_readwrite`), a plain Python class as PySpark's: of PySpark's
    docstring (None), and of the native `save` (`load`) as PySpark's `RemoteMLWriter.save`
    (`RemoteMLReader.load`) names it, documents it and binds a call of it (see `_synced`), of no
    params to sync, which reads the object's attributes at the save (load). The rest is PySpark's
    own code, which the class takes of PySpark's class: its `__init__`, which sets the attributes
    (`RemoteMLWriter.__init__(instance)`) after the native `__new__`, which leaves the arguments to
    it as `object.__new__` does, so that a copy and a pickle make an object of none, as of
    PySpark's class; `session`, `sparkSession` and `sc` (of `BaseReadWrite`, and of
    `RemoteMLWriter`); `overwrite` and `option` (of `MLWriter`). It has no `__del__` and no
    `_java_obj`, as PySpark's has none."""
    namespace = {
        "__module__": native.__module__,
        "__qualname__": native.__qualname__,
        "__doc__": official.__doc__,
    }
    for name in ("save", "load"):
        if hasattr(native, name):
            namespace[name] = _synced(_NativeMember(native, name), name, official, sync=False)

    def __reduce__(self):
        reduced = native.__reduce__(self)
        if reduced[0] in (_native._new_remote_ml_writer, _native._new_remote_ml_reader):
            # A fresh interpreter may not have imported any native ML namespace yet.
            return (_restore_readwrite, (reduced[0],), *reduced[2:])
        return reduced

    namespace["__reduce__"] = __reduce__
    return namespace


def _summary_api(native, official, namespace):
    """The members of the public class of the native summary class `native` and PySpark's summary
    class `official`, in `namespace`: the native class's members of PySpark's server-side object, of
    PySpark's docstrings (and signatures), as of a class of params (see `_params_api`), of no params
    to sync: a property of a native getter, as PySpark's are properties, and a method of a native
    method."""
    for name in dir(native):
        if name.startswith("__"):
            continue
        _, member = _member_of(official, name)
        if member is None:
            continue
        kind = type(inspect.getattr_static(native, name)).__name__
        if kind == "getset_descriptor" and isinstance(member, property):
            namespace[name] = _property_of(_NativeMember(native, name), member)
        elif kind == "method_descriptor":
            namespace[name] = _method_of(_NativeMember(native, name), name, official)
    _text_api(native, official, namespace)


def _method_of(member, name, official):
    """The native method `member` (`name`, its `_NativeMember`) of PySpark's class `official`, as
    PySpark names it, documents it and binds a call of it (see `_synced`), of no params to sync."""
    return _synced(member, name, official, sync=False)


def publish(namespace, official_module, *, public_module=None, eager_params=()):
    """Replace each native class of the module of the globals `namespace` by its public class, of
    PySpark's class of the same name of `official_module`. A reference namespace also works,
    so a public adapter can retain upstream definitions before publishing native classes under
    `public_module` without importing itself recursively. Names in `eager_params` also sync
    after Python parameter mutations; other classes keep operation-time synchronization."""
    official = (
        importlib.import_module(official_module)
        if isinstance(official_module, str)
        else official_module
    )
    natives = [
        (name, value)
        for name, value in list(namespace.items())
        if isinstance(value, type) and getattr(_native, name, None) is value
    ]
    # A native base class (LDA's `LDAModel`) before its subclasses.
    natives.sort(key=lambda item: len(item[1].__mro__))
    eager_types = tuple(native for name, native in natives if name in eager_params)
    published = {}
    for name, native in natives:
        official_class = getattr(official, name, None)
        if official_class is native:
            continue
        if official_class is None:
            # A module of pyspark.ml that is not imported in full yet (an import cycle) lacks it.
            raise ImportError(f"{official_module} has no {name} of the native class {native}")
        # The public class of the native base class of `native` (LDA's `LDAModel`, a training
        # summary's summary), and those of PySpark's bases of the same module whose native classes
        # are not bases of `native` (the LogisticRegressionTrainingSummary of a
        # BinaryLogisticRegressionTrainingSummary, whose native class extends
        # BinaryLogisticRegressionSummary): the public class is a subclass of each, as PySpark's
        # class is of theirs. The native classes of a summary add no fields to their native base,
        # so that a class of two native bases has their layout.
        parents = [
            published[b.__name__][1] for b in native.__mro__[1:] if b.__name__ in published
        ][:1]
        for base in official_class.__mro__[1:]:
            if base.__name__ in published and published[base.__name__][0] is base:
                ours = published[base.__name__][1]
                if not any(issubclass(parent, ours) for parent in parents):
                    parents.append(ours)
        public = _public_class(
            native, official_class, tuple(parents), public_module, eager_types
        )
        published[name] = (official_class, public)
        namespace[name] = public
    return {name: public for name, (_, public) in published.items()}
