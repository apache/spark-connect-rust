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
- `copy` is PySpark's own; `copy.copy` is the native class's `__copy__`, of the class of the object
  (a subclass) and a copy of the object's attributes, as `copy.copy` copies those of PySpark's.

A public class of a native summary constructs of PySpark's own `JavaWrapper.__init__`, which keeps
the `java_obj` in the object's attributes (`_java_obj`), which the native members read, and has
properties and methods of PySpark's docstrings and signatures over the native members.

A public class of a native class of params keeps them in PySpark's param maps (`_paramMap`,
`_defaultParamMap`), in the object's attributes, and takes PySpark's own Python code over them (the
Params API, the getters and setters, `setParams`, a `keyword_only` constructor) as it is. Before a
native member reads the params (a fit, a transform, an evaluation, a writer, a model's member), the
native object's params are made those of the maps, and so are those of the objects that it holds
(`_sync`): the native object reads them of their native objects. The maps of an object that
PySpark's constructor did not make (a fit's model, a load, a native copy) are made of the native
object's values at their first read.
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


def _del(self):
    """The native class releases its server-side objects itself: PySpark's `JavaWrapper.__del__`
    has none to release, and its imports fail at the interpreter's exit."""


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


class _LazyParamMap:
    """PySpark's `_paramMap` (or `_defaultParamMap`) of an object that PySpark's constructor did not
    make (of a fit, a load, a native copy): at the first read, the dicts of the native object's
    values, keyed by the object's Params (PySpark's `_copy_params`), in the object's attributes,
    where PySpark's constructor, `copy` and `_resetUid` set them."""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        _materialize(obj)
        return obj.__dict__[self.name]


def _materialize(obj):
    """Make PySpark's param maps of `obj` of its native values, if it has none."""
    attributes = obj.__dict__
    if "_paramMap" in attributes and "_defaultParamMap" in attributes:
        return
    obj._copy_params()
    set_values, defaults = obj._native_params()
    attributes.setdefault("_paramMap", {getattr(obj, n): v for n, v in set_values.items()})
    attributes.setdefault("_defaultParamMap", {getattr(obj, n): v for n, v in defaults.items()})


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
    obj._native_sync(set_values, defaults)


def _sync_values(value, synced):
    """Sync (`_sync`) each object of native params of `value`, of its lists, tuples and dicts (a
    grid's param maps) too."""
    if isinstance(value, dict):
        for item in value.values():
            _sync_values(item, synced)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _sync_values(item, synced)
    elif not isinstance(value, type) and hasattr(type(value), "_native_sync"):
        _sync(value, synced)


def _native_writer(obj):
    """The writer of the native class of `obj`, of the params of its maps (and of the objects that
    it holds) as of now: a native writer's `save` reads them of it at the save, as PySpark's
    `RemoteMLWriter.save` reads those of its instance."""
    _sync(obj)
    for cls in type(obj).__mro__:
        member = vars(cls).get("write")
        if type(member).__name__ == "method_descriptor":
            return member(obj)
    raise TypeError(f"{type(obj).__name__} has no native writer")


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


def _synced(member, name, official, sync=True):
    """The member `name` of the public class of PySpark's class `official`, which runs the native
    method `member` of the native object's params synced first: a function of the parameters of
    PySpark's member of the name, as Python binds a call of it (`_binding_of`), which passes them
    to `member` as they are given, of the qualified name, module, docstring and signature of
    PySpark's member (of the class of PySpark's that has it, `JavaMLWritable.write`). So a call that
    does not bind raises PySpark's message (`JavaMLWritable.write() takes 1 positional argument but
    2 were given`). Of a name that PySpark's class does not have, it is the native member's.
    Without `sync`, it syncs no params, as of an object of none (a summary). Of a class method or
    a static method of PySpark's (see `_class_member`), it is of PySpark's function of it."""
    _, pysparks = _member_of(official, name)
    if isinstance(pysparks, (classmethod, staticmethod)):
        pysparks = pysparks.__func__
    if not callable(pysparks):
        pysparks = member
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
    if sync:
        body = f"_sync({parameters[0].name})\n    {body}"
    qualname = getattr(pysparks, "__qualname__", None) or f"{official.__name__}.{name}"
    scope = {"_sync": _sync, "__member": member}
    return _function(pysparks, name, qualname, body, scope, binding=binding)


def _class_member(own, name, official):
    """The class method (`read`, `from_labels`) or static method (`loadDefaultStopWords`) `name` of
    the public class of PySpark's class `official`, over the native one `own` (a
    `classmethod_descriptor` or a `staticmethod`), as `_synced` makes a method of it, of no params to
    sync: of PySpark's qualified name, module, docstring and signature (`JavaMLReadable.read`, of the
    wrapper of `try_remote_read`), so that Python binds a call of it as it binds PySpark's function
    and raises PySpark's message of a call that does not bind (`JavaMLReadable.read() takes 1
    positional argument but 2 were given`, of its `cls` too). A class method runs the native one of
    the class that it is called of (a subclass)."""
    if isinstance(own, staticmethod):
        return staticmethod(_synced(own.__func__, name, official, sync=False))

    def of_class(cls, *args, **kwargs):
        return own.__get__(None, cls)(*args, **kwargs)

    return classmethod(_synced(of_class, name, official, sync=False))


def _synced_property(descriptor, doc):
    """A property of the native getter `descriptor`, of the native object's params synced first, so
    that PySpark's `params` walk does not read it (it reads no property), as of PySpark's
    property."""

    def fget(self):
        _sync(self)
        return descriptor.__get__(self, type(self))

    return property(fget, doc=doc)


def _function(official, name, qualname, body, scope, binding=None):
    """A function `name` of the parameters of the function `official` (their names, kinds and
    defaults), or of the signature `binding` if given, of the qualified name `qualname`, whose body
    is `body` of the names of `scope`: Python binds a call of it as it binds `official`, and raises
    PySpark's messages of a call that does not bind (`LinearSVC.__init__() got an unexpected
    keyword argument 'foo'`). Its module, docstring and signature are those of `official` (of
    this module and of `binding`, of a native `official` of none)."""
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
    namespace = {"__name__": __name__, **scope, **defaults}
    exec(f"def {name}({', '.join(parts)}):\n    {body}\n", namespace)  # noqa: S102
    function = namespace[name]
    if sys.version_info >= (3, 11):
        function.__code__ = function.__code__.replace(co_qualname=qualname)
    function.__qualname__ = qualname
    function.__module__ = getattr(official, "__module__", None) or function.__module__
    function.__doc__ = getattr(official, "__doc__", None)
    function.__signature__ = signature
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
# persistence of the native writers and readers, and the uid of the native object.
_NATIVE_ALWAYS = frozenset({"write", "read", "uid"})


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
        # server-side object (the native constructor raises the AssertionError of PySpark's of
        # another `java_model`, which asserts a `RemoteModelRef`): its `Params.__init__` makes the
        # param maps and the Params of the object, `Identifiable.__init__` its uid (of the class's
        # name), and the `_setDefault` of its classes the defaults in their order.
        __init__ = init

        def check(*args, **kwargs):
            checker(None, *args, **kwargs)

    else:
        # The native constructor takes the objects of the arguments (a PipelineModel's `stages`, a
        # OneVsRestModel's `models`, a validator model's `bestModel`), which PySpark's `__init__`
        # sets as attributes of the instance that are the native class's own.
        __init__ = _function(inner, "__init__", qualname, "pass", {})

        def check(*args, **kwargs):
            checker(None, *args, **kwargs)

    try:
        no_arguments = all(
            p.default is not inspect.Parameter.empty
            or p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in inspect.signature(native).parameters.values()
        )
    except (TypeError, ValueError):
        no_arguments = False

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


class _JavaObject:
    """PySpark's `_java_obj` of an object of a native model or summary that PySpark's constructor
    did not make (of a fit, a load, a copy, an evaluation, a fetch), at its first read, in the
    object's attributes, where PySpark's sets it, which its `_call_java` (`try_remote_call`)
    reads: of a model, the `RemoteModelRef` of its server-side object (None of a model of none);
    of a summary, the id of its object and the methods of the chain (`"<id>.summary"`). A summary
    of PySpark's constructor has the `java_obj` of its `__init__` (and one of none PySpark's
    AttributeError)."""

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        if hasattr(type(obj), "_native_object_id"):
            ref = obj._native_object_id()
            if ref is None:
                value = None
            else:
                from pyspark.ml.util import RemoteModelRef

                value = RemoteModelRef(ref)
        else:
            value = obj._native_java_obj()
            if value is None:
                raise _no_attribute(obj, "_java_obj")
        obj.__dict__["_java_obj"] = value
        return value


def _no_attribute(obj, name):
    """Python's AttributeError of the attribute `name` that `obj` does not have, of its `name`
    and `obj` (which Python 3.10 and up read), as a missing attribute of PySpark's raises it."""
    error = AttributeError(f"'{type(obj).__name__}' object has no attribute '{name}'")
    error.name, error.obj = name, obj
    return error


def _training_summary(obj):
    """PySpark's `_summary` of a native model, which PySpark's fit sets: at its first read, the
    model's training summary (`summary`), in the object's attributes, of a model that has one
    (`hasSummary`); one of none (a load) has none, as PySpark's."""
    attributes = obj.__dict__
    if "_summary" in attributes:
        return attributes["_summary"]
    if not obj.hasSummary:
        raise _no_attribute(obj, "_summary")
    summary = obj.summary
    # The `summary()` method of a LinearSVCModel, an MLP's and an FM classifier's.
    summary = summary() if callable(summary) else summary
    attributes["_summary"] = summary
    return summary


def _set_training_summary(obj, summary):
    obj.__dict__["_summary"] = summary


# A property, as PySpark's params walk (`Params.params`) reads every attribute of `dir()` of the
# object but its class's properties, and a model of no summary has no `_summary`.
_TRAINING_SUMMARY = property(_training_summary, _set_training_summary)


def _has_training_summary(official):
    """Whether PySpark's class `official` has a training summary (`HasTrainingSummary`)."""
    from pyspark.ml.util import HasTrainingSummary

    return issubclass(official, HasTrainingSummary)


def _public_class(native, official, parents):
    """The public class of `native`, of PySpark's class `official` and of the public classes
    `parents` of its bases of the module (see `publish`)."""
    bases = (native,) + parents + (official,)
    # PySpark's metaclass (`ABCMeta` of the Params, `type` of the summaries).
    meta = type(official)
    holder = [None]
    __new__, __init__ = _constructors(native, official, holder)
    namespace = {
        "__module__": native.__module__,
        "__qualname__": native.__qualname__,
        "__doc__": official.__doc__ or native.__doc__,
        "__new__": __new__,
        "__init__": __init__,
        "__del__": _del,
        # The object of the server of PySpark's Java wrappers, which the native class holds
        # itself: `JavaWrapper.__del__` releases none. A model and a summary have PySpark's value
        # of it, which its `_call_java` reads.
        "_java_obj": (
            _JavaObject()
            if hasattr(native, "_native_object_id") or hasattr(native, "_native_java_obj")
            else None
        ),
    }
    if _has_training_summary(official):
        namespace["_summary"] = _TRAINING_SUMMARY
    try:
        # PySpark's signature, of its constructor (and of its `setParams`, which a class of params
        # takes of PySpark's).
        namespace["__signature__"] = inspect.signature(official)
    except (TypeError, ValueError):
        pass
    if hasattr(native, "_native_sync"):
        _params_api(native, official, namespace)
    elif _is_summary(official):
        _summary_api(native, official, namespace)
    getnewargs = _newargs_of(official)
    if getnewargs is not None:
        namespace["__getnewargs__"] = getnewargs
    public = meta(native.__name__, bases, namespace)
    holder[0] = public
    _native._register_public_class(native, public)
    return public


def _params_api(native, official, namespace):
    """The members of the public class of the native class `native` of params and PySpark's class
    `official`, in `namespace`: PySpark's own code of its param maps (the Params API, the getters
    and setters, `fit`, `copy`, `save`, `load`), and the native class's members of PySpark's
    server-side object (its fit, transform and evaluate, the model's members, the persistence), of
    the native object's params synced first."""
    namespace.update(
        {
            # PySpark's param maps, of the object's attributes, and its cache of `params`.
            "_paramMap": _LazyParamMap(),
            "_defaultParamMap": _LazyParamMap(),
            "_params": None,
        }
    )
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
        own = inspect.getattr_static(native, name)
        kind = type(own).__name__
        if kind == "getset_descriptor" and isinstance(member, property):
            # A property, as PySpark's (its `params` walk reads none), of PySpark's docstring; a
            # getter of PySpark's instance attribute (a PipelineModel's `stages`) or cached member
            # (`trees`) stays the native one, which the walk reads, as PySpark's reads them.
            namespace[name] = _synced_property(own, member.__doc__)
        elif kind == "method_descriptor":
            namespace[name] = _synced(getattr(native, name), name, official)
        elif kind in ("classmethod_descriptor", "staticmethod") and isinstance(
            member, (classmethod, staticmethod)
        ):
            # A class or static method, as PySpark's (`from_labels`, `loadDefaultStopWords`).
            namespace[name] = _class_member(own, name, official)
    for name, primitive in _PRIMITIVES.items():
        if hasattr(native, name) and (_member_of(official, name)[0], name) in dispatchers:
            namespace[primitive] = _synced(getattr(native, name), primitive, official)
    if hasattr(native, "__getstate__"):
        namespace["__getstate__"] = _synced(native.__getstate__, "__getstate__", official)
    if hasattr(native, "write"):
        # The native writer, of the params of the object (and of the objects that it holds) as of
        # the call: it takes them at its making.
        namespace["write"] = _synced(native.write, "write", official)
    if hasattr(native, "read") and isinstance(_member_of(official, "read")[1], classmethod):
        # The native reader, of PySpark's class method (`JavaMLReadable.read`, `Pipeline.read`).
        namespace["read"] = _class_member(inspect.getattr_static(native, "read"), "read", official)


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
        own = inspect.getattr_static(native, name)
        kind = type(own).__name__
        if kind == "getset_descriptor" and isinstance(member, property):
            namespace[name] = _property_of(own, member.__doc__)
        elif kind == "method_descriptor":
            namespace[name] = _method_of(getattr(native, name), name, official)


def _property_of(descriptor, doc):
    """A property of the native getter `descriptor`, of PySpark's docstring `doc`."""

    def fget(self):
        return descriptor.__get__(self, type(self))

    return property(fget, doc=doc)


def _method_of(member, name, official):
    """The native method `member` (`name`) of PySpark's class `official`, as PySpark names it,
    documents it and binds a call of it (see `_synced`), of no params to sync."""
    return _synced(member, name, official, sync=False)


def _newargs_of(official):
    """The `__getnewargs__` of a class whose PySpark constructor takes the objects of the instance
    (a PipelineModel's `stages`, a CrossValidatorModel's `bestModel`, `avgMetrics`, ...), which
    `copy.deepcopy` and `pickle` construct a copy of, as of PySpark's instance attributes of the
    same names; None of the others."""
    _, init = _member_of(official, "__init__")
    parameters = list(inspect.signature(getattr(init, "__wrapped__", init)).parameters.values())[1:]
    positional = [
        p
        for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not any(p.default is inspect.Parameter.empty for p in positional):
        return None
    names = tuple(p.name for p in positional)

    def __getnewargs__(self):
        return tuple(getattr(self, name) for name in names)

    return __getnewargs__


def publish(namespace, official_module):
    """Replace each native class of the module of the globals `namespace` by its public class, of
    PySpark's class of the same name of `official_module`."""
    official = importlib.import_module(official_module)
    natives = [
        (name, value)
        for name, value in list(namespace.items())
        if isinstance(value, type) and getattr(_native, name, None) is value
    ]
    # A native base class (LDA's `LDAModel`) before its subclasses.
    natives.sort(key=lambda item: len(item[1].__mro__))
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
        public = _public_class(native, official_class, tuple(parents))
        published[name] = (official_class, public)
        namespace[name] = public
    return {name: public for name, (_, public) in published.items()}
