"""The public classes of the native ML classes.

The public class of a native class is a subclass of it and of PySpark's class of the same name,
which a module of `pyspark.ml` has: its objects are instances of every class of the MRO of PySpark's
class, as PySpark's objects are (`MLWritable`, `Identifiable`, `HasMaxIter`, `ClassificationModel`,
the summary bases, the Java wrappers, and PySpark's class itself), and the members of the native
class come first. The native module (`pyspark._pyspark`) makes every object of a native class of
its public class (`_register_public_class`), as the constructor of the public class does.

A class of PySpark's of the same module that is a base of PySpark's class, and whose native class
is not a base of the native class (a training summary of its summary), has the public class as a
virtual subclass, so that `isinstance` holds of the public classes as of PySpark's.

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

A public class of a native class of params keeps them in PySpark's param maps (`_paramMap`,
`_defaultParamMap`), in the object's attributes, and takes PySpark's own Python code over them (the
Params API, the getters and setters, `setParams`, a `keyword_only` constructor) as it is. Before a
native member reads the params (a fit, a transform, an evaluation, a writer, a model's member), the
native object's params are made those of the maps, and so are those of the objects that it holds
(`_sync`): the native object reads them of their native objects. The maps of an object that
PySpark's constructor did not make (a fit's model, a load, a native copy) are made of the native
object's values at their first read.
"""

import abc
import functools
import importlib
import inspect
import sys
import types

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


def _of_pysparks(function, member, name, official):
    """`function`, the member `name` of the public class of PySpark's class `official` over the
    native member `member`, of PySpark's qualified name, and of the docstring and signature of
    PySpark's member of the name (the native member's, if PySpark's class has none)."""
    function.__name__ = name
    function.__qualname__ = f"{official.__name__}.{name}"
    _, pysparks = _member_of(official, name)
    if not callable(pysparks):
        pysparks = member
    function.__doc__ = pysparks.__doc__
    try:
        function.__signature__ = inspect.signature(pysparks)
    except (TypeError, ValueError):
        pass
    return function


def _synced(member, name, official):
    """The native method `member` (`name`), of the native object's params synced first."""

    def synced(self, *args, **kwargs):
        _sync(self)
        return member(self, *args, **kwargs)

    return _of_pysparks(synced, member, name, official)


def _synced_property(descriptor, doc):
    """A property of the native getter `descriptor`, of the native object's params synced first, so
    that PySpark's `params` walk does not read it (it reads no property), as of PySpark's
    property."""

    def fget(self):
        _sync(self)
        return descriptor.__get__(self, type(self))

    return property(fget, doc=doc)


def _function(official, name, qualname, body, scope):
    """A function `name` of the parameters of the function `official` (their names, kinds and
    defaults), of the qualified name `qualname`, whose body is `body` of the names of `scope`:
    Python binds a call of it as it binds `official`, and raises PySpark's messages of a call that
    does not bind (`LinearSVC.__init__() got an unexpected keyword argument 'foo'`)."""
    signature = inspect.signature(official)
    parts, defaults, star, positional_only = [], {}, False, False
    for index, parameter in enumerate(signature.parameters.values()):
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
    namespace = {**scope, **defaults}
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
# no param map) is the public class's.
_PRIMITIVES = {"fit": "_fit", "transform": "_transform", "evaluate": "_evaluate"}


def _primitive(member, name, official):
    """PySpark's `_fit` (`_transform`, `_evaluate`) of the native member `member` (`fit`, ...), of
    no param map, of the native object's params synced first."""

    def primitive(self, dataset):
        _sync(self)
        return member(self, dataset)

    return _of_pysparks(primitive, member, name, official)


# The native members that a public class keeps of the native class whatever PySpark's are: the
# persistence of the native writers and readers, and the uid of the native object.
_NATIVE_ALWAYS = frozenset({"write", "read", "uid"})


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

    else:
        # The native constructor takes the object of the arguments (a model's `java_model`, a
        # PipelineModel's `stages`); PySpark's `JavaModel.__init__` asserts a server-side object.
        body = "assert java_model is None" if "java_model" in signature.parameters else "pass"
        __init__ = _function(inner, "__init__", qualname, body, {})

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


def _public_class(native, official, parent):
    """The public class of `native`, of PySpark's class `official` and of the public class
    `parent` of the native base class of `native`, if it has one."""
    bases = (native,) + ((parent,) if parent is not None else ()) + (official,)
    # A metaclass of `abc`, so that the public classes take virtual subclasses (see `publish`).
    meta = abc.ABCMeta if type(official) is type else type(official)
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
        # itself: `JavaWrapper.__del__` releases none.
        "_java_obj": None,
    }
    try:
        # PySpark's signature, of its constructor (and of its `setParams`, which a class of params
        # takes of PySpark's).
        namespace["__signature__"] = inspect.signature(official)
    except (TypeError, ValueError):
        pass
    if hasattr(native, "_native_sync"):
        _params_api(native, official, namespace)
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
    for name, primitive in _PRIMITIVES.items():
        if hasattr(native, name) and (_member_of(official, name)[0], name) in dispatchers:
            namespace[primitive] = _primitive(getattr(native, name), primitive, official)
    if hasattr(native, "__getstate__"):
        namespace["__getstate__"] = _synced(native.__getstate__, "__getstate__", official)
    if hasattr(native, "write"):
        # The native writer, of the params of the object (and of the objects that it holds) as of
        # the call: it takes them at its making.
        namespace["write"] = _synced(native.write, "write", official)


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


def publish(namespace, official_module, bases=None):
    """Replace each native class of the module of the globals `namespace` by its public class, of
    PySpark's class of the same name of `official_module`; `bases` gives PySpark's class of a name
    whose module this client does not vendor (`pyspark.ml.stat`)."""
    official = importlib.import_module(official_module) if official_module else None
    natives = [
        (name, value)
        for name, value in list(namespace.items())
        if isinstance(value, type) and getattr(_native, name, None) is value
    ]
    # A native base class (LDA's `LDAModel`) before its subclasses.
    natives.sort(key=lambda item: len(item[1].__mro__))
    published = {}
    for name, native in natives:
        official_class = (bases or {}).get(name) or getattr(official, name, None)
        if official_class is native or (official is None and official_class is None):
            continue
        if official_class is None:
            # A module of pyspark.ml that is not imported in full yet (an import cycle) lacks it.
            raise ImportError(f"{official_module} has no {name} of the native class {native}")
        parent = next(
            (published[b.__name__][1] for b in native.__mro__[1:] if b.__name__ in published),
            None,
        )
        public = _public_class(native, official_class, parent)
        published[name] = (official_class, public)
        namespace[name] = public
    # The classes of PySpark's of the same module that are bases of PySpark's class, of which the
    # public class is a virtual subclass where its native class is not a subclass.
    for official_class, public in published.values():
        for base in official_class.__mro__[1:]:
            if base.__name__ in published and published[base.__name__][0] is base:
                ours = published[base.__name__][1]
                if not issubclass(public, ours):
                    ours.register(public)
    return {name: public for name, (_, public) in published.items()}
