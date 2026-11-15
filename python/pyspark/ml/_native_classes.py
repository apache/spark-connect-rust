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
- `copy` is the native class's, of the class of the object (a subclass) and a copy of the object's
  attributes, as PySpark's `copy.copy` copies them.
"""

import abc
import importlib
import inspect
import sys

from pyspark import _pyspark as _native
from pyspark import keyword_only


def _del(self):
    """The native class releases its server-side objects itself: PySpark's `JavaWrapper.__del__`
    has none to release, and its imports fail at the interpreter's exit."""


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
    no param map."""

    def primitive(self, dataset):
        return member(self, dataset)

    primitive.__name__ = name
    primitive.__qualname__ = f"{official.__name__}.{name}"
    return primitive


def _copy_of(native_copy):
    """The native `copy`, of the class of the object (a subclass of the public class) and a copy of
    its attributes, as PySpark's `Params.copy` makes its copy of `copy.copy`."""

    def copy(self, extra):
        that = native_copy(self, extra)
        cls = type(self)
        if type(that) is not cls:
            try:
                that.__class__ = cls
            except TypeError:  # A subclass of `__slots__`, of another layout.
                pass
        that.__dict__.update(self.__dict__)
        return that

    return copy


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
        # PySpark's `__init__` sets the params of its keywords by `setParams` (`_input_kwargs`, of
        # `keyword_only`).
        if hasattr(native, "setParams"):
            body = "self.setParams(**self._input_kwargs)"
        else:
            body = "\n    ".join(
                [
                    "for __name, __value in self._input_kwargs.items():",
                    "    self.set(self.getParam(__name), __value)",
                ]
            )
        __init__ = keyword_only(_function(inner, "__init__", qualname, body, {}))

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
        if keyword and cls is not holder[0]:
            obj.uid = cls._randomUID()
        return obj

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
        namespace["__signature__"] = inspect.signature(native)
    except (TypeError, ValueError):
        pass
    dispatchers = _dispatchers()
    for name in ("fit", "fitMultiple", "transform", "evaluate", "save", "load"):
        owner, member = _member_of(official, name)
        if (owner, name) not in dispatchers or not hasattr(native, name):
            continue
        namespace[name] = member
        if name in _PRIMITIVES:
            primitive = _PRIMITIVES[name]
            namespace[primitive] = _primitive(getattr(native, name), primitive, official)
    if hasattr(native, "copy"):
        _, member = _member_of(official, "copy")
        if member is not None and "extra" in inspect.signature(member).parameters:
            namespace["copy"] = _function(
                member,
                "copy",
                member.__qualname__,
                "return __copy(self, extra)",
                {"__copy": _copy_of(native.copy)},
            )
    getnewargs = _newargs_of(official)
    if getnewargs is not None:
        namespace["__getnewargs__"] = getnewargs
    public = meta(native.__name__, bases, namespace)
    holder[0] = public
    _native._register_public_class(native, public)
    return public


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
