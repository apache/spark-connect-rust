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
"""

import abc
import importlib
import inspect

from pyspark import _pyspark as _native


def _init(self, *args, **kwargs):  # noqa: ARG001 - the native class's `__new__` takes them
    """The native class's constructor (its `__new__`) makes the object: PySpark's `__init__`s,
    which write its `uid` and its param maps, do not run."""


def _del(self):
    """The native class releases its server-side objects itself: PySpark's `JavaWrapper.__del__`
    has none to release, and its imports fail at the interpreter's exit."""


def _public_class(native, official, parent):
    """The public class of `native`, of PySpark's class `official` and of the public class
    `parent` of the native base class of `native`, if it has one."""
    bases = (native,) + ((parent,) if parent is not None else ()) + (official,)
    # A metaclass of `abc`, so that the public classes take virtual subclasses (see `publish`).
    meta = abc.ABCMeta if type(official) is type else type(official)
    namespace = {
        "__module__": native.__module__,
        "__qualname__": native.__qualname__,
        "__doc__": native.__doc__,
        "__init__": _init,
        "__del__": _del,
        # The object of the server of PySpark's Java wrappers, which the native class holds
        # itself: `JavaWrapper.__del__` releases none.
        "_java_obj": None,
    }
    try:
        namespace["__signature__"] = inspect.signature(native)
    except (TypeError, ValueError):
        pass
    public = meta(native.__name__, bases, namespace)
    _native._register_public_class(native, public)
    return public


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
