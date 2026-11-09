# Known behavior differences from PySpark Connect

Behaviors of the Rust-backed Python client that differ from PySpark Connect 4.3.0, kept
here so that a version upgrade does not rediscover them. Each entry says whether the
difference is intentional or tracked for a fix.

## `createDataFrame`

- **A pandas DataFrame with a StructType of another width** (intentional, SPARK-60902).
  PySpark's pandas conversion zips the DataFrame's columns with the schema's fields. With
  fewer fields than columns it silently drops the extra columns, and with more fields it
  raises PyArrow's `ArrowInvalid`. This client raises `AXIS_LENGTH_MISMATCH` for both
  rather than drop data.
- **NumPy ndarray input** (tracked by SPARK-60772). Not accepted yet.
- **Customized pandas Row columns without PyArrow** (intentional optional-dependency
  fallback). When PyArrow is unavailable, the native row fallback retains its previous
  named-Row struct inference; it does not use Arrow's array inference or mixed-element
  rejection. Official PySpark requires PyArrow for pandas input. With PyArrow available,
  inferred customized pandas columns containing Rows use its whole-column inference.
  This fallback covers unavailable modules detected by `pyarrow_available`, not an
  installed PyArrow module that fails during import.
- **Nested pandas Decimal inference and failures** (intentional; related change:
  SPARK-60696).

  - **Inferred precision and scale.** On the native row fallback, inferred nested Decimal
    fields in non-Row columns keep precision 38 and the greatest observed scale; PySpark
    infers the smallest precision that covers the values. When PyArrow is available,
    columns containing Rows follow Arrow's array inference, smallest covering Decimal
    precision and incompatible-value errors instead. Without PyArrow, those columns retain
    native struct inference and this precision difference. Increasing scale on the native
    path can make a large value exceed precision 38 even when the first value's scale
    could encode every value by discarding only trailing zeros. For example, with
    `[Decimal("9" * 37 + ".0")]` followed by `[Decimal("1.0000000")]`, the first value
    would need 44 digits at scale 7, beyond precision 38; the previous native fallback
    accepted both at scale 1. PySpark Connect 4.3.0 constructs a DataFrame for this
    example, but requesting its schema or collecting results raises `ArithmeticException`
    (`DECIMAL_PRECISION_EXCEEDS_MAX_PRECISION`, precision 44, maximum 38). This client's
    native fallback instead raises `PySparkRuntimeError` during construction, before
    sending a relation. The rejection stage and error class differ.

  - **Scale beyond 38.** A later value that increases the scale beyond 38 raises
    `ValueError` before upload. This can reject later zero or trailing-zero values that
    the previous native fallback accepted at the first value's scale. This scale handling
    applies to the non-Row columns and no-PyArrow fallback described above.

  - **Explicit schema.** With an explicit schema, a nested Decimal that does not fit
    raises `PySparkRuntimeError` only on the native row fallback, for frames rejected by
    `accepts_pandas_table`. Ordinary frames use PySpark's Arrow conversion and raise its
    errors. Both paths reject an explicit-schema value that does not fit before a
    relation is sent.
- **Multiple invalid pandas columns with an explicit Decimal field** (intentional,
  SPARK-60696). On native row fallback, Decimal exactness is checked before the encoder
  converts other fields. If an earlier non-Decimal column and a later Decimal column are
  both invalid, this client reports the Decimal column first; PySpark reports the first
  invalid column in schema order. Both reject the input before a relation is sent.
- **Hook timing in customized pandas frames containing Rows** (intentional). With PyArrow
  available, Row columns are prepared with Arrow before native scalar conversion.
  Detection covers direct Row cells and nesting through list, tuple and dict instances,
  including subclasses. Other cell containers are not traversed by this adaptation.
  Discovery walks those containers with an explicit frame stack in base-iterator order.
  Shared acyclic containers are visited at each occurrence. A cycle on the active path
  raises `ValueError` with `cyclic container in inferred pandas Row discovery`, including
  in Row-free cells. Ordinary discovery errors retain their column index for deferred
  error handling and cause the frame to read Arrow settings even if discovery found no
  Row. Native and Arrow conversions retain their existing nesting limits.
  Replacing a nested dict value containing a Row hashes its original key and may compare
  keys during lookup. Custom key hooks can therefore run during discovery; callback
  counts follow normal Python dict behavior, including collisions.
  Adapted frames use the Arrow importer's session settings and transport policy:
  `spark.sql.session.localRelationCacheThreshold` selects inline data or cached-artifact
  upload, with the importer's size and chunk limits. Frames whose discovery finds neither
  Rows nor an error retain native conversion and its upstream transport policy. For native
  rows, Arrow data below 64 KiB stays inline without that settings RPC; at 64 KiB or more,
  the upstream path reads the local-relation settings. For adapted frames, a failure
  reading the shared settings is reported at the first selected column's position, so an
  earlier invalid native column is reported first. Official PySpark reads configuration
  before converting any column.

  Other columns run native scalar conversion (`py_to_value`) at most once per cell.
  Decimal typing can read original objects again: column typing calls `as_tuple()` on
  values other than exact C Decimals, and Arrow rereads columns mixing Decimals with other
  values. After an ordinary native scalar-conversion exception, later cells in that column
  are not converted. Ordinary failures are retained and checked in column order alongside
  inference and encoding, so the earliest invalid column wins. The deferred error
  handling retains only Python `Exception` instances; other propagated exceptions, such
  as `KeyboardInterrupt` from a Decimal hook, return immediately. On failing input,
  callbacks in later columns may already have run, and callbacks in a Row column may
  precede those in an earlier native column. Official PySpark converts complete columns
  from left to right. The fallback's frame normalization hooks are preserved; callback
  timing differs from official PySpark.
- **Decimal context hooks during local-row conversion** (intentional, SPARK-60704).
  Quantum and context objects are shared by Decimal type within one conversion and built
  when its first non-null, non-NaN value is converted. PySpark constructs a context for
  each field's converter before converting values. A custom hook that changes
  `decimal.DefaultContext` or mutates its supplied context during a conversion can
  observe different timing or sharing. Changes between `createDataFrame` calls apply to
  the next conversion.
- **Decimal class rebinding** (intentional, SPARK-60704). Native Decimal conversion and
  inference for local rows and pandas input recognize the cached C `_decimal.Decimal`
  class, falling back to the cached public `decimal.Decimal` when the accelerator is
  unavailable; accelerator absence is cached too. If `_decimal` is importable at first
  resolution, its class takes precedence even when the public `decimal.Decimal` is the
  pure-Python class. Values of that pure-Python class are then not recognized as
  Decimals: local-row Decimal conversion raises `AssertionError` and inference raises
  `CANNOT_INFER_TYPE_FOR_FIELD`, while the native row fallback of pandas input (frames
  that the Arrow path does not take) without a schema or with names only reads them as
  doubles through `__float__`, potentially losing precision. Pandas frames on the Arrow
  path convert them as PySpark does. Quantization also uses a quantum of the selected C
  class, even when the public alias is rebound. The context factory and rounding mode
  still come from the public `decimal` module at first use; mixing its pure-Python
  context with explicit C Decimal values is unsupported. The text shortcut assumes
  `_decimal.Decimal` is immutable, as on CPython; mutable implementations of that class
  on other interpreters are not covered. This can differ from PySpark when an
  application rebinds the public class alias during the process.

## `DataFrame`

- **Public base and native MRO.** `pyspark.sql.DataFrame` is implemented as a
  Python base/factory, allowing subclasses such as
  `pyspark.testing.mlutils.MockDataset` to allocate their own Python state with
  `object.__new__`. The concrete class remains
  `pyspark.sql.connect.dataframe.DataFrame`, implemented by the Rust extension. It is
  registered as a virtual subclass of the public base: `isinstance` and `issubclass`
  recognize it, while its actual MRO stays native. Consequently, `super` and MRO
  introspection do not treat a native frame as inheriting from the public base.

- **Metaclass and subclass serialization.** The public base uses an `ABCMeta`
  subclass for virtual registration and attribute forwarding, while official
  PySpark uses `type`. It exposes `register()`, and
  multiple inheritance with an unrelated metaclass requires a combined metaclass.
  Ordinary Python subclasses use the standard `abc.ABCMeta` so by-value
  cloudpickling names an importable metaclass on official Python workers. Explicit
  user metaclass subclasses are preserved and retain their own serialization
  requirements. A local subclass and a closure capturing that subclass were
  cloudpickled by the native client and reconstructed and exercised in an
  official-only Python process without importing the native extension. This does
  not establish portability of instances, explicit user metaclasses or server UDFs.

- **Public class mutation and dispatch.** Public-base method and property calls
  look up the concrete implementation at call time, including concrete-class
  patches made after import. Assigning or deleting user extensions directly on the
  public class also updates the native class, so added helpers and overridden
  methods reach existing native frames;
  assignments on Python subclasses remain local to those subclasses.
  Deleting an initial public method or property removes its public-base exposure
  and restores the concrete implementation that preceded a public override, while
  preserving any later independent concrete patch. Deleting an added
  helper removes the forwarded helper; deleting a temporary extension that shadows
  a native-only attribute restores that native attribute.
  Class layout, constructor, import/ABC metadata and instance attribute protocols
  `__getattribute__`, `__setattr__` and `__delattr__` stay on the public base. Those
  protocols follow ordinary Python inheritance on independent subclasses; virtual
  native membership does not inherit them.

- **Attribute resolution.** Explicit public `__getattr__` calls use the native
  column resolver, including existing attribute names, unsupported JVM names, and
  dunder/legacy column validation. While a user override of public `__getattr__`
  is active, native frames use generic Python attribute lookup before calling that
  hook. Undo restores the saved native lookup entry and prior hook state, including
  absence of a native `__getattr__` entry. Importing the public base does not alter
  the native lookup entries. The native-only private method
  `_get_column_by_attribute` bridges explicit public lookup to the Rust resolver;
  it is an intentional extra name in concrete-class introspection and completion.

- **Independent Python subclasses and construction.** Python subclasses override
  operations for their own data model. As in PySpark, inherited public properties
  read `None`, while native-dispatched methods,
  including `repr`, `dir`, and missing-attribute `__getattr__`, raise
  `NOT_IMPLEMENTED` on independent Python data. Consequently, `hasattr` for an
  unknown name also raises that exception, including on `MockDataset`.
  The official undecorated empty methods `checkpoint`, `localCheckpoint`,
  `lateralJoin`, `nearestByJoin`, `toPandas`, `asTable`, `scalar` and `exists` return
  `None` after argument binding. The inherited `__arrow_c_stream__` uses the
  generic `SparkArrowCStreamer` helper with the independent frame and requested
  schema, as on the official public base.
  Internal `_session` and `_plan` remain ordinary instance state on those subclasses.
  `is_cached` is an annotation and assignable instance state, as on the official
  public base. Concrete getters for these names are not installed as public
  properties.
  A public-base subclass that inherits the factory's `__new__` receives a plain
  concrete frame, and its own `__init__` is not called. Define `__new__` using
  `object.__new__` for independent Python data, or subclass
  `pyspark.sql.connect.dataframe.DataFrame` for native-backed subclass behavior.
  A direct native subclass must use the native constructor to initialize its Rust
  payload; `object.__new__` cannot safely allocate that payload.

- **Validation status.** All 22 subclass cases passed in the native offline suite.
  The unchanged official pipeline, tuning and one-vs-rest selection passed on both
  the official and native high-level clients with `PYTHONHASHSEED=0`: 16 passed and
  four upstream-disabled GC tests skipped on each client. This includes
  `test_pipeline` and `test_identity_pipeline`. An earlier run with independent
  default Python hash seeds failed native Random Forest cross-validation on a
  one-class training fold; that failure is retained, and its cause is not proven
  by the fixed-hash passes. API inspection records the public-base and
  concrete-class differences; inherited signature and annotation differences remain.
  These results do not establish complete DataFrame or ML parity.
