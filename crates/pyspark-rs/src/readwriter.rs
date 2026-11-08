//! PyO3 wrapper for spark_connect::readwriter::DataFrameReader (`spark.read`).

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::readwriter::DataFrameReader;

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

/// Python wrapper for the batch DataFrameReader.
///
/// PySpark's reader is mutable: `format`/`schema`/`option`/`options` change `self`
/// and return it, and a reader can be used again after a read. The core reader is a
/// consuming builder, so the setters replace the wrapped value in place and return
/// the same Python object, and the terminal reads build their DataFrame from a clone.
#[pyclass(
    name = "DataFrameReader",
    module = "pyspark.sql.readwriter",
    weakref,
    dict
)]
pub struct PyDataFrameReader {
    inner: DataFrameReader,
}

impl PyDataFrameReader {
    pub fn new(reader: DataFrameReader) -> Self {
        PyDataFrameReader { inner: reader }
    }

    /// A copy of the current reader with per-call `**options` applied (skipping
    /// None, lowercasing bools), for a format-specific read.
    fn with_opts(&self, options: Option<&Bound<'_, PyDict>>) -> PyResult<DataFrameReader> {
        let mut r = self.inner.clone();
        if let Some(opts) = options {
            for (k, v) in opts.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    r = r.option(&k.str()?.to_string(), &val);
                }
            }
        }
        Ok(r)
    }

    /// Keep the options of a format-specific read on this reader, as PySpark does
    /// (its `json`/`csv`/... set the options and the format on `self`, then `load`).
    /// Called only after every argument was validated, so a failed call leaves the
    /// reader unchanged.
    fn commit(&mut self, r: &DataFrameReader, format: &str) {
        self.inner = r.clone().format(format);
    }

    /// The DataFrame of `r`, this reader with the options of a call of `json`, `csv` or `xml`
    /// of `path`, as PySpark's: a DataFrame of strings, which it parses as `parse_format`,
    /// keeping the schema and the options on this reader but not the format, or else a path
    /// or a list of them, which it loads as `format`. `csv`, unlike the others, passes any
    /// other `path` to `load`, which asserts it.
    fn read_or_parse(
        &mut self,
        r: DataFrameReader,
        format: &str,
        parse_format: spark_connect::plan::ParseFormat,
        path: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataFrame> {
        if let Ok(df) = path.extract::<PyRef<'_, PyDataFrame>>() {
            let parsed = r
                .clone()
                .parse(parse_format, &df.dataframe)
                .map(PyDataFrame::new)
                .to_pyerr()?;
            self.inner = r;
            return Ok(parsed);
        }
        if path.is_instance_of::<pyo3::types::PyString>()
            || path.is_instance_of::<pyo3::types::PyList>()
            || parse_format == spark_connect::plan::ParseFormat::Csv
        {
            let paths = paths_of(path)?;
            self.commit(&r, format);
            return Ok(PyDataFrame::new(r.format(format).load_paths(paths)));
        }
        Err(not_expected_type(path, "path", "str, list, or DataFrame"))
    }
}

/// The paths of `path`, a path or a list of them, or None for none, as PySpark's `load` takes
/// them: anything else fails its assertion.
fn paths_of(path: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if path.is_none() {
        return Ok(vec![]);
    }
    if path.is_instance_of::<pyo3::types::PyString>() {
        return Ok(vec![path.extract()?]);
    }
    if !path.is_instance_of::<pyo3::types::PyList>() {
        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
    }
    path_list(&path.try_iter()?.collect::<PyResult<Vec<_>>>()?)
}

/// The paths of `paths`, strings, as PySpark's `load` asserts them.
fn path_list(paths: &[Bound<'_, PyAny>]) -> PyResult<Vec<String>> {
    paths
        .iter()
        .map(|path| {
            if path.is_instance_of::<pyo3::types::PyString>() {
                path.extract()
            } else {
                Err(pyo3::exceptions::PyAssertionError::new_err(()))
            }
        })
        .collect()
}

/// PySpark's `NOT_EXPECTED_TYPE` error of the argument `name`, `value`, which should be of
/// the type `expected`.
fn not_expected_type(value: &Bound<'_, PyAny>, name: &str, expected: &str) -> PyErr {
    let type_name = value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_default();
    crate::column::raise_pyspark(
        value.py(),
        "PySparkTypeError",
        "NOT_EXPECTED_TYPE",
        &[
            ("arg_name", name),
            ("expected_type", expected),
            ("arg_type", &type_name),
        ],
    )
}

/// The columns of `df.write.bucketBy(numBuckets, col, *cols)` and `sortBy(col, *cols)`, as
/// PySpark's: column names, or a list or tuple of them alone.
pub(crate) fn writer_columns(
    col: &Bound<'_, PyAny>,
    cols: &[Bound<'_, PyAny>],
) -> PyResult<Vec<String>> {
    let sequence =
        col.is_instance_of::<pyo3::types::PyList>() || col.is_instance_of::<pyo3::types::PyTuple>();
    let (col, cols) = if sequence {
        if !cols.is_empty() {
            let type_name = col.get_type().name()?.to_string();
            return Err(crate::column::raise_pyspark(
                col.py(),
                "PySparkValueError",
                "CANNOT_SET_TOGETHER",
                &[("arg_list", &format!("`col` of type {type_name} and `cols`"))],
            ));
        }
        // An empty list or tuple raises an IndexError, as PySpark's `col[0]`.
        let first = col.get_item(0)?;
        let rest: Vec<Bound<'_, PyAny>> = col.try_iter()?.skip(1).collect::<PyResult<_>>()?;
        (first, rest)
    } else {
        (col.clone(), cols.to_vec())
    };
    if let Some(other) = cols
        .iter()
        .find(|c| !c.is_instance_of::<pyo3::types::PyString>())
    {
        return Err(not_expected_type(other, "cols", "list[str]"));
    }
    if !col.is_instance_of::<pyo3::types::PyString>() {
        return Err(not_expected_type(&col, "col", "list[str]"));
    }
    std::iter::once(&col)
        .chain(cols.iter())
        .map(|c| c.extract())
        .collect()
}

/// The number of buckets of `df.write.bucketBy`, an int, as PySpark's.
pub(crate) fn bucket_count(num_buckets: &Bound<'_, PyAny>) -> PyResult<i32> {
    if num_buckets.is_instance_of::<pyo3::types::PyInt>() {
        num_buckets.extract()
    } else {
        Err(not_expected_type(num_buckets, "numBuckets", "int"))
    }
}

/// Apply a single reader option (skipping None; bool->"true"/"false").
fn set_opt(
    r: DataFrameReader,
    name: &str,
    v: Option<&Bound<'_, PyAny>>,
) -> PyResult<DataFrameReader> {
    match v {
        Some(x) => match crate::coerce_option_value(x)? {
            Some(sv) => Ok(r.option(name, &sv)),
            None => Ok(r),
        },
        None => Ok(r),
    }
}

/// A reader `schema` as pyspark's `DataFrameReader.schema` and `DataStreamReader.schema`
/// send it: a StructType as its JSON, which keeps field metadata, and a string as is, for
/// the server to parse. Anything else raises `NOT_EXPECTED_TYPE`.
pub(crate) fn reader_schema(schema: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(schema) = schema.extract::<String>() {
        return Ok(schema);
    }
    if let Ok(st) = schema.extract::<PyRef<crate::types::PyStructType>>() {
        return Ok(spark_connect::types::DataType::Struct {
            fields: st.current_fields(schema.py())?,
        }
        .json());
    }
    let arg_type = schema.get_type().name()?.to_string();
    Err(crate::column::raise_pyspark(
        schema.py(),
        "PySparkTypeError",
        "NOT_EXPECTED_TYPE",
        &[
            ("arg_name", "schema"),
            ("expected_type", "str or struct type"),
            ("arg_type", &arg_type),
        ],
    ))
}

/// The `AnalysisException` of `_LEGACY_ERROR_TEMP_1189` that PySpark's `DataFrameReader.changes`
/// and `DataStreamReader.changes` raise for a reader of a schema, of the message of the JVM
/// error class, which PySpark's error classes lack.
pub(crate) fn changes_of_schema_error(py: Python<'_>) -> PyErr {
    let exception = (|| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("errorClass", "_LEGACY_ERROR_TEMP_1189")?;
        let parameters = PyDict::new(py);
        parameters.set_item("operation", "changes")?;
        kwargs.set_item("messageParameters", parameters)?;
        py.import("pyspark.errors")?
            .getattr("AnalysisException")?
            .call(
                ("User specified schema not supported with `changes`.",),
                Some(&kwargs),
            )
    })();
    match exception {
        Ok(exception) => PyErr::from_value(exception),
        Err(err) => err,
    }
}

/// Apply the `schema` argument of a format method, as pyspark does through `schema()`.
fn apply_reader_schema(r: DataFrameReader, v: &Bound<'_, PyAny>) -> PyResult<DataFrameReader> {
    Ok(r.schema(reader_schema(v)?))
}

/// The `_set_opts` of the readers and writers, as PySpark's `OptionUtils._set_opts`: the schema of
/// `slf`, if any, and its options of `options` that are not None, through its own `schema` and
/// `option`, which convert them and raise their errors.
pub(crate) fn set_opts(
    slf: &Bound<'_, PyAny>,
    schema: Option<&Bound<'_, PyAny>>,
    options: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<()> {
    if let Some(schema) = schema.filter(|schema| !schema.is_none()) {
        slf.call_method1("schema", (schema,))?;
    }
    if let Some(options) = options {
        for (key, value) in options.iter() {
            if !value.is_none() {
                slf.call_method1("option", (key, value))?;
            }
        }
    }
    Ok(())
}

#[pymethods]
impl PyDataFrameReader {
    /// Set the schema, if any, and the options of `options` that are not None, through its own
    /// `schema` and `option`, as PySpark's `OptionUtils._set_opts`, which pandas-on-Spark calls.
    #[pyo3(signature = (schema=None, **options))]
    fn _set_opts(
        slf: &Bound<'_, Self>,
        schema: Option<Bound<'_, PyAny>>,
        options: Option<Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        crate::readwriter::set_opts(slf.as_any(), schema.as_ref(), options.as_ref())
    }

    /// Set the source format (e.g. "parquet", "json", "csv"). As PySpark's, None unsets it, so
    /// that the server reads its default source.
    fn format<'py>(mut slf: PyRefMut<'py, Self>, source: Option<&str>) -> PyRefMut<'py, Self> {
        slf.inner = match source {
            Some(source) => slf.inner.clone().format(source),
            None => slf.inner.clone().clear_format(),
        };
        slf
    }

    /// Set the schema: a StructType or a DDL string, as in pyspark.
    fn schema<'py>(
        mut slf: PyRefMut<'py, Self>,
        schema: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let schema = reader_schema(schema)?;
        slf.inner = slf.inner.clone().schema(schema);
        Ok(slf)
    }

    /// Set a single read option. `None` leaves the option unset (not the string
    /// "None"); booleans lowercase to "true"/"false" (reference `to_str` semantics).
    fn option<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        if let Some(v) = crate::coerce_option_value(value)? {
            slf.inner = slf.inner.clone().option(key, &v);
        }
        Ok(slf)
    }

    /// Set multiple read options. Mirrors reference `DataFrameReader.options(**options)`
    /// - keyword args; `None` values are skipped and booleans lowercased.
    #[pyo3(signature = (**options))]
    fn options<'py>(
        mut slf: PyRefMut<'py, Self>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut map = HashMap::new();
        if let Some(options) = options {
            for (k, v) in options.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    map.insert(k.str()?.to_string(), val);
                }
            }
        }
        slf.inner = slf.inner.clone().options(map);
        Ok(slf)
    }

    /// The configured source format (PySpark Connect's `DataFrameReader._format`).
    #[getter]
    fn _format(&self) -> Option<String> {
        self.inner.get_format().map(str::to_string)
    }

    /// The configured schema string, empty when unset (`DataFrameReader._schema`).
    #[getter]
    fn _schema(&self) -> String {
        self.inner.get_schema().to_string()
    }

    /// The configured read options (`DataFrameReader._options`); a copy.
    #[getter]
    fn _options(&self) -> HashMap<String, String> {
        self.inner.get_options().clone()
    }

    /// The JVM reader of classic PySpark, which PySpark Connect's raises
    /// `JVM_ATTRIBUTE_NOT_SUPPORTED` for.
    #[getter]
    fn _jreader(&self, py: Python<'_>) -> PyResult<()> {
        Err(crate::column::raise_pyspark(
            py,
            "PySparkAttributeError",
            "JVM_ATTRIBUTE_NOT_SUPPORTED",
            &[("attr_name", "_jreader")],
        ))
    }

    /// Load data from the (optional) path. Mirrors the reference
    /// `DataFrameReader.load(path=None, format=None, schema=None, **options)`: the
    /// one-shot form applies format/schema/options here before reading, equivalent to
    /// chaining `.format(...).schema(...).options(...).load(path)`.
    #[pyo3(signature = (path=None, format=None, schema=None, **options))]
    fn load(
        &mut self,
        path: Option<&Bound<'_, PyAny>>,
        format: Option<&str>,
        schema: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.with_opts(options)?;
        if let Some(f) = format {
            r = r.format(f);
        }
        if let Some(s) = schema {
            r = apply_reader_schema(r, s)?;
        }
        let paths = path.map(paths_of).transpose()?.unwrap_or_default();
        // PySpark's one-shot load sets format/schema/options on the reader itself.
        self.inner = r.clone();
        Ok(PyDataFrame::new(r.load_paths(paths)))
    }

    /// Read a table by name.
    fn table(&self, tableName: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.inner.clone().table(tableName)))
    }

    /// Read the CDC changes of a named table. Mirrors `DataFrameReader.changes`, which raises an
    /// `AnalysisException` of `_LEGACY_ERROR_TEMP_1189` for a reader of a schema.
    #[allow(non_snake_case)]
    fn changes(&self, py: Python<'_>, tableName: &str) -> PyResult<PyDataFrame> {
        if !self.inner.get_schema().is_empty() {
            return Err(changes_of_schema_error(py));
        }
        Ok(PyDataFrame::new(self.inner.clone().changes(tableName)))
    }

    /// Read JSON file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
    #[pyo3(signature = (path, schema=None, primitivesAsString=None, prefersDecimal=None, allowComments=None, allowUnquotedFieldNames=None, allowSingleQuotes=None, allowNumericLeadingZero=None, allowBackslashEscapingAnyCharacter=None, mode=None, columnNameOfCorruptRecord=None, dateFormat=None, timestampFormat=None, multiLine=None, allowUnquotedControlChars=None, lineSep=None, samplingRatio=None, dropFieldIfAllNull=None, encoding=None, locale=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, allowNonNumericNumbers=None, useUnsafeRow=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn json(
        &mut self,
        path: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
        primitivesAsString: Option<&Bound<'_, PyAny>>,
        prefersDecimal: Option<&Bound<'_, PyAny>>,
        allowComments: Option<&Bound<'_, PyAny>>,
        allowUnquotedFieldNames: Option<&Bound<'_, PyAny>>,
        allowSingleQuotes: Option<&Bound<'_, PyAny>>,
        allowNumericLeadingZero: Option<&Bound<'_, PyAny>>,
        allowBackslashEscapingAnyCharacter: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        columnNameOfCorruptRecord: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        multiLine: Option<&Bound<'_, PyAny>>,
        allowUnquotedControlChars: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        samplingRatio: Option<&Bound<'_, PyAny>>,
        dropFieldIfAllNull: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        locale: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        modifiedBefore: Option<&Bound<'_, PyAny>>,
        modifiedAfter: Option<&Bound<'_, PyAny>>,
        allowNonNumericNumbers: Option<&Bound<'_, PyAny>>,
        useUnsafeRow: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        if let Some(v) = schema {
            r = apply_reader_schema(r, v)?;
        }
        r = set_opt(r, "primitivesAsString", primitivesAsString)?;
        r = set_opt(r, "prefersDecimal", prefersDecimal)?;
        r = set_opt(r, "allowComments", allowComments)?;
        r = set_opt(r, "allowUnquotedFieldNames", allowUnquotedFieldNames)?;
        r = set_opt(r, "allowSingleQuotes", allowSingleQuotes)?;
        r = set_opt(r, "allowNumericLeadingZero", allowNumericLeadingZero)?;
        r = set_opt(
            r,
            "allowBackslashEscapingAnyCharacter",
            allowBackslashEscapingAnyCharacter,
        )?;
        r = set_opt(r, "mode", mode)?;
        r = set_opt(r, "columnNameOfCorruptRecord", columnNameOfCorruptRecord)?;
        r = set_opt(r, "dateFormat", dateFormat)?;
        r = set_opt(r, "timestampFormat", timestampFormat)?;
        r = set_opt(r, "multiLine", multiLine)?;
        r = set_opt(r, "allowUnquotedControlChars", allowUnquotedControlChars)?;
        r = set_opt(r, "lineSep", lineSep)?;
        r = set_opt(r, "samplingRatio", samplingRatio)?;
        r = set_opt(r, "dropFieldIfAllNull", dropFieldIfAllNull)?;
        r = set_opt(r, "encoding", encoding)?;
        r = set_opt(r, "locale", locale)?;
        r = set_opt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_opt(r, "recursiveFileLookup", recursiveFileLookup)?;
        r = set_opt(r, "modifiedBefore", modifiedBefore)?;
        r = set_opt(r, "modifiedAfter", modifiedAfter)?;
        r = set_opt(r, "allowNonNumericNumbers", allowNonNumericNumbers)?;
        r = set_opt(r, "useUnsafeRow", useUnsafeRow)?;
        self.read_or_parse(r, "json", spark_connect::plan::ParseFormat::Json, path)
    }

    /// Read Parquet file(s).
    #[pyo3(signature = (*paths, **options))]
    fn parquet(
        &mut self,
        paths: Vec<Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        // As PySpark's `parquet(*paths)`, of every path.
        let r = self.with_opts(options)?;
        let paths = path_list(&paths)?;
        self.commit(&r, "parquet");
        Ok(PyDataFrame::new(r.format("parquet").load_paths(paths)))
    }

    /// Read CSV file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
    #[pyo3(signature = (path, schema=None, sep=None, encoding=None, quote=None, escape=None, comment=None, header=None, inferSchema=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, maxMalformedLogPerPartition=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, charToEscapeQuoteEscaping=None, samplingRatio=None, enforceSchema=None, emptyValue=None, locale=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, unescapedQuoteHandling=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn csv(
        &mut self,
        path: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
        sep: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        quote: Option<&Bound<'_, PyAny>>,
        escape: Option<&Bound<'_, PyAny>>,
        comment: Option<&Bound<'_, PyAny>>,
        header: Option<&Bound<'_, PyAny>>,
        inferSchema: Option<&Bound<'_, PyAny>>,
        ignoreLeadingWhiteSpace: Option<&Bound<'_, PyAny>>,
        ignoreTrailingWhiteSpace: Option<&Bound<'_, PyAny>>,
        nullValue: Option<&Bound<'_, PyAny>>,
        nanValue: Option<&Bound<'_, PyAny>>,
        positiveInf: Option<&Bound<'_, PyAny>>,
        negativeInf: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        maxColumns: Option<&Bound<'_, PyAny>>,
        maxCharsPerColumn: Option<&Bound<'_, PyAny>>,
        maxMalformedLogPerPartition: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        columnNameOfCorruptRecord: Option<&Bound<'_, PyAny>>,
        multiLine: Option<&Bound<'_, PyAny>>,
        charToEscapeQuoteEscaping: Option<&Bound<'_, PyAny>>,
        samplingRatio: Option<&Bound<'_, PyAny>>,
        enforceSchema: Option<&Bound<'_, PyAny>>,
        emptyValue: Option<&Bound<'_, PyAny>>,
        locale: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        modifiedBefore: Option<&Bound<'_, PyAny>>,
        modifiedAfter: Option<&Bound<'_, PyAny>>,
        unescapedQuoteHandling: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        if let Some(v) = schema {
            r = apply_reader_schema(r, v)?;
        }
        r = set_opt(r, "sep", sep)?;
        r = set_opt(r, "encoding", encoding)?;
        r = set_opt(r, "quote", quote)?;
        r = set_opt(r, "escape", escape)?;
        r = set_opt(r, "comment", comment)?;
        r = set_opt(r, "header", header)?;
        r = set_opt(r, "inferSchema", inferSchema)?;
        r = set_opt(r, "ignoreLeadingWhiteSpace", ignoreLeadingWhiteSpace)?;
        r = set_opt(r, "ignoreTrailingWhiteSpace", ignoreTrailingWhiteSpace)?;
        r = set_opt(r, "nullValue", nullValue)?;
        r = set_opt(r, "nanValue", nanValue)?;
        r = set_opt(r, "positiveInf", positiveInf)?;
        r = set_opt(r, "negativeInf", negativeInf)?;
        r = set_opt(r, "dateFormat", dateFormat)?;
        r = set_opt(r, "timestampFormat", timestampFormat)?;
        r = set_opt(r, "maxColumns", maxColumns)?;
        r = set_opt(r, "maxCharsPerColumn", maxCharsPerColumn)?;
        r = set_opt(
            r,
            "maxMalformedLogPerPartition",
            maxMalformedLogPerPartition,
        )?;
        r = set_opt(r, "mode", mode)?;
        r = set_opt(r, "columnNameOfCorruptRecord", columnNameOfCorruptRecord)?;
        r = set_opt(r, "multiLine", multiLine)?;
        r = set_opt(r, "charToEscapeQuoteEscaping", charToEscapeQuoteEscaping)?;
        r = set_opt(r, "samplingRatio", samplingRatio)?;
        r = set_opt(r, "enforceSchema", enforceSchema)?;
        r = set_opt(r, "emptyValue", emptyValue)?;
        r = set_opt(r, "locale", locale)?;
        r = set_opt(r, "lineSep", lineSep)?;
        r = set_opt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_opt(r, "recursiveFileLookup", recursiveFileLookup)?;
        r = set_opt(r, "modifiedBefore", modifiedBefore)?;
        r = set_opt(r, "modifiedAfter", modifiedAfter)?;
        r = set_opt(r, "unescapedQuoteHandling", unescapedQuoteHandling)?;
        self.read_or_parse(r, "csv", spark_connect::plan::ParseFormat::Csv, path)
    }

    /// Read ORC file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
    #[pyo3(signature = (path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn orc(
        &mut self,
        path: &Bound<'_, PyAny>,
        mergeSchema: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        modifiedBefore: Option<&Bound<'_, PyAny>>,
        modifiedAfter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        r = set_opt(r, "mergeSchema", mergeSchema)?;
        r = set_opt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_opt(r, "recursiveFileLookup", recursiveFileLookup)?;
        r = set_opt(r, "modifiedBefore", modifiedBefore)?;
        r = set_opt(r, "modifiedAfter", modifiedAfter)?;
        let paths = paths_of(path)?;
        self.commit(&r, "orc");
        Ok(PyDataFrame::new(r.format("orc").load_paths(paths)))
    }

    /// Read TEXT file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
    #[pyo3(signature = (paths, wholetext=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn text(
        &mut self,
        paths: &Bound<'_, PyAny>,
        wholetext: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        modifiedBefore: Option<&Bound<'_, PyAny>>,
        modifiedAfter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        r = set_opt(r, "wholetext", wholetext)?;
        r = set_opt(r, "lineSep", lineSep)?;
        r = set_opt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_opt(r, "recursiveFileLookup", recursiveFileLookup)?;
        r = set_opt(r, "modifiedBefore", modifiedBefore)?;
        r = set_opt(r, "modifiedAfter", modifiedAfter)?;
        let paths = paths_of(paths)?;
        self.commit(&r, "text");
        Ok(PyDataFrame::new(r.format("text").load_paths(paths)))
    }

    /// Read XML file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
    #[pyo3(signature = (path, rowTag=None, schema=None, excludeAttribute=None, attributePrefix=None, valueTag=None, ignoreSurroundingSpaces=None, rowValidationXSDPath=None, ignoreNamespace=None, wildcardColName=None, encoding=None, inferSchema=None, nullValue=None, dateFormat=None, timestampFormat=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, samplingRatio=None, locale=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn xml(
        &mut self,
        path: &Bound<'_, PyAny>,
        rowTag: Option<&Bound<'_, PyAny>>,
        schema: Option<&Bound<'_, PyAny>>,
        excludeAttribute: Option<&Bound<'_, PyAny>>,
        attributePrefix: Option<&Bound<'_, PyAny>>,
        valueTag: Option<&Bound<'_, PyAny>>,
        ignoreSurroundingSpaces: Option<&Bound<'_, PyAny>>,
        rowValidationXSDPath: Option<&Bound<'_, PyAny>>,
        ignoreNamespace: Option<&Bound<'_, PyAny>>,
        wildcardColName: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        inferSchema: Option<&Bound<'_, PyAny>>,
        nullValue: Option<&Bound<'_, PyAny>>,
        dateFormat: Option<&Bound<'_, PyAny>>,
        timestampFormat: Option<&Bound<'_, PyAny>>,
        mode: Option<&Bound<'_, PyAny>>,
        columnNameOfCorruptRecord: Option<&Bound<'_, PyAny>>,
        multiLine: Option<&Bound<'_, PyAny>>,
        samplingRatio: Option<&Bound<'_, PyAny>>,
        locale: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        r = set_opt(r, "rowTag", rowTag)?;
        if let Some(v) = schema {
            r = apply_reader_schema(r, v)?;
        }
        r = set_opt(r, "excludeAttribute", excludeAttribute)?;
        r = set_opt(r, "attributePrefix", attributePrefix)?;
        r = set_opt(r, "valueTag", valueTag)?;
        r = set_opt(r, "ignoreSurroundingSpaces", ignoreSurroundingSpaces)?;
        r = set_opt(r, "rowValidationXSDPath", rowValidationXSDPath)?;
        r = set_opt(r, "ignoreNamespace", ignoreNamespace)?;
        r = set_opt(r, "wildcardColName", wildcardColName)?;
        r = set_opt(r, "encoding", encoding)?;
        r = set_opt(r, "inferSchema", inferSchema)?;
        r = set_opt(r, "nullValue", nullValue)?;
        r = set_opt(r, "dateFormat", dateFormat)?;
        r = set_opt(r, "timestampFormat", timestampFormat)?;
        r = set_opt(r, "mode", mode)?;
        r = set_opt(r, "columnNameOfCorruptRecord", columnNameOfCorruptRecord)?;
        r = set_opt(r, "multiLine", multiLine)?;
        r = set_opt(r, "samplingRatio", samplingRatio)?;
        r = set_opt(r, "locale", locale)?;
        self.read_or_parse(r, "xml", spark_connect::plan::ParseFormat::Xml, path)
    }

    /// Read from a JDBC source. Mirrors `DataFrameReader.jdbc(url, table, column=None,
    /// lowerBound=None, upperBound=None, numPartitions=None, predicates=None, properties=None)`,
    /// which keeps the format, then the options, on the reader:
    /// - of a `column`, it asserts the bounds and `numPartitions`, and reads the partitions of
    ///   the column, ignoring `predicates`;
    /// - else, it reads the partitions of `predicates`, ignoring the bounds and `numPartitions`.
    ///
    /// The column, the bounds, `numPartitions` and the values of `properties` are options, of any
    /// type (None skipped, bools lowercased).
    #[pyo3(signature = (url, table, column=None, lowerBound=None, upperBound=None, numPartitions=None, predicates=None, properties=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn jdbc(
        &mut self,
        url: &str,
        table: &str,
        column: Option<&Bound<'_, PyAny>>,
        lowerBound: Option<&Bound<'_, PyAny>>,
        upperBound: Option<&Bound<'_, PyAny>>,
        numPartitions: Option<&Bound<'_, PyAny>>,
        predicates: Option<Vec<String>>,
        properties: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        self.inner = self.inner.clone().format("jdbc");
        let mut r = self.inner.clone();
        let predicates = match column {
            Some(column) => {
                for (bound, name) in [
                    (lowerBound, "lowerBound"),
                    (upperBound, "upperBound"),
                    (numPartitions, "numPartitions"),
                ] {
                    if bound.is_none() {
                        return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
                            "{name} can not be None when ``column`` is specified"
                        )));
                    }
                }
                r = set_opt(r, "partitionColumn", Some(column))?;
                r = set_opt(r, "lowerBound", lowerBound)?;
                r = set_opt(r, "upperBound", upperBound)?;
                r = set_opt(r, "numPartitions", numPartitions)?;
                None
            }
            None => predicates,
        };
        if let Some(properties) = properties {
            for (key, value) in properties.iter() {
                r = set_opt(r, &key.str()?.to_cow()?, Some(&value))?;
            }
        }
        // PySpark's jdbc also leaves url/dbtable on the reader.
        self.inner = r.clone().option("url", url).option("dbtable", table);
        Ok(PyDataFrame::new(r.jdbc(url, table, predicates)))
    }
}
