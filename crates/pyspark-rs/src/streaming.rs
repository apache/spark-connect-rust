//! PyO3 wrappers for Structured Streaming.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use spark_connect::streaming::{
    DataStreamReader, DataStreamWriter, ListenerEventStream, StreamingQuery,
    StreamingQueryListener, StreamingQueryListenerEvent, StreamingQueryManager, Trigger,
};
use spark_connect::udf::PythonUDFPayload;
use std::collections::HashMap;
use std::sync::Arc;

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

/// Adapts a Python `StreamingQueryListener` to the native Rust listener trait: on each
/// event it acquires the GIL and calls the Python-side dispatch helper (which builds the
/// typed event object and invokes the right `onQuery*` callback).
struct PyListenerAdapter {
    listener: Py<PyAny>,
}

impl StreamingQueryListener for PyListenerAdapter {
    fn on_event(&self, event: &StreamingQueryListenerEvent) {
        Python::attach(|py| {
            let dispatch = py
                .import("pyspark.sql.streaming.query")
                .and_then(|m| m.getattr("_dispatch_listener_event"));
            if let Ok(dispatch) = dispatch {
                // Swallow listener/callback errors so one bad listener cannot kill the
                // dispatch thread (reference pyspark also isolates callback exceptions).
                let _ = dispatch.call1((
                    self.listener.bind(py),
                    event.event_type,
                    event.event_json.as_str(),
                ));
            }
        });
    }
}

/// Apply a named read option to a streaming reader when supplied (mirrors the batch
/// `set_opt` in readwriter.rs): `None` leaves the option unset (not the string "None");
/// booleans lowercase to "true"/"false" via `coerce_option_value`.
fn set_ropt(
    r: DataStreamReader,
    name: &str,
    v: Option<&Bound<'_, PyAny>>,
) -> PyResult<DataStreamReader> {
    match v {
        Some(x) => match crate::coerce_option_value(x)? {
            Some(sv) => Ok(r.option(name, &sv)),
            None => Ok(r),
        },
        None => Ok(r),
    }
}

/// Apply a `schema` argument (a StructType or a DDL string) to a streaming reader, as the
/// batch reader does.
fn apply_stream_schema(r: DataStreamReader, v: &Bound<'_, PyAny>) -> PyResult<DataStreamReader> {
    Ok(r.schema(crate::readwriter::reader_schema(v)?))
}

/// The path of `DataStreamReader.load`, as PySpark checks it: a non-empty str, else
/// `VALUE_NOT_NON_EMPTY_STR` of its `str()`.
fn load_path(path: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(text) = path.cast::<pyo3::types::PyString>() {
        let text = text.to_cow()?;
        if !text.trim().is_empty() {
            return Ok(text.into_owned());
        }
    }
    let shown = path.str()?.to_cow()?.into_owned();
    Err(crate::column::raise_pyspark(
        path.py(),
        "PySparkValueError",
        "VALUE_NOT_NON_EMPTY_STR",
        &[("arg_name", "path"), ("arg_value", &shown)],
    ))
}

/// The path of a format reader of `DataStreamReader`, such as `json`, as PySpark's takes it: a
/// str, else `NOT_EXPECTED_TYPE` of `path`, which it then loads as `load` does.
fn format_path(path: &Bound<'_, PyAny>) -> PyResult<String> {
    if !path.is_instance_of::<pyo3::types::PyString>() {
        return Err(crate::stat_arguments::not_expected_type(
            path, "path", "str",
        ));
    }
    load_path(path)
}

/// Coerce a `partitionBy` argument (a single column name or a list of names) to a
/// column-name vector, mirroring PySpark's `str | list[str]` acceptance.
fn extract_partition_cols(v: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(vec![s]);
    }
    v.extract::<Vec<String>>()
}

/// The column names of `partitionBy(*cols)` and `clusterBy(*cols)`, as PySpark's take them:
/// the names, or a list or tuple of them alone.
fn column_names(cols: &Bound<'_, PyTuple>) -> PyResult<Vec<String>> {
    if cols.len() == 1 {
        let only = cols.get_item(0)?;
        if only.is_instance_of::<pyo3::types::PyList>() || only.is_instance_of::<PyTuple>() {
            return only.extract();
        }
    }
    cols.extract()
}

/// The name of a query, as PySpark's `queryName` checks it: a non-empty str, else
/// `VALUE_NOT_NON_EMPTY_STR` of its `str()`.
fn query_name_of(name: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(text) = name.cast::<pyo3::types::PyString>() {
        let text = text.to_cow()?;
        if !text.trim().is_empty() {
            return Ok(text.into_owned());
        }
    }
    let shown = name.str()?.to_cow()?.into_owned();
    Err(crate::column::raise_pyspark(
        name.py(),
        "PySparkValueError",
        "VALUE_NOT_NON_EMPTY_STR",
        &[("arg_name", "queryName"), ("arg_value", &shown)],
    ))
}

/// Python wrapper for DataStreamReader.
///
/// Like PySpark's reader, the setters change this object and return it, and the
/// reader can be used again after a read: the core reader is a consuming builder, so
/// the setters replace the wrapped value in place and the reads work on a clone.
#[pyclass(name = "DataStreamReader", module = "pyspark.sql.streaming.readwriter")]
pub struct PyDataStreamReader {
    inner: DataStreamReader,
}

impl PyDataStreamReader {
    pub fn new(reader: DataStreamReader) -> Self {
        PyDataStreamReader { inner: reader }
    }
}

#[pymethods]
impl PyDataStreamReader {
    fn format<'py>(mut slf: PyRefMut<'py, Self>, source: &str) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner = slf.inner.clone().format(source);
        Ok(slf)
    }

    /// Set the schema: a StructType or a DDL string, as in pyspark.
    fn schema<'py>(
        mut slf: PyRefMut<'py, Self>,
        schema: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let schema = crate::readwriter::reader_schema(schema)?;
        slf.inner = slf.inner.clone().schema(&schema);
        Ok(slf)
    }

    fn option<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        // None -> option left unset; bools -> "true"/"false" (reference `to_str`).
        match crate::coerce_option_value(value)? {
            Some(v) => {
                slf.inner = slf.inner.clone().option(key, &v);
                Ok(slf)
            }
            None => Ok(slf),
        }
    }

    // Mirrors reference `DataStreamReader.options(**options)`: keyword args; None values
    // skipped, booleans lowercased.
    #[pyo3(signature = (**options))]
    fn options<'py>(
        mut slf: PyRefMut<'py, Self>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut opts = HashMap::new();
        if let Some(options) = options {
            for (k, v) in options.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    opts.insert(k.str()?.to_string(), val);
                }
            }
        }
        slf.inner = slf.inner.clone().options(opts);
        Ok(slf)
    }

    /// The configured source format (PySpark Connect's `DataStreamReader._format`).
    #[getter]
    fn _format(&self) -> Option<String> {
        self.inner.get_format().map(str::to_string)
    }

    /// The configured schema string, empty when unset (`DataStreamReader._schema`).
    #[getter]
    fn _schema(&self) -> String {
        self.inner.get_schema().to_string()
    }

    /// The configured read options (`DataStreamReader._options`); a copy.
    #[getter]
    fn _options(&self) -> HashMap<String, String> {
        self.inner.get_options().clone()
    }

    #[pyo3(signature = (path=None, format=None, schema=None, **options))]
    fn load(
        &mut self,
        path: Option<&Bound<'_, PyAny>>,
        format: Option<&str>,
        schema: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        if let Some(fmt) = format {
            r = r.format(fmt);
        }
        if let Some(v) = schema {
            r = apply_stream_schema(r, v)?;
        }
        if let Some(options) = options {
            let mut opts = HashMap::new();
            for (k, v) in options.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    opts.insert(k.str()?.to_string(), val);
                }
            }
            r = r.options(opts);
        }
        // PySpark's one-shot load sets format/schema/options on the reader itself.
        self.inner = r.clone();
        let path = path.map(load_path).transpose()?;
        Ok(PyDataFrame::new(r.load(path.as_deref())))
    }

    /// Set the source name (for checkpoint stability). Mirrors `DataStreamReader.name`,
    /// which raises `NOT_EXPECTED_TYPE` for a name that is not a str, and
    /// `INVALID_STREAMING_SOURCE_NAME` for one of other characters than ASCII letters, digits
    /// and underscores.
    fn name<'py>(
        mut slf: PyRefMut<'py, Self>,
        source_name: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let Ok(name) = source_name.cast::<pyo3::types::PyString>() else {
            return Err(crate::stat_arguments::not_expected_type(
                source_name,
                "source_name",
                "str",
            ));
        };
        let name = name.to_cow()?;
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(crate::column::raise_pyspark(
                source_name.py(),
                "PySparkValueError",
                "INVALID_STREAMING_SOURCE_NAME",
                &[("source_name", &name)],
            ));
        }
        slf.inner = slf.inner.clone().name(&name);
        Ok(slf)
    }

    /// Read the streaming CDC changes of a named table. Mirrors `DataStreamReader.changes`,
    /// which raises an `AnalysisException` of `_LEGACY_ERROR_TEMP_1189` for a reader of a
    /// schema.
    fn changes(&self, py: Python<'_>, tableName: &str) -> PyResult<PyDataFrame> {
        if !self.inner.get_schema().is_empty() {
            // Of the message of the JVM error class, which PySpark's error classes lack.
            let kwargs = PyDict::new(py);
            kwargs.set_item("errorClass", "_LEGACY_ERROR_TEMP_1189")?;
            let parameters = PyDict::new(py);
            parameters.set_item("operation", "changes")?;
            kwargs.set_item("messageParameters", parameters)?;
            let exception = py
                .import("pyspark.errors")?
                .getattr("AnalysisException")?
                .call(
                    ("User specified schema not supported with `changes`.",),
                    Some(&kwargs),
                )?;
            return Err(PyErr::from_value(exception));
        }
        Ok(PyDataFrame::new(self.inner.clone().changes(tableName)))
    }

    /// Load an XML streaming source. Mirrors `DataStreamReader.xml(...)`: set each named
    /// option, then `format("xml").load(path)`.
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
        let mut r = self.inner.clone().format("xml");
        r = set_ropt(r, "rowTag", rowTag)?;
        if let Some(v) = schema {
            r = apply_stream_schema(r, v)?;
        }
        r = set_ropt(r, "excludeAttribute", excludeAttribute)?;
        r = set_ropt(r, "attributePrefix", attributePrefix)?;
        r = set_ropt(r, "valueTag", valueTag)?;
        r = set_ropt(r, "ignoreSurroundingSpaces", ignoreSurroundingSpaces)?;
        r = set_ropt(r, "rowValidationXSDPath", rowValidationXSDPath)?;
        r = set_ropt(r, "ignoreNamespace", ignoreNamespace)?;
        r = set_ropt(r, "wildcardColName", wildcardColName)?;
        r = set_ropt(r, "encoding", encoding)?;
        r = set_ropt(r, "inferSchema", inferSchema)?;
        r = set_ropt(r, "nullValue", nullValue)?;
        r = set_ropt(r, "dateFormat", dateFormat)?;
        r = set_ropt(r, "timestampFormat", timestampFormat)?;
        r = set_ropt(r, "mode", mode)?;
        r = set_ropt(r, "columnNameOfCorruptRecord", columnNameOfCorruptRecord)?;
        r = set_ropt(r, "multiLine", multiLine)?;
        r = set_ropt(r, "samplingRatio", samplingRatio)?;
        r = set_ropt(r, "locale", locale)?;
        // Like PySpark, the options and the xml format stay on the reader.
        self.inner = r.clone();
        let path = format_path(path)?;
        Ok(PyDataFrame::new(r.load(Some(&path))))
    }

    fn table(&self, tableName: &str) -> PyResult<PyDataFrame> {
        let df = self.inner.clone().table(tableName);
        Ok(PyDataFrame::new(df))
    }

    #[pyo3(signature = (path, schema=None, primitivesAsString=None, prefersDecimal=None, allowComments=None, allowUnquotedFieldNames=None, allowSingleQuotes=None, allowNumericLeadingZero=None, allowBackslashEscapingAnyCharacter=None, mode=None, columnNameOfCorruptRecord=None, dateFormat=None, timestampFormat=None, multiLine=None, allowUnquotedControlChars=None, lineSep=None, locale=None, dropFieldIfAllNull=None, encoding=None, pathGlobFilter=None, recursiveFileLookup=None, allowNonNumericNumbers=None, useUnsafeRow=None))]
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
        locale: Option<&Bound<'_, PyAny>>,
        dropFieldIfAllNull: Option<&Bound<'_, PyAny>>,
        encoding: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        allowNonNumericNumbers: Option<&Bound<'_, PyAny>>,
        useUnsafeRow: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        if let Some(v) = schema {
            r = apply_stream_schema(r, v)?;
        }
        r = set_ropt(r, "primitivesAsString", primitivesAsString)?;
        r = set_ropt(r, "prefersDecimal", prefersDecimal)?;
        r = set_ropt(r, "allowComments", allowComments)?;
        r = set_ropt(r, "allowUnquotedFieldNames", allowUnquotedFieldNames)?;
        r = set_ropt(r, "allowSingleQuotes", allowSingleQuotes)?;
        r = set_ropt(r, "allowNumericLeadingZero", allowNumericLeadingZero)?;
        r = set_ropt(
            r,
            "allowBackslashEscapingAnyCharacter",
            allowBackslashEscapingAnyCharacter,
        )?;
        r = set_ropt(r, "mode", mode)?;
        r = set_ropt(r, "columnNameOfCorruptRecord", columnNameOfCorruptRecord)?;
        r = set_ropt(r, "dateFormat", dateFormat)?;
        r = set_ropt(r, "timestampFormat", timestampFormat)?;
        r = set_ropt(r, "multiLine", multiLine)?;
        r = set_ropt(r, "allowUnquotedControlChars", allowUnquotedControlChars)?;
        r = set_ropt(r, "lineSep", lineSep)?;
        r = set_ropt(r, "locale", locale)?;
        r = set_ropt(r, "dropFieldIfAllNull", dropFieldIfAllNull)?;
        r = set_ropt(r, "encoding", encoding)?;
        r = set_ropt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_ropt(r, "recursiveFileLookup", recursiveFileLookup)?;
        r = set_ropt(r, "allowNonNumericNumbers", allowNonNumericNumbers)?;
        r = set_ropt(r, "useUnsafeRow", useUnsafeRow)?;
        // Like PySpark, the options and the format stay on the reader.
        self.inner = r.clone().format("json");
        let path = format_path(path)?;
        Ok(PyDataFrame::new(r.json(&path)))
    }

    #[pyo3(signature = (path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None, datetimeRebaseMode=None, int96RebaseMode=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn parquet(
        &mut self,
        path: &Bound<'_, PyAny>,
        mergeSchema: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        datetimeRebaseMode: Option<&Bound<'_, PyAny>>,
        int96RebaseMode: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        r = set_ropt(r, "mergeSchema", mergeSchema)?;
        r = set_ropt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_ropt(r, "recursiveFileLookup", recursiveFileLookup)?;
        r = set_ropt(r, "datetimeRebaseMode", datetimeRebaseMode)?;
        r = set_ropt(r, "int96RebaseMode", int96RebaseMode)?;
        // Like PySpark, the options and the format stay on the reader.
        self.inner = r.clone().format("parquet");
        let path = format_path(path)?;
        Ok(PyDataFrame::new(r.parquet(&path)))
    }

    #[pyo3(signature = (path, schema=None, sep=None, encoding=None, quote=None, escape=None, comment=None, header=None, inferSchema=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, maxMalformedLogPerPartition=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, charToEscapeQuoteEscaping=None, enforceSchema=None, emptyValue=None, locale=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, unescapedQuoteHandling=None))]
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
        enforceSchema: Option<&Bound<'_, PyAny>>,
        emptyValue: Option<&Bound<'_, PyAny>>,
        locale: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        unescapedQuoteHandling: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        if let Some(v) = schema {
            r = apply_stream_schema(r, v)?;
        }
        r = set_ropt(r, "sep", sep)?;
        r = set_ropt(r, "encoding", encoding)?;
        r = set_ropt(r, "quote", quote)?;
        r = set_ropt(r, "escape", escape)?;
        r = set_ropt(r, "comment", comment)?;
        r = set_ropt(r, "header", header)?;
        r = set_ropt(r, "inferSchema", inferSchema)?;
        r = set_ropt(r, "ignoreLeadingWhiteSpace", ignoreLeadingWhiteSpace)?;
        r = set_ropt(r, "ignoreTrailingWhiteSpace", ignoreTrailingWhiteSpace)?;
        r = set_ropt(r, "nullValue", nullValue)?;
        r = set_ropt(r, "nanValue", nanValue)?;
        r = set_ropt(r, "positiveInf", positiveInf)?;
        r = set_ropt(r, "negativeInf", negativeInf)?;
        r = set_ropt(r, "dateFormat", dateFormat)?;
        r = set_ropt(r, "timestampFormat", timestampFormat)?;
        r = set_ropt(r, "maxColumns", maxColumns)?;
        r = set_ropt(r, "maxCharsPerColumn", maxCharsPerColumn)?;
        r = set_ropt(
            r,
            "maxMalformedLogPerPartition",
            maxMalformedLogPerPartition,
        )?;
        r = set_ropt(r, "mode", mode)?;
        r = set_ropt(r, "columnNameOfCorruptRecord", columnNameOfCorruptRecord)?;
        r = set_ropt(r, "multiLine", multiLine)?;
        r = set_ropt(r, "charToEscapeQuoteEscaping", charToEscapeQuoteEscaping)?;
        r = set_ropt(r, "enforceSchema", enforceSchema)?;
        r = set_ropt(r, "emptyValue", emptyValue)?;
        r = set_ropt(r, "locale", locale)?;
        r = set_ropt(r, "lineSep", lineSep)?;
        r = set_ropt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_ropt(r, "recursiveFileLookup", recursiveFileLookup)?;
        r = set_ropt(r, "unescapedQuoteHandling", unescapedQuoteHandling)?;
        // Like PySpark, the options and the format stay on the reader.
        self.inner = r.clone().format("csv");
        let path = format_path(path)?;
        Ok(PyDataFrame::new(r.csv(&path)))
    }

    #[pyo3(signature = (path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None))]
    #[allow(non_snake_case)]
    fn orc(
        &mut self,
        path: &Bound<'_, PyAny>,
        mergeSchema: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        r = set_ropt(r, "mergeSchema", mergeSchema)?;
        r = set_ropt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_ropt(r, "recursiveFileLookup", recursiveFileLookup)?;
        // Like PySpark, the options and the format stay on the reader.
        self.inner = r.clone().format("orc");
        let path = format_path(path)?;
        Ok(PyDataFrame::new(r.orc(&path)))
    }

    #[pyo3(signature = (path, wholetext=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None))]
    #[allow(non_snake_case)]
    fn text(
        &mut self,
        path: &Bound<'_, PyAny>,
        wholetext: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.inner.clone();
        r = set_ropt(r, "wholetext", wholetext)?;
        r = set_ropt(r, "lineSep", lineSep)?;
        r = set_ropt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_ropt(r, "recursiveFileLookup", recursiveFileLookup)?;
        // Like PySpark, the options and the format stay on the reader.
        self.inner = r.clone().format("text");
        let path = format_path(path)?;
        Ok(PyDataFrame::new(r.text(&path)))
    }
}

/// Python wrapper for Trigger.
#[pyclass(
    name = "Trigger",
    module = "pyspark.sql.streaming.readwriter",
    from_py_object
)]
#[derive(Clone)]
pub struct PyTrigger {
    inner: Trigger,
}

impl PyTrigger {
    pub fn new(trigger: Trigger) -> Self {
        PyTrigger { inner: trigger }
    }

    pub fn get(&self) -> Trigger {
        self.inner.clone()
    }
}

#[pymethods]
impl PyTrigger {
    #[staticmethod]
    fn processingTime(interval: &str) -> PyTrigger {
        PyTrigger::new(Trigger::ProcessingTime(interval.to_string()))
    }

    #[staticmethod]
    fn once() -> PyTrigger {
        PyTrigger::new(Trigger::Once)
    }

    #[staticmethod]
    fn availableNow() -> PyTrigger {
        PyTrigger::new(Trigger::AvailableNow)
    }

    #[staticmethod]
    fn continuous(interval: &str) -> PyTrigger {
        PyTrigger::new(Trigger::Continuous(interval.to_string()))
    }
}

/// Python wrapper for DataStreamWriter.
#[pyclass(name = "DataStreamWriter", module = "pyspark.sql.streaming.readwriter")]
pub struct PyDataStreamWriter {
    inner: DataStreamWriter,
}

impl PyDataStreamWriter {
    pub fn new(writer: DataStreamWriter) -> Self {
        PyDataStreamWriter { inner: writer }
    }

    /// Apply the one-shot writer options accepted by `start`/`toTable`
    /// (`format`, `outputMode`, `partitionBy`, `queryName`, `**options`) before
    /// terminating the stream. Mirrors the reference, where these are keyword args.
    fn apply_write_opts(
        &mut self,
        format: Option<&str>,
        output_mode: Option<&str>,
        partition_by: Option<&Bound<'_, PyAny>>,
        query_name: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<DataStreamWriter> {
        let mut w = self.inner.clone();
        if let Some(fmt) = format {
            w = w.format(fmt);
        }
        if let Some(m) = output_mode {
            w = w.output_mode(m);
        }
        if let Some(pb) = partition_by {
            let cols = extract_partition_cols(pb)?;
            let refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
            w = w.partition_by(refs);
        }
        if let Some(qn) = query_name {
            w = w.query_name(&query_name_of(qn)?);
        }
        if let Some(options) = options {
            let mut opts = HashMap::new();
            for (k, v) in options.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    opts.insert(k.str()?.to_string(), val);
                }
            }
            w = w.options(opts);
        }
        // PySpark's start/toTable set these on the writer itself before starting.
        self.inner = w.clone();
        Ok(w)
    }
}

#[pymethods]
impl PyDataStreamWriter {
    fn outputMode<'py>(
        mut slf: PyRefMut<'py, Self>,
        outputMode: &str,
    ) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner = slf.inner.clone().output_mode(outputMode);
        Ok(slf)
    }

    fn format<'py>(mut slf: PyRefMut<'py, Self>, source: &str) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner = slf.inner.clone().format(source);
        Ok(slf)
    }

    fn option<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        // None -> option left unset; bools -> "true"/"false" (reference `to_str`).
        match crate::coerce_option_value(value)? {
            Some(v) => {
                slf.inner = slf.inner.clone().option(key, &v);
                Ok(slf)
            }
            None => Ok(slf),
        }
    }

    // Mirrors reference `DataStreamWriter.options(**options)`: keyword args; None values
    // skipped, booleans lowercased.
    #[pyo3(signature = (**options))]
    fn options<'py>(
        mut slf: PyRefMut<'py, Self>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let mut opts = HashMap::new();
        if let Some(options) = options {
            for (k, v) in options.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    opts.insert(k.str()?.to_string(), val);
                }
            }
        }
        slf.inner = slf.inner.clone().options(opts);
        Ok(slf)
    }

    /// Partition by the column names, or a list or tuple of them alone, as PySpark's.
    #[pyo3(signature = (*cols))]
    fn partitionBy<'py>(
        mut slf: PyRefMut<'py, Self>,
        cols: &Bound<'_, PyTuple>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let cols = column_names(cols)?;
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        slf.inner = slf.inner.clone().partition_by(col_refs);
        Ok(slf)
    }

    /// Cluster by the column names, or a list or tuple of them alone, as PySpark's.
    #[pyo3(signature = (*cols))]
    fn clusterBy<'py>(
        mut slf: PyRefMut<'py, Self>,
        cols: &Bound<'_, PyTuple>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let cols = column_names(cols)?;
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        slf.inner = slf.inner.clone().cluster_by(col_refs);
        Ok(slf)
    }

    /// Name the query, a non-empty str, else `VALUE_NOT_NON_EMPTY_STR`, as PySpark's.
    fn queryName<'py>(
        mut slf: PyRefMut<'py, Self>,
        queryName: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let name = query_name_of(queryName)?;
        slf.inner = slf.inner.clone().query_name(&name);
        Ok(slf)
    }

    /// `DataStreamWriter.trigger(...)`, as PySpark's: exactly one of `processingTime`, `once`,
    /// `continuous`, `availableNow` and `realTime`, else `ONLY_ALLOW_SINGLE_TRIGGER`; an
    /// interval or duration is a non-empty string, which is stripped, else
    /// `VALUE_NOT_NON_EMPTY_STR`, and `once` and `availableNow` are True, else
    /// `VALUE_NOT_ALLOWED`.
    #[pyo3(signature = (*, processingTime=None, once=None, continuous=None, availableNow=None, realTime=None))]
    fn trigger<'py>(
        mut slf: PyRefMut<'py, Self>,
        processingTime: Option<Bound<'py, PyAny>>,
        once: Option<Bound<'py, PyAny>>,
        continuous: Option<Bound<'py, PyAny>>,
        availableNow: Option<Bound<'py, PyAny>>,
        realTime: Option<Bound<'py, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        use crate::column::raise_pyspark;
        let py = slf.py();
        let given = [
            &processingTime,
            &once,
            &continuous,
            &availableNow,
            &realTime,
        ]
        .iter()
        .filter(|value| value.is_some())
        .count();
        if given != 1 {
            return Err(raise_pyspark(
                py,
                "PySparkValueError",
                "ONLY_ALLOW_SINGLE_TRIGGER",
                &[],
            ));
        }
        let interval = |value: &Bound<'py, PyAny>, name: &str| -> PyResult<String> {
            let text = match value.cast::<pyo3::types::PyString>() {
                Ok(text) => text.to_cow()?.trim().to_string(),
                Err(_) => String::new(),
            };
            if text.is_empty() {
                return Err(raise_pyspark(
                    py,
                    "PySparkValueError",
                    "VALUE_NOT_NON_EMPTY_STR",
                    &[("arg_name", name), ("arg_value", &value.str()?.to_string())],
                ));
            }
            Ok(text)
        };
        // As PySpark's `once is not True`.
        let true_flag = |value: &Bound<'py, PyAny>, name: &str| -> PyResult<()> {
            if value.is(pyo3::types::PyBool::new(py, true)) {
                Ok(())
            } else {
                Err(raise_pyspark(
                    py,
                    "PySparkValueError",
                    "VALUE_NOT_ALLOWED",
                    &[("arg_name", name), ("allowed_values", "[True]")],
                ))
            }
        };
        let trigger = if let Some(value) = &processingTime {
            Trigger::ProcessingTime(interval(value, "processingTime")?)
        } else if let Some(value) = &once {
            true_flag(value, "once")?;
            Trigger::Once
        } else if let Some(value) = &continuous {
            Trigger::Continuous(interval(value, "continuous")?)
        } else if let Some(value) = &realTime {
            Trigger::RealTime(interval(value, "realTime")?)
        } else {
            if let Some(value) = &availableNow {
                true_flag(value, "availableNow")?;
            }
            Trigger::AvailableNow
        };
        slf.inner = slf.inner.clone().trigger(trigger);
        Ok(slf)
    }

    #[pyo3(signature = (path=None, format=None, outputMode=None, partitionBy=None, queryName=None, **options))]
    #[allow(non_snake_case)]
    fn start(
        &mut self,
        path: Option<&str>,
        format: Option<&str>,
        outputMode: Option<&str>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        queryName: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyStreamingQuery> {
        let writer = self.apply_write_opts(format, outputMode, partitionBy, queryName, options)?;
        // Memory/console/foreach sinks take no path; the core treats "" as unset.
        let query = writer.start(path.unwrap_or("")).to_pyerr()?;
        Ok(PyStreamingQuery::new(query))
    }

    #[pyo3(signature = (tableName, format=None, outputMode=None, partitionBy=None, queryName=None, **options))]
    #[allow(non_snake_case)]
    fn toTable(
        &mut self,
        tableName: &str,
        format: Option<&str>,
        outputMode: Option<&str>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        queryName: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyStreamingQuery> {
        let writer = self.apply_write_opts(format, outputMode, partitionBy, queryName, options)?;
        let query = writer.to_table(tableName).to_pyerr()?;
        Ok(PyStreamingQuery::new(query))
    }

    /// `DataStreamWriter.foreachBatch(func)`: cloudpickle the `(batch_df, batch_id)`
    /// function (with PySpark's `CloudPickleSerializer`) and attach it as the
    /// foreach-batch sink. Pickling happens here so the Python skin can re-export this
    /// class directly rather than subclassing a (non-subclassable) PyO3 type.
    fn foreachBatch<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let command = streaming_function_command(py, func, "foreachBatch")?;
        let payload = PythonUDFPayload::new(
            spark_connect::types::DataType::Struct { fields: vec![] },
            0, // eval type is unused for the streaming foreach sinks
            command,
            crate::dataframe::py_version(py),
        );
        slf.inner = slf.inner.clone().foreach_batch(payload);
        Ok(slf)
    }

    /// `DataStreamWriter.foreach(f)`, as PySpark's: the function of the partition and the
    /// rows of each partition of `_construct_foreach_function(f)`, which raises
    /// `ATTRIBUTE_NOT_CALLABLE` for an object without a callable `process`, in the command
    /// `(func, None, serializer, serializer)` of `AutoBatchedSerializer(CPickleSerializer())`,
    /// which the worker of the server deserializes with its own `pyspark.serializers`. Built
    /// here to avoid a Python subclass of the PyO3 class.
    fn foreach<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'_>,
        f: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let func = py
            .import("pyspark.sql.streaming.readwriter")?
            .getattr("_construct_foreach_function")?
            .call1((f,))?;
        let serializers = py.import("pyspark.serializers")?;
        let cpickle = serializers.getattr("CPickleSerializer")?.call0()?;
        let serializer = serializers
            .getattr("AutoBatchedSerializer")?
            .call1((cpickle,))?;
        // (func, return_type=None, input_serializer, output_serializer) — the shape the
        // worker's foreach runner expects; the same serializer instance is used twice.
        let command_tuple = pyo3::types::PyTuple::new(
            py,
            [
                func,
                py.None().into_bound(py),
                serializer.clone(),
                serializer,
            ],
        )?;
        let command = streaming_function_command(py, command_tuple.as_any(), "foreach")?;
        let payload = PythonUDFPayload::new(
            spark_connect::types::DataType::Struct { fields: vec![] },
            0,
            command,
            crate::dataframe::py_version(py),
        );
        slf.inner = slf.inner.clone().foreach(payload);
        Ok(slf)
    }
}

/// The progress of a query from its JSON, as PySpark's
/// `StreamingQueryProgress.fromJson(json.loads(progress))`.
fn progress_from_json<'py>(py: Python<'py>, json: &str) -> PyResult<Bound<'py, PyAny>> {
    let loaded = py.import("json")?.call_method1("loads", (json,))?;
    py.import("pyspark.sql.streaming.listener")?
        .getattr("StreamingQueryProgress")?
        .call_method1("fromJson", (loaded,))
}

/// The timeout of `awaitTermination` and `awaitAnyTermination`, in seconds, as PySpark's take
/// it: an int or a float, which must be positive, else `VALUE_NOT_POSITIVE` of the name of its
/// type.
fn positive_timeout(timeout: Option<&Bound<'_, PyAny>>) -> PyResult<Option<f64>> {
    let Some(timeout) = timeout else {
        return Ok(None);
    };
    let number = timeout.is_instance_of::<pyo3::types::PyInt>()
        || timeout.is_instance_of::<pyo3::types::PyFloat>();
    match number.then(|| timeout.extract::<f64>()).transpose()? {
        Some(seconds) if seconds > 0.0 => Ok(Some(seconds)),
        _ => {
            let type_name = timeout.get_type().name()?.to_string();
            Err(crate::column::raise_pyspark(
                timeout.py(),
                "PySparkValueError",
                "VALUE_NOT_POSITIVE",
                &[("arg_name", "timeout"), ("arg_value", &type_name)],
            ))
        }
    }
}

/// A bool argument that PySpark sets on a proto bool field, which takes a bool or an int, and
/// raises the `TypeError` of protobuf for another value.
pub(crate) struct ProtoBool(pub(crate) bool);

impl<'a, 'py> FromPyObject<'a, 'py> for ProtoBool {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if obj.is_instance_of::<pyo3::types::PyInt>() {
            return Ok(ProtoBool(obj.is_truthy()?));
        }
        let shown = obj.repr()?;
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "Cannot set spark.connect.StreamingQueryCommand.ExplainCommand.extended to {shown}: \
             {shown} has type {}, but expected one of: (<class 'bool'>, <class 'int'>)",
            obj.get_type().repr()?
        )))
    }
}

/// The command of the function of `foreach` or `foreachBatch`, as PySpark's: of
/// `CloudPickleSerializer().dumps`, which raises a `pickle.PicklingError` for an object that
/// does not pickle, as `STREAMING_CONNECT_SERIALIZATION_ERROR` of `name`.
fn streaming_function_command(
    py: Python<'_>,
    function: &Bound<'_, PyAny>,
    name: &str,
) -> PyResult<Vec<u8>> {
    let serializer = py
        .import("pyspark.serializers")?
        .getattr("CloudPickleSerializer")?
        .call0()?;
    match serializer.call_method1("dumps", (function,)) {
        Ok(command) => command.extract(),
        Err(error) if error.matches(py, py.import("pickle")?.getattr("PicklingError")?)? => {
            let raised = crate::column::raise_pyspark(
                py,
                "PySparkPicklingError",
                "STREAMING_CONNECT_SERIALIZATION_ERROR",
                &[("name", name)],
            );
            raised.set_cause(py, Some(error));
            Err(raised)
        }
        Err(error) => Err(error),
    }
}

/// Python wrapper for StreamingQuery.
#[pyclass(name = "StreamingQuery", module = "pyspark.sql.streaming.query")]
pub struct PyStreamingQuery {
    inner: StreamingQuery,
}

impl PyStreamingQuery {
    pub fn new(query: StreamingQuery) -> Self {
        PyStreamingQuery { inner: query }
    }
}

#[pymethods]
impl PyStreamingQuery {
    #[getter]
    fn id(&self) -> String {
        self.inner.id().to_string()
    }

    #[getter]
    fn runId(&self) -> String {
        self.inner.run_id().to_string()
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.inner.name().map(|s| s.to_string())
    }

    #[getter]
    fn isActive(&self) -> PyResult<bool> {
        self.inner.is_active().to_pyerr()
    }

    /// The status of the query, as PySpark's: a dict of its `message`, `isDataAvailable` and
    /// `isTriggerActive`.
    #[getter]
    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let status = self.inner.status().to_pyerr()?;
        let dict = PyDict::new(py);
        dict.set_item("message", status.status_message)?;
        dict.set_item("isDataAvailable", status.is_data_available)?;
        dict.set_item("isTriggerActive", status.is_trigger_active)?;
        Ok(dict)
    }

    fn stop(&self) -> PyResult<()> {
        self.inner.stop().to_pyerr()
    }

    #[pyo3(signature = (timeout=None))]
    fn awaitTermination(&self, timeout: Option<&Bound<'_, PyAny>>) -> PyResult<Option<bool>> {
        let timeout = positive_timeout(timeout)?;
        self.inner.await_termination(timeout).to_pyerr()
    }

    /// The last progress of the query, as PySpark's `StreamingQueryProgress`, or None.
    #[getter]
    fn lastProgress<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self.inner.last_progress().to_pyerr()? {
            Some(json) => progress_from_json(py, &json).map(Some),
            None => Ok(None),
        }
    }

    /// The recent progress of the query, each as PySpark's `StreamingQueryProgress`.
    #[getter]
    fn recentProgress<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        self.inner
            .recent_progress()
            .to_pyerr()?
            .iter()
            .map(|json| progress_from_json(py, json))
            .collect()
    }

    fn processAllAvailable(&self) -> PyResult<()> {
        self.inner.process_all_available().to_pyerr()
    }

    /// Print the plan of the query with Python's `print`, as PySpark's
    /// `StreamingQuery.explain(extended=False)`.
    #[pyo3(signature = (extended=ProtoBool(false)))]
    fn explain(&self, py: Python<'_>, extended: ProtoBool) -> PyResult<()> {
        let plan = py
            .detach(|| self.inner.explain(extended.0).map_err(Box::new))
            .to_pyerr()?;
        py.import("builtins")?.call_method1("print", (plan,))?;
        Ok(())
    }

    /// The exception of a failed query, as PySpark's: the `StreamingQueryException` of
    /// `pyspark.errors.exceptions.connect`, of the message of the server without the type of
    /// its Java exception, and with its JVM stack trace, and of the error class as its reason.
    fn exception(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Some(exception) = self.inner.exception().to_pyerr()? else {
            return Ok(None);
        };
        // The message is the `toString` of the Java StreamingQueryException: drop its type.
        let mut message = match exception.message.split_once(": ") {
            Some((_, message)) => message.to_string(),
            None => exception.message,
        };
        if let Some(stack_trace) = exception.stack_trace {
            message.push_str(&format!("\n\nJVM stacktrace:\n{stack_trace}"));
        }
        let class = py
            .import("pyspark.errors.exceptions.connect")?
            .getattr("StreamingQueryException")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("reason", exception.error_class)?;
        Ok(Some(class.call((message,), Some(&kwargs))?.unbind()))
    }
}

/// Python wrapper for StreamingQueryManager.
#[pyclass(name = "StreamingQueryManager", module = "pyspark.sql.streaming.query")]
pub struct PyStreamingQueryManager {
    inner: StreamingQueryManager,
}

impl PyStreamingQueryManager {
    pub fn new(manager: StreamingQueryManager) -> Self {
        PyStreamingQueryManager { inner: manager }
    }
}

#[pymethods]
impl PyStreamingQueryManager {
    #[getter]
    fn active(&self) -> PyResult<Vec<PyStreamingQuery>> {
        let queries = self.inner.active().to_pyerr()?;
        Ok(queries.into_iter().map(PyStreamingQuery::new).collect())
    }

    fn get(&self, id: &str) -> PyResult<Option<PyStreamingQuery>> {
        let query = self.inner.get(id).to_pyerr()?;
        Ok(query.map(PyStreamingQuery::new))
    }

    #[pyo3(signature = (timeout=None))]
    fn awaitAnyTermination(&self, timeout: Option<&Bound<'_, PyAny>>) -> PyResult<Option<bool>> {
        let timeout = positive_timeout(timeout)?;
        self.inner.await_any_termination(timeout).to_pyerr()
    }

    fn resetTerminated(&self) -> PyResult<()> {
        self.inner.reset_terminated().to_pyerr()
    }

    /// Register a client-side listener object (with onQueryStarted/Progress/Idle/
    /// Terminated callbacks). Mirrors `StreamingQueryManager.addListener`. The native
    /// Rust bus streams events and dispatches to this listener; the assigned id is
    /// stashed on the listener so `removeListener` can find it.
    fn addListener(&self, py: Python<'_>, listener: Py<PyAny>) -> PyResult<()> {
        let adapter = Arc::new(PyListenerAdapter {
            listener: listener.clone_ref(py),
        });
        let id = self.inner.add_listener(adapter).to_pyerr()?;
        listener.bind(py).setattr("_rust_listener_id", id)?;
        Ok(())
    }

    /// Remove a previously-added client-side listener object. Mirrors
    /// `StreamingQueryManager.removeListener`.
    fn removeListener(&self, py: Python<'_>, listener: Py<PyAny>) -> PyResult<()> {
        let bound = listener.bind(py);
        if let Ok(id_obj) = bound.getattr("_rust_listener_id") {
            let id: String = id_obj.extract()?;
            self.inner.remove_listener(&id).to_pyerr()?;
            let _ = bound.delattr("_rust_listener_id");
        }
        Ok(())
    }

    /// Remove all client-side listeners and stop the dispatch thread. Mirrors
    /// `StreamingQueryManager.close`.
    fn close(&self) -> PyResult<()> {
        self.inner.close().to_pyerr()
    }

    fn streamListenerEvents(&self) -> PyResult<PyListenerEventStream> {
        let stream = self.inner.listener_event_stream().to_pyerr()?;
        Ok(PyListenerEventStream::new(stream))
    }
}

/// Python wrapper for ListenerEventStream.
/// Implements `__iter__` and `__next__` to yield (event_type: i32, event_json: String) tuples.
#[pyclass(name = "ListenerEventStream", module = "pyspark.sql.streaming.query")]
pub struct PyListenerEventStream {
    inner: Option<ListenerEventStream>,
}

impl PyListenerEventStream {
    pub fn new(stream: ListenerEventStream) -> Self {
        PyListenerEventStream {
            inner: Some(stream),
        }
    }
}

#[pymethods]
impl PyListenerEventStream {
    fn __iter__(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<(i32, String)>> {
        let Some(stream) = self.inner.as_mut() else {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "ListenerEventStream already consumed",
            ));
        };
        // The listener bus blocks in `next()` waiting for the server's next event
        // (arbitrarily far apart on a live query), so release the GIL across it —
        // otherwise the daemon thread would freeze the whole interpreter between
        // events. Mirrors `PyLocalRowIterator::__next__`.
        match py.detach(|| stream.next()) {
            Some(Ok((event_type, event_json))) => Ok(Some((event_type, event_json))),
            Some(Err(e)) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error reading listener events: {}",
                e
            ))),
            None => Ok(None),
        }
    }
}
