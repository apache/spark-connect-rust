//! PyO3 wrappers for Structured Streaming.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::streaming::{
    DataStreamReader, DataStreamWriter, ListenerEventStream, StreamingQuery,
    StreamingQueryException, StreamingQueryListener, StreamingQueryListenerEvent,
    StreamingQueryManager, StreamingQueryStatus, Trigger,
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

/// Python wrapper for DataStreamReader.
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

/// Apply a `schema` argument (a DDL string or a StructType) to a streaming reader,
/// mirroring the batch `apply_reader_schema`.
fn apply_stream_schema(r: DataStreamReader, v: &Bound<'_, PyAny>) -> PyResult<DataStreamReader> {
    let dt = crate::types::py_to_data_type(v)?;
    Ok(r.schema(dt.simple_string()))
}

/// Coerce a `partitionBy` argument (a single column name or a list of names) to a
/// column-name vector, mirroring PySpark's `str | list[str]` acceptance.
fn extract_partition_cols(v: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(s) = v.extract::<String>() {
        return Ok(vec![s]);
    }
    v.extract::<Vec<String>>()
}

#[pyclass(name = "DataStreamReader", module = "pyspark.sql.streaming.readwriter")]
pub struct PyDataStreamReader {
    inner: Option<DataStreamReader>,
}

impl PyDataStreamReader {
    pub fn new(reader: DataStreamReader) -> Self {
        PyDataStreamReader {
            inner: Some(reader),
        }
    }

    fn take(&mut self) -> PyResult<DataStreamReader> {
        self.inner.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("DataStreamReader already consumed")
        })
    }
}

#[pymethods]
impl PyDataStreamReader {
    fn format(&mut self, source: &str) -> PyResult<PyDataStreamReader> {
        Ok(PyDataStreamReader {
            inner: Some(self.take()?.format(source)),
        })
    }

    fn schema(&mut self, schema: &str) -> PyResult<PyDataStreamReader> {
        Ok(PyDataStreamReader {
            inner: Some(self.take()?.schema(schema)),
        })
    }

    fn option(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<PyDataStreamReader> {
        // None -> option left unset; bools -> "true"/"false" (reference `to_str`).
        match crate::coerce_option_value(value)? {
            Some(v) => Ok(PyDataStreamReader {
                inner: Some(self.take()?.option(key, &v)),
            }),
            None => Ok(PyDataStreamReader {
                inner: Some(self.take()?),
            }),
        }
    }

    // Mirrors reference `DataStreamReader.options(**options)`: keyword args; None values
    // skipped, booleans lowercased.
    #[pyo3(signature = (**options))]
    fn options(&mut self, options: Option<&Bound<'_, PyDict>>) -> PyResult<PyDataStreamReader> {
        let mut opts = HashMap::new();
        if let Some(options) = options {
            for (k, v) in options.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    opts.insert(k.str()?.to_string(), val);
                }
            }
        }
        Ok(PyDataStreamReader {
            inner: Some(self.take()?.options(opts)),
        })
    }

    #[pyo3(signature = (path=None, format=None, schema=None, **options))]
    fn load(
        &mut self,
        path: Option<&str>,
        format: Option<&str>,
        schema: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.take()?;
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
        Ok(PyDataFrame::new(r.load(path)))
    }

    /// Set the source name (for checkpoint stability). Mirrors `DataStreamReader.name`,
    /// which validates the name is a non-empty `[A-Za-z0-9_]+` string.
    fn name(&mut self, source_name: &str) -> PyResult<PyDataStreamReader> {
        if source_name.is_empty()
            || !source_name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Invalid streaming source name: {source_name:?}; only ASCII letters, digits, and underscores are allowed"
            )));
        }
        Ok(PyDataStreamReader {
            inner: Some(self.take()?.name(source_name)),
        })
    }

    /// Read the streaming CDC changes of a named table. Mirrors `DataStreamReader.changes`.
    fn changes(&mut self, tableName: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take()?.changes(tableName)))
    }

    /// Load an XML streaming source. Mirrors `DataStreamReader.xml(...)`: set each named
    /// option, then `format("xml").load(path)`.
    #[pyo3(signature = (path, rowTag=None, schema=None, excludeAttribute=None, attributePrefix=None, valueTag=None, ignoreSurroundingSpaces=None, rowValidationXSDPath=None, ignoreNamespace=None, wildcardColName=None, encoding=None, inferSchema=None, nullValue=None, dateFormat=None, timestampFormat=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, samplingRatio=None, locale=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn xml(
        &mut self,
        path: &str,
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
        let mut r = self.take()?.format("xml");
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
        Ok(PyDataFrame::new(r.load(Some(path))))
    }

    fn table(&mut self, tableName: &str) -> PyResult<PyDataFrame> {
        let df = self.take()?.table(tableName);
        Ok(PyDataFrame::new(df))
    }

    #[pyo3(signature = (path, schema=None, primitivesAsString=None, prefersDecimal=None, allowComments=None, allowUnquotedFieldNames=None, allowSingleQuotes=None, allowNumericLeadingZero=None, allowBackslashEscapingAnyCharacter=None, mode=None, columnNameOfCorruptRecord=None, dateFormat=None, timestampFormat=None, multiLine=None, allowUnquotedControlChars=None, lineSep=None, locale=None, dropFieldIfAllNull=None, encoding=None, pathGlobFilter=None, recursiveFileLookup=None, allowNonNumericNumbers=None, useUnsafeRow=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn json(
        &mut self,
        path: &str,
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
        let mut r = self.take()?;
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
        Ok(PyDataFrame::new(r.json(path)))
    }

    #[pyo3(signature = (path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None, datetimeRebaseMode=None, int96RebaseMode=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn parquet(
        &mut self,
        path: &str,
        mergeSchema: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        datetimeRebaseMode: Option<&Bound<'_, PyAny>>,
        int96RebaseMode: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.take()?;
        r = set_ropt(r, "mergeSchema", mergeSchema)?;
        r = set_ropt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_ropt(r, "recursiveFileLookup", recursiveFileLookup)?;
        r = set_ropt(r, "datetimeRebaseMode", datetimeRebaseMode)?;
        r = set_ropt(r, "int96RebaseMode", int96RebaseMode)?;
        Ok(PyDataFrame::new(r.parquet(path)))
    }

    #[pyo3(signature = (path, schema=None, sep=None, encoding=None, quote=None, escape=None, comment=None, header=None, inferSchema=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, maxMalformedLogPerPartition=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, charToEscapeQuoteEscaping=None, enforceSchema=None, emptyValue=None, locale=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, unescapedQuoteHandling=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn csv(
        &mut self,
        path: &str,
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
        let mut r = self.take()?;
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
        Ok(PyDataFrame::new(r.csv(path)))
    }

    #[pyo3(signature = (path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None))]
    #[allow(non_snake_case)]
    fn orc(
        &mut self,
        path: &str,
        mergeSchema: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.take()?;
        r = set_ropt(r, "mergeSchema", mergeSchema)?;
        r = set_ropt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_ropt(r, "recursiveFileLookup", recursiveFileLookup)?;
        Ok(PyDataFrame::new(r.orc(path)))
    }

    #[pyo3(signature = (path, wholetext=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None))]
    #[allow(non_snake_case)]
    fn text(
        &mut self,
        path: &str,
        wholetext: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.take()?;
        r = set_ropt(r, "wholetext", wholetext)?;
        r = set_ropt(r, "lineSep", lineSep)?;
        r = set_ropt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_ropt(r, "recursiveFileLookup", recursiveFileLookup)?;
        Ok(PyDataFrame::new(r.text(path)))
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
    inner: Option<DataStreamWriter>,
}

impl PyDataStreamWriter {
    pub fn new(writer: DataStreamWriter) -> Self {
        PyDataStreamWriter {
            inner: Some(writer),
        }
    }

    fn take(&mut self) -> PyResult<DataStreamWriter> {
        self.inner.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("DataStreamWriter already consumed")
        })
    }

    /// Apply the one-shot writer options accepted by `start`/`toTable`
    /// (`format`, `outputMode`, `partitionBy`, `queryName`, `**options`) before
    /// terminating the stream. Mirrors the reference, where these are keyword args.
    fn apply_write_opts(
        &mut self,
        format: Option<&str>,
        output_mode: Option<&str>,
        partition_by: Option<&Bound<'_, PyAny>>,
        query_name: Option<&str>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<DataStreamWriter> {
        let mut w = self.take()?;
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
            w = w.query_name(qn);
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
        Ok(w)
    }
}

#[pymethods]
impl PyDataStreamWriter {
    fn outputMode(&mut self, outputMode: &str) -> PyResult<PyDataStreamWriter> {
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.output_mode(outputMode)),
        })
    }

    fn format(&mut self, source: &str) -> PyResult<PyDataStreamWriter> {
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.format(source)),
        })
    }

    fn option(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<PyDataStreamWriter> {
        // None -> option left unset; bools -> "true"/"false" (reference `to_str`).
        match crate::coerce_option_value(value)? {
            Some(v) => Ok(PyDataStreamWriter {
                inner: Some(self.take()?.option(key, &v)),
            }),
            None => Ok(PyDataStreamWriter {
                inner: Some(self.take()?),
            }),
        }
    }

    // Mirrors reference `DataStreamWriter.options(**options)`: keyword args; None values
    // skipped, booleans lowercased.
    #[pyo3(signature = (**options))]
    fn options(&mut self, options: Option<&Bound<'_, PyDict>>) -> PyResult<PyDataStreamWriter> {
        let mut opts = HashMap::new();
        if let Some(options) = options {
            for (k, v) in options.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    opts.insert(k.str()?.to_string(), val);
                }
            }
        }
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.options(opts)),
        })
    }

    #[pyo3(signature = (*cols))]
    fn partitionBy(&mut self, cols: Vec<String>) -> PyResult<PyDataStreamWriter> {
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.partition_by(col_refs)),
        })
    }

    #[pyo3(signature = (*cols))]
    fn clusterBy(&mut self, cols: Vec<String>) -> PyResult<PyDataStreamWriter> {
        let col_refs: Vec<&str> = cols.iter().map(|s| s.as_str()).collect();
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.cluster_by(col_refs)),
        })
    }

    fn queryName(&mut self, queryName: &str) -> PyResult<PyDataStreamWriter> {
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.query_name(queryName)),
        })
    }

    /// `DataStreamWriter.trigger(...)`: mirrors the reference keyword API — exactly one
    /// of `processingTime` / `once` / `availableNow` / `continuous` is given.
    #[pyo3(signature = (*, processingTime=None, once=None, continuous=None, availableNow=None, realTime=None))]
    fn trigger(
        &mut self,
        processingTime: Option<&str>,
        once: Option<bool>,
        continuous: Option<&str>,
        availableNow: Option<bool>,
        realTime: Option<&str>,
    ) -> PyResult<PyDataStreamWriter> {
        let trigger = if let Some(interval) = processingTime {
            Trigger::ProcessingTime(interval.to_string())
        } else if once == Some(true) {
            Trigger::Once
        } else if let Some(interval) = continuous {
            Trigger::Continuous(interval.to_string())
        } else if availableNow == Some(true) {
            Trigger::AvailableNow
        } else if realTime.is_some() {
            // `realTime` is accepted for signature parity with reference pyspark; the
            // connect core's Trigger enum does not yet model a real-time trigger.
            return Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>(
                "the realTime trigger is not yet supported by this client",
            ));
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "trigger() requires exactly one of processingTime, once, continuous, availableNow, realTime",
            ));
        };
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.trigger(trigger)),
        })
    }

    #[pyo3(signature = (path=None, format=None, outputMode=None, partitionBy=None, queryName=None, **options))]
    #[allow(non_snake_case)]
    fn start(
        &mut self,
        path: Option<&str>,
        format: Option<&str>,
        outputMode: Option<&str>,
        partitionBy: Option<&Bound<'_, PyAny>>,
        queryName: Option<&str>,
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
        queryName: Option<&str>,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyStreamingQuery> {
        let writer = self.apply_write_opts(format, outputMode, partitionBy, queryName, options)?;
        let query = writer.to_table(tableName).to_pyerr()?;
        Ok(PyStreamingQuery::new(query))
    }

    /// `DataStreamWriter.foreachBatch(func)`: cloudpickle the `(batch_df, batch_id)`
    /// function (via the bundled `pyspark.cloudpickle`) and attach it as the
    /// foreach-batch sink. Pickling happens here so the Python skin can re-export this
    /// class directly rather than subclassing a (non-subclassable) PyO3 type.
    fn foreachBatch(
        &mut self,
        py: Python<'_>,
        func: &Bound<'_, PyAny>,
    ) -> PyResult<PyDataStreamWriter> {
        let command = crate::dataframe::py_cloudpickle(py, func)?;
        let payload = PythonUDFPayload::new(
            spark_connect::types::DataType::Struct { fields: vec![] },
            0, // eval type is unused for the streaming foreach sinks
            command,
            crate::dataframe::py_version(py),
        );
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.foreach_batch(payload)),
        })
    }

    /// `DataStreamWriter.foreach(f)`: wrap the row handler as the reference client does
    /// — `(f, None, serializer, serializer)` with `AutoBatchedSerializer(CPickleSerializer())`
    /// — and cloudpickle it so the worker deserializes it against its own
    /// `pyspark.serializers`. Built here to avoid a Python subclass of the PyO3 class.
    fn foreach(&mut self, py: Python<'_>, f: &Bound<'_, PyAny>) -> PyResult<PyDataStreamWriter> {
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
                f.clone(),
                py.None().into_bound(py),
                serializer.clone(),
                serializer,
            ],
        )?;
        let command = crate::dataframe::py_cloudpickle(py, command_tuple.as_any())?;
        let payload = PythonUDFPayload::new(
            spark_connect::types::DataType::Struct { fields: vec![] },
            0,
            command,
            crate::dataframe::py_version(py),
        );
        Ok(PyDataStreamWriter {
            inner: Some(self.take()?.foreach(payload)),
        })
    }
}

/// Python wrapper for StreamingQueryStatus.
#[pyclass(
    name = "StreamingQueryStatus",
    module = "pyspark.sql.streaming.query",
    from_py_object
)]
#[derive(Clone)]
pub struct PyStreamingQueryStatus {
    inner: StreamingQueryStatus,
}

impl PyStreamingQueryStatus {
    pub fn new(status: StreamingQueryStatus) -> Self {
        PyStreamingQueryStatus { inner: status }
    }
}

#[pymethods]
impl PyStreamingQueryStatus {
    #[getter]
    fn is_active(&self) -> bool {
        self.inner.is_active
    }

    #[getter]
    fn status_message(&self) -> String {
        self.inner.status_message.clone()
    }

    #[getter]
    fn is_data_available(&self) -> bool {
        self.inner.is_data_available
    }

    #[getter]
    fn is_trigger_active(&self) -> bool {
        self.inner.is_trigger_active
    }
}

/// Python wrapper for StreamingQueryException.
#[pyclass(
    name = "StreamingQueryException",
    module = "pyspark.sql.streaming.query",
    from_py_object
)]
#[derive(Clone)]
pub struct PyStreamingQueryException {
    inner: StreamingQueryException,
}

impl PyStreamingQueryException {
    pub fn new(exc: StreamingQueryException) -> Self {
        PyStreamingQueryException { inner: exc }
    }
}

#[pymethods]
impl PyStreamingQueryException {
    #[getter]
    fn message(&self) -> String {
        self.inner.message.clone()
    }

    #[getter]
    fn error_class(&self) -> String {
        self.inner.error_class.clone()
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

    #[getter]
    fn status(&self) -> PyResult<PyStreamingQueryStatus> {
        let status = self.inner.status().to_pyerr()?;
        Ok(PyStreamingQueryStatus::new(status))
    }

    fn stop(&self) -> PyResult<()> {
        self.inner.stop().to_pyerr()
    }

    #[pyo3(signature = (timeout=None))]
    fn awaitTermination(&self, timeout: Option<f64>) -> PyResult<Option<bool>> {
        self.inner.await_termination(timeout).to_pyerr()
    }

    #[getter]
    fn lastProgress(&self) -> PyResult<Option<String>> {
        self.inner.last_progress().to_pyerr()
    }

    #[getter]
    fn recentProgress(&self) -> PyResult<Vec<String>> {
        self.inner.recent_progress().to_pyerr()
    }

    fn processAllAvailable(&self) -> PyResult<()> {
        self.inner.process_all_available().to_pyerr()
    }

    // Mirrors reference `StreamingQuery.explain(extended=False)` - the arg is optional.
    #[pyo3(signature = (extended=false))]
    fn explain(&self, extended: bool) -> PyResult<String> {
        self.inner.explain(extended).to_pyerr()
    }

    fn exception(&self) -> PyResult<Option<PyStreamingQueryException>> {
        let exc = self.inner.exception().to_pyerr()?;
        Ok(exc.map(PyStreamingQueryException::new))
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
    fn awaitAnyTermination(&self, timeout: Option<f64>) -> PyResult<Option<bool>> {
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
