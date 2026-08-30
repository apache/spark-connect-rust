//! PyO3 wrapper for spark_connect::readwriter::DataFrameReader (`spark.read`).

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::readwriter::DataFrameReader;

use crate::dataframe::PyDataFrame;

/// Python wrapper for the batch DataFrameReader. The core reader is a consuming
/// builder, so each step takes the inner value and returns a fresh wrapper (mirrors
/// the existing PyDataStreamReader).
#[pyclass(name = "DataFrameReader", module = "pyspark.sql.readwriter")]
pub struct PyDataFrameReader {
    inner: Option<DataFrameReader>,
}

impl PyDataFrameReader {
    pub fn new(reader: DataFrameReader) -> Self {
        PyDataFrameReader {
            inner: Some(reader),
        }
    }

    fn take(&mut self) -> PyResult<DataFrameReader> {
        self.inner.take().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("DataFrameReader already consumed")
        })
    }

    /// Take the inner reader and apply per-call `**options` (skipping None,
    /// lowercasing bools) before a format-specific read.
    fn take_with_opts(&mut self, options: Option<&Bound<'_, PyDict>>) -> PyResult<DataFrameReader> {
        let mut r = self.take()?;
        if let Some(opts) = options {
            for (k, v) in opts.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    r = r.option(&k.str()?.to_string(), &val);
                }
            }
        }
        Ok(r)
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

/// Resolve a reader `schema` arg (a StructType/DataType or a DDL string) to the
/// reader's DDL schema.
fn apply_reader_schema(r: DataFrameReader, v: &Bound<'_, PyAny>) -> PyResult<DataFrameReader> {
    let dt = crate::types::py_to_data_type(v)?;
    Ok(r.schema(dt.simple_string()))
}

#[pymethods]
impl PyDataFrameReader {
    /// Set the source format (e.g. "parquet", "json", "csv").
    fn format(&mut self, source: &str) -> PyResult<PyDataFrameReader> {
        Ok(PyDataFrameReader::new(self.take()?.format(source)))
    }

    /// Set the schema (a DDL string).
    fn schema(&mut self, schema: &str) -> PyResult<PyDataFrameReader> {
        Ok(PyDataFrameReader::new(
            self.take()?.schema(schema.to_string()),
        ))
    }

    /// Set a single read option. `None` leaves the option unset (not the string
    /// "None"); booleans lowercase to "true"/"false" (reference `to_str` semantics).
    fn option(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<PyDataFrameReader> {
        match crate::coerce_option_value(value)? {
            Some(v) => Ok(PyDataFrameReader::new(self.take()?.option(key, &v))),
            None => Ok(PyDataFrameReader::new(self.take()?)),
        }
    }

    /// Set multiple read options. Mirrors reference `DataFrameReader.options(**options)`
    /// - keyword args; `None` values are skipped and booleans lowercased.
    #[pyo3(signature = (**options))]
    fn options(&mut self, options: Option<&Bound<'_, PyDict>>) -> PyResult<PyDataFrameReader> {
        let mut map = HashMap::new();
        if let Some(options) = options {
            for (k, v) in options.iter() {
                if let Some(val) = crate::coerce_option_value(&v)? {
                    map.insert(k.str()?.to_string(), val);
                }
            }
        }
        Ok(PyDataFrameReader::new(self.take()?.options(map)))
    }

    /// Load data from the (optional) path using the configured format/options.
    #[pyo3(signature = (path=None))]
    fn load(&mut self, path: Option<&str>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take()?.load(path)))
    }

    /// Read a table by name.
    fn table(&mut self, table_name: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take()?.table(table_name)))
    }

    /// Read the CDC changes of a named table. Mirrors `DataFrameReader.changes`.
    #[allow(non_snake_case)]
    fn changes(&mut self, tableName: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take()?.changes(tableName)))
    }

    /// Read JSON file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
    #[pyo3(signature = (path, schema=None, primitivesAsString=None, prefersDecimal=None, allowComments=None, allowUnquotedFieldNames=None, allowSingleQuotes=None, allowNumericLeadingZero=None, allowBackslashEscapingAnyCharacter=None, mode=None, columnNameOfCorruptRecord=None, dateFormat=None, timestampFormat=None, multiLine=None, allowUnquotedControlChars=None, lineSep=None, samplingRatio=None, dropFieldIfAllNull=None, encoding=None, locale=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, allowNonNumericNumbers=None, useUnsafeRow=None))]
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
        let mut r = self.take()?;
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
        Ok(PyDataFrame::new(r.json(path)))
    }

    /// Read Parquet file(s).
    #[pyo3(signature = (path, **options))]
    fn parquet(
        &mut self,
        path: &str,
        options: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(
            self.take_with_opts(options)?.parquet(path),
        ))
    }

    /// Read CSV file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
    #[pyo3(signature = (path, schema=None, sep=None, encoding=None, quote=None, escape=None, comment=None, header=None, inferSchema=None, ignoreLeadingWhiteSpace=None, ignoreTrailingWhiteSpace=None, nullValue=None, nanValue=None, positiveInf=None, negativeInf=None, dateFormat=None, timestampFormat=None, maxColumns=None, maxCharsPerColumn=None, maxMalformedLogPerPartition=None, mode=None, columnNameOfCorruptRecord=None, multiLine=None, charToEscapeQuoteEscaping=None, samplingRatio=None, enforceSchema=None, emptyValue=None, locale=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None, unescapedQuoteHandling=None))]
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
        let mut r = self.take()?;
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
        Ok(PyDataFrame::new(r.csv(path)))
    }

    /// Read ORC file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
    #[pyo3(signature = (path, mergeSchema=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn orc(
        &mut self,
        path: &str,
        mergeSchema: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        modifiedBefore: Option<&Bound<'_, PyAny>>,
        modifiedAfter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.take()?;
        r = set_opt(r, "mergeSchema", mergeSchema)?;
        r = set_opt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_opt(r, "recursiveFileLookup", recursiveFileLookup)?;
        r = set_opt(r, "modifiedBefore", modifiedBefore)?;
        r = set_opt(r, "modifiedAfter", modifiedAfter)?;
        Ok(PyDataFrame::new(r.orc(path)))
    }

    /// Read TEXT file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
    #[pyo3(signature = (path, wholetext=None, lineSep=None, pathGlobFilter=None, recursiveFileLookup=None, modifiedBefore=None, modifiedAfter=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn text(
        &mut self,
        path: &str,
        wholetext: Option<&Bound<'_, PyAny>>,
        lineSep: Option<&Bound<'_, PyAny>>,
        pathGlobFilter: Option<&Bound<'_, PyAny>>,
        recursiveFileLookup: Option<&Bound<'_, PyAny>>,
        modifiedBefore: Option<&Bound<'_, PyAny>>,
        modifiedAfter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.take()?;
        r = set_opt(r, "wholetext", wholetext)?;
        r = set_opt(r, "lineSep", lineSep)?;
        r = set_opt(r, "pathGlobFilter", pathGlobFilter)?;
        r = set_opt(r, "recursiveFileLookup", recursiveFileLookup)?;
        r = set_opt(r, "modifiedBefore", modifiedBefore)?;
        r = set_opt(r, "modifiedAfter", modifiedAfter)?;
        Ok(PyDataFrame::new(r.text(path)))
    }

    /// Read XML file(s) - full pyspark signature; each named option is
    /// applied as a read option when provided.
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
        let mut r = self.take()?;
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
        Ok(PyDataFrame::new(r.xml(path)))
    }

    /// Read from a JDBC source. Mirrors `DataFrameReader.jdbc(url, table,
    /// column=None, lowerBound=None, upperBound=None, numPartitions=None,
    /// predicates=None, properties=None)`: the column/bound/partition args and the
    /// connection `properties` are threaded as reader options (connect represents
    /// them that way); `predicates` stays the partitioning predicate list.
    #[pyo3(signature = (url, table, column=None, lowerBound=None, upperBound=None, numPartitions=None, predicates=None, properties=None))]
    #[allow(non_snake_case, clippy::too_many_arguments)]
    fn jdbc(
        &mut self,
        url: &str,
        table: &str,
        column: Option<String>,
        lowerBound: Option<&Bound<'_, PyAny>>,
        upperBound: Option<&Bound<'_, PyAny>>,
        numPartitions: Option<i32>,
        predicates: Option<Vec<String>>,
        properties: Option<HashMap<String, String>>,
    ) -> PyResult<PyDataFrame> {
        let mut r = self.take()?;
        if let Some(c) = column {
            r = r.option("partitionColumn", &c);
        }
        if let Some(lb) = lowerBound {
            r = r.option("lowerBound", &lb.str()?.to_string());
        }
        if let Some(ub) = upperBound {
            r = r.option("upperBound", &ub.str()?.to_string());
        }
        if let Some(n) = numPartitions {
            r = r.option("numPartitions", &n.to_string());
        }
        if let Some(props) = properties {
            for (k, v) in props {
                r = r.option(&k, &v);
            }
        }
        Ok(PyDataFrame::new(r.jdbc(url, table, predicates)))
    }
}
