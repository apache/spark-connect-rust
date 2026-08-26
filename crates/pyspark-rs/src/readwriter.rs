//! PyO3 wrapper for spark_connect::readwriter::DataFrameReader (`spark.read`).

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use spark_connect::readwriter::DataFrameReader;

use crate::dataframe::PyDataFrame;

/// Python wrapper for the batch DataFrameReader. The core reader is a consuming
/// builder, so each step takes the inner value and returns a fresh wrapper (mirrors
/// the existing PyDataStreamReader).
#[pyclass(name = "DataFrameReader")]
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

    /// Read JSON file(s).
    #[pyo3(signature = (path, **options))]
    fn json(&mut self, path: &str, options: Option<&Bound<'_, PyDict>>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take_with_opts(options)?.json(path)))
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

    /// Read CSV file(s).
    #[pyo3(signature = (path, **options))]
    fn csv(&mut self, path: &str, options: Option<&Bound<'_, PyDict>>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take_with_opts(options)?.csv(path)))
    }

    /// Read ORC file(s).
    #[pyo3(signature = (path, **options))]
    fn orc(&mut self, path: &str, options: Option<&Bound<'_, PyDict>>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take_with_opts(options)?.orc(path)))
    }

    /// Read text file(s) (one `value` string column per line).
    #[pyo3(signature = (path, **options))]
    fn text(&mut self, path: &str, options: Option<&Bound<'_, PyDict>>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take_with_opts(options)?.text(path)))
    }

    /// Read XML file(s).
    #[pyo3(signature = (path, **options))]
    fn xml(&mut self, path: &str, options: Option<&Bound<'_, PyDict>>) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take_with_opts(options)?.xml(path)))
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
