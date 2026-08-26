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
    fn json(&mut self, path: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take()?.json(path)))
    }

    /// Read Parquet file(s).
    fn parquet(&mut self, path: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take()?.parquet(path)))
    }

    /// Read CSV file(s).
    fn csv(&mut self, path: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take()?.csv(path)))
    }

    /// Read ORC file(s).
    fn orc(&mut self, path: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take()?.orc(path)))
    }

    /// Read text file(s) (one `value` string column per line).
    fn text(&mut self, path: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take()?.text(path)))
    }

    /// Read XML file(s).
    fn xml(&mut self, path: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take()?.xml(path)))
    }

    /// Read from a JDBC source. `predicates` optionally partitions the read.
    #[pyo3(signature = (url, table, predicates=None))]
    fn jdbc(
        &mut self,
        url: &str,
        table: &str,
        predicates: Option<Vec<String>>,
    ) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.take()?.jdbc(url, table, predicates)))
    }
}
