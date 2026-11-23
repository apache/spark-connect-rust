//! The plans of the reads of the readers of Spark Connect, as PySpark Connect's `DataSource`,
//! `Read`, `RelationChanges` and `Parse` of `pyspark.sql.connect.plan`, which its
//! `DataFrameReader` and `DataStreamReader`, PySpark's Python classes, make of their state.
//!
//! A plan keeps its arguments as PySpark's plan checks them when it is made, with its assertions
//! and in their order, and as the core reader takes them, with the error of the plan of the first
//! argument that it cannot send, as PySpark's plan raises it when it is sent: a str that UTF-8 cannot
//! encode, or an object that a string field of protobuf refuses. `_read_dataframe` makes the
//! DataFrame of the core of a plan, of the session of a reader's `_client`, as PySpark's `_df` makes
//! the DataFrame of its plan, and that DataFrame fails each request with that error.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use spark_connect::plan::ParseFormat;
use spark_connect_core::error::SparkError;

use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;
use crate::readwriter::{table_name_of, with_plan_error};

/// The relation of a read, of the arguments of its plan.
enum ReadRelation {
    /// PySpark's `DataSource`, of a reader's `load` and of a `jdbc` of predicates.
    Source {
        format: Option<String>,
        schema: Option<String>,
        options: Vec<(String, String)>,
        paths: Option<Vec<String>>,
        predicates: Option<Vec<String>>,
        streaming: bool,
        source_name: Option<String>,
    },
    /// PySpark's `Read` of a named table.
    Table {
        name: String,
        options: Vec<(String, String)>,
        streaming: bool,
    },
    /// PySpark's `RelationChanges` of a named table.
    Changes {
        name: String,
        options: Vec<(String, String)>,
        streaming: bool,
    },
    /// PySpark's `Parse` of a DataFrame of strings, the plan of `child`.
    Parse {
        format: ParseFormat,
        schema: Option<String>,
        options: Vec<(String, String)>,
    },
}

/// A plan of a read, which `_read_dataframe` makes the DataFrame of.
#[pyclass(name = "_ReadPlan", module = "pyspark.sql.connect.readwriter", frozen)]
pub struct PyReadPlan {
    relation: ReadRelation,
    /// The error of the plan, of an argument that it cannot send, which each request of its
    /// DataFrame raises, as PySpark's plan raises it when it is sent.
    error: Option<SparkError>,
    /// The plan of the DataFrame of strings of a `Parse`, PySpark's `path._plan`.
    child: Option<Py<PyAny>>,
}

#[pymethods]
impl PyReadPlan {
    /// The plan of the DataFrame of a `Parse`, which Python's garbage collector sees.
    fn __traverse__(&self, visit: pyo3::PyVisit<'_>) -> Result<(), pyo3::PyTraverseError> {
        if let Some(child) = &self.child {
            visit.call(child)?;
        }
        Ok(())
    }
}

/// The bare `AssertionError` of an `assert` of PySpark's plan.
fn assertion() -> PyErr {
    pyo3::exceptions::PyAssertionError::new_err(())
}

/// The error that PySpark's plan raises when it sends `text`, a str that UTF-8 cannot encode.
fn encode_error(text: &Bound<'_, PyAny>) -> PyResult<SparkError> {
    Ok(SparkError::unencodable_str(&crate::options::utf32le_of(
        text,
    )?))
}

/// The text of `text`, a str, as a string field of protobuf takes it, or the error of the plan for
/// a str that UTF-8 cannot encode, which is kept as the text of the str where its lone surrogates
/// are replaced: the plan fails with the error, so that text is never sent, and it is not empty,
/// as the core reader requires of a path.
fn sent_text(text: &Bound<'_, PyAny>, error: &mut Option<SparkError>) -> PyResult<String> {
    match text.extract::<String>() {
        Ok(text) => Ok(text),
        Err(_) => {
            if error.is_none() {
                *error = Some(encode_error(text)?);
            }
            Ok(text.cast::<PyString>()?.to_string_lossy().into_owned())
        }
    }
}

/// The strs of `items`, a list that PySpark's `DataSource` asserts to be of strs, as it sends them.
fn source_list(
    items: Option<&Bound<'_, PyAny>>,
    error: &mut Option<SparkError>,
) -> PyResult<Option<Vec<String>>> {
    let Some(items) = items.filter(|items| !items.is_none()) else {
        return Ok(None);
    };
    // `assert isinstance(paths, list)` and `assert all(isinstance(path, str) for path in paths)`.
    let items = items.cast::<PyList>().map_err(|_| assertion())?;
    let items: Vec<Bound<'_, PyAny>> = items.iter().collect();
    if items.iter().any(|item| !item.is_instance_of::<PyString>()) {
        return Err(assertion());
    }
    items
        .iter()
        .map(|item| sent_text(item, error))
        .collect::<PyResult<_>>()
        .map(Some)
}

/// The options of `options` of a `DataSource`, as PySpark's checks them when it is made: of each
/// entry of a value that is not None, whose key and value it asserts to be strs, in its order.
fn source_options(
    options: Option<&Bound<'_, PyAny>>,
    error: &mut Option<SparkError>,
) -> PyResult<Vec<(String, String)>> {
    let Some(options) = options.filter(|options| !options.is_none()) else {
        return Ok(vec![]);
    };
    let mut pairs = Vec::new();
    for item in options.call_method0("items")?.try_iter()? {
        let (key, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
        if value.is_none() {
            continue;
        }
        if !key.is_instance_of::<PyString>() || !value.is_instance_of::<PyString>() {
            return Err(assertion());
        }
        pairs.push((key, value));
    }
    // Of the plan, which sends the options once they are all checked.
    pairs
        .iter()
        .map(|(key, value)| Ok((sent_text(key, error)?, sent_text(value, error)?)))
        .collect()
}

/// The options of `options` of a `Read`, a `RelationChanges` or a `Parse`, which PySpark's plan sets
/// each in the map of its message when it is sent: the error of the plan is that of the first entry
/// that the map refuses, of its key and then its value, such as a value of None.
fn sent_options(
    options: Option<&Bound<'_, PyAny>>,
    error: &mut Option<SparkError>,
) -> PyResult<Vec<(String, String)>> {
    let Some(options) = options.filter(|options| !options.is_none()) else {
        return Ok(vec![]);
    };
    let mut pairs = Vec::new();
    for item in options.call_method0("items")?.try_iter()? {
        let (key, value): (Bound<'_, PyAny>, Bound<'_, PyAny>) = item?.extract()?;
        match crate::options::plan_entry(&key, &value)? {
            Ok(pair) => pairs.push(pair),
            Err(entry_error) => {
                error.get_or_insert(entry_error);
            }
        }
    }
    Ok(pairs)
}

/// The argument `value`, unless it is None.
fn given<'py>(value: Option<&Bound<'py, PyAny>>) -> Option<Bound<'py, PyAny>> {
    value.filter(|value| !value.is_none()).cloned()
}

/// PySpark's `DataSource(format=None, schema=None, options=None, paths=None, predicates=None,
/// is_streaming=None, source_name=None)` of a read, which asserts the types of its arguments.
#[pyfunction]
#[pyo3(name = "_DataSource")]
#[pyo3(signature = (format=None, schema=None, options=None, paths=None, predicates=None, is_streaming=None, source_name=None))]
pub fn data_source(
    format: Option<&Bound<'_, PyAny>>,
    schema: Option<&Bound<'_, PyAny>>,
    options: Option<&Bound<'_, PyAny>>,
    paths: Option<&Bound<'_, PyAny>>,
    predicates: Option<&Bound<'_, PyAny>>,
    is_streaming: Option<&Bound<'_, PyAny>>,
    source_name: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyReadPlan> {
    let (format, schema, source_name) = (given(format), given(schema), given(source_name));
    // `assert format is None or isinstance(format, str)`, and so of the schema.
    for value in [&format, &schema].into_iter().flatten() {
        if !value.is_instance_of::<PyString>() {
            return Err(assertion());
        }
    }
    let mut error = None;
    let mut checked = None;
    let options = source_options(options, &mut checked)?;
    let paths = source_list(paths, &mut checked)?;
    let predicates = source_list(predicates, &mut checked)?;
    // `assert source_name is None or isinstance(source_name, str)`.
    if source_name
        .as_ref()
        .is_some_and(|name| !name.is_instance_of::<PyString>())
    {
        return Err(assertion());
    }
    let streaming = match is_streaming {
        Some(value) if !value.is_none() => value.is_truthy()?,
        _ => false,
    };
    // Of the plan, which sends its format, its schema, its options, its paths, its predicates
    // and its source name in that order.
    let format = format
        .map(|format| sent_text(&format, &mut error))
        .transpose()?;
    let schema = schema
        .map(|schema| sent_text(&schema, &mut error))
        .transpose()?;
    let mut error = error.or(checked);
    let source_name = source_name
        .map(|name| sent_text(&name, &mut error))
        .transpose()?;
    Ok(PyReadPlan {
        relation: ReadRelation::Source {
            format,
            schema,
            options,
            paths,
            predicates,
            streaming,
            source_name,
        },
        error,
        child: None,
    })
}

/// PySpark's `Read(table_name, options=None, is_streaming=None)` of a named table.
#[pyfunction]
#[pyo3(name = "_Read")]
#[pyo3(signature = (table_name, options=None, is_streaming=None))]
pub fn read_table(
    table_name: &Bound<'_, PyAny>,
    options: Option<&Bound<'_, PyAny>>,
    is_streaming: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyReadPlan> {
    let (name, mut error) = table_name_of(table_name)?;
    let options = sent_options(options, &mut error)?;
    let streaming = is_streaming.is_some_and(|value| value.is_truthy().unwrap_or(false));
    Ok(PyReadPlan {
        relation: ReadRelation::Table {
            name,
            options,
            streaming,
        },
        error,
        child: None,
    })
}

/// PySpark's `RelationChanges(table_name, options=None, is_streaming=None)` of a named table.
#[pyfunction]
#[pyo3(name = "_RelationChanges")]
#[pyo3(signature = (table_name, options=None, is_streaming=None))]
pub fn relation_changes(
    table_name: &Bound<'_, PyAny>,
    options: Option<&Bound<'_, PyAny>>,
    is_streaming: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyReadPlan> {
    let (name, mut error) = table_name_of(table_name)?;
    let options = sent_options(options, &mut error)?;
    let streaming = is_streaming.is_some_and(|value| value.is_truthy().unwrap_or(false));
    Ok(PyReadPlan {
        relation: ReadRelation::Changes {
            name,
            options,
            streaming,
        },
        error,
        child: None,
    })
}

/// PySpark's `Parse(child, format, schema=None, options=None)` of the DataFrame of strings of the
/// plan `child`, of a `proto.Parse.ParseFormat` of CSV, JSON or XML.
#[pyfunction]
#[pyo3(name = "_Parse")]
#[pyo3(signature = (child, format, schema=None, options=None))]
pub fn parse(
    child: &Bound<'_, PyAny>,
    format: i32,
    schema: Option<&Bound<'_, PyAny>>,
    options: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyReadPlan> {
    let format = match format {
        1 => ParseFormat::Csv,
        2 => ParseFormat::Json,
        3 => ParseFormat::Xml,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown enum label of spark.connect.Parse.ParseFormat: {format}"
            )))
        }
    };
    let mut error = None;
    let schema = match schema.filter(|schema| !schema.is_none()) {
        Some(schema) if schema.is_instance_of::<PyString>() => Some(sent_text(schema, &mut error)?),
        Some(_) => {
            error = Some(crate::options::builtin_type_error());
            None
        }
        None => None,
    };
    let options = sent_options(options, &mut error)?;
    Ok(PyReadPlan {
        relation: ReadRelation::Parse {
            format,
            schema,
            options,
        },
        error,
        child: Some(child.clone().unbind()),
    })
}

/// The DataFrame of the plan `plan`, of the session of `client`, as PySpark's `_df` makes it of its
/// `_client`: a session of this client, or the session of the drop-in's client of the `client` of
/// another object (see `state::session_of_value`).
#[pyfunction]
#[pyo3(name = "_read_dataframe")]
pub fn read_dataframe(
    plan: &Bound<'_, PyReadPlan>,
    client: &Bound<'_, PyAny>,
) -> PyResult<PyDataFrame> {
    let py = plan.py();
    let session = crate::state::session_of_value(client, pyo3::intern!(py, "client"))?;
    let plan = plan.get();
    let df = match &plan.relation {
        ReadRelation::Source {
            format,
            schema,
            options,
            paths,
            predicates,
            streaming: false,
            ..
        } => {
            let mut reader = session.read();
            if let Some(format) = format {
                reader = reader.format(format);
            }
            reader = reader.schema(schema.clone().unwrap_or_default());
            for (key, value) in options {
                reader = reader.option(key, value);
            }
            match predicates {
                // Of the url and the table of the options, which `jdbc` sets them of.
                Some(predicates) => {
                    let option = |name: &str| {
                        options
                            .iter()
                            .rev()
                            .find(|(key, _)| key == name)
                            .map(|(_, value)| value.clone())
                            .unwrap_or_default()
                    };
                    reader.jdbc(&option("url"), &option("dbtable"), Some(predicates.clone()))
                }
                None => reader.load_paths(paths.clone().unwrap_or_default()),
            }
        }
        ReadRelation::Source {
            format,
            schema,
            options,
            paths,
            source_name,
            ..
        } => {
            let mut reader = session.read_stream();
            if let Some(format) = format {
                reader = reader.format(format);
            }
            reader = reader.schema(schema.clone().unwrap_or_default());
            for (key, value) in options {
                reader = reader.option(key, value);
            }
            if let Some(name) = source_name {
                reader = reader.name(name);
            }
            let path = paths.as_ref().and_then(|paths| paths.first());
            reader.load(path.map(String::as_str))
        }
        ReadRelation::Table {
            name,
            options,
            streaming,
        } => {
            if *streaming {
                let mut reader = session.read_stream();
                for (key, value) in options {
                    reader = reader.option(key, value);
                }
                reader.table(name)
            } else {
                let mut reader = session.read();
                for (key, value) in options {
                    reader = reader.option(key, value);
                }
                reader.table(name)
            }
        }
        ReadRelation::Changes {
            name,
            options,
            streaming,
        } => {
            if *streaming {
                let mut reader = session.read_stream();
                for (key, value) in options {
                    reader = reader.option(key, value);
                }
                reader.changes(name)
            } else {
                let mut reader = session.read();
                for (key, value) in options {
                    reader = reader.option(key, value);
                }
                reader.changes(name)
            }
        }
        ReadRelation::Parse {
            format,
            schema,
            options,
        } => {
            let child = plan.child.as_ref().expect("the plan of a Parse").bind(py);
            let input = child.getattr(pyo3::intern!(py, "_df"))?;
            let input = input.cast::<PyDataFrame>()?;
            let mut reader = session.read().schema(schema.clone().unwrap_or_default());
            for (key, value) in options {
                reader = reader.option(key, value);
            }
            reader
                .parse(*format, &input.borrow().dataframe)
                .to_pyerr()?
        }
    };
    Ok(PyDataFrame::new(with_plan_error(df, plan.error.clone())))
}
