// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of the clustering models, over the server-side
//! objects of [`super::object`].
//!
//! Each estimator fits its typed model (`KMeans::fit` a [`KMeansModel`]), whose members are
//! fetches of its server-side object, as PySpark's `_call_java` of each member, and whose
//! summaries implement [`ClusteringSummary`]. Each family has its own file.

use crate::dataframe::DataFrame;
use crate::ml::object::MlValue;
use crate::ml::regression_api::{as_i64, unexpected};
use crate::ml::{CachedModel, MlMatrix, MlVector, Params};
use crate::row::Value;
use spark_connect_core::error::Result;

mod bisecting_kmeans;
mod gaussian_mixture;
mod kmeans;
mod lda;
pub use lda::*;
mod power_iteration;
mod summary;
pub use summary::*;

/// A boolean value of `method`.
fn as_bool(method: &str, value: MlValue) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| unexpected(method, "a boolean", &value))
}

/// An array of longs of `method`, as PySpark reads a list of ints.
fn as_i64s(method: &str, value: MlValue) -> Result<Vec<i64>> {
    match value {
        MlValue::Array(items) => items.into_iter().map(|item| as_i64(method, item)).collect(),
        other => Err(unexpected(method, "an array of longs", &other)),
    }
}

/// An array of strings of `method`, as PySpark reads a list of strings.
fn as_strings(method: &str, value: MlValue) -> Result<Vec<String>> {
    match value {
        MlValue::Array(items) => items
            .into_iter()
            .map(|item| match item {
                MlValue::String(s) => Ok(s),
                other => Err(unexpected(method, "an array of strings", &other)),
            })
            .collect(),
        other => Err(unexpected(method, "an array of strings", &other)),
    }
}

/// The summary dataset of a model of the training dataset `train`, of a class that keeps PySpark's
/// `HasTrainingSummary._summary_dataset`: the model's transform of `train`. PySpark transforms
/// `train` before it copies the estimator's params to the model, so the transform has no params.
fn transformed_summary_dataset(model: &CachedModel, train: &DataFrame) -> Result<DataFrame> {
    model.transform(train, &Params::new())
}

/// The rows of `matrix`, as PySpark's `Matrix.toArray` lays out its values: a dense matrix of its
/// values by column (by row of a transposed one), and a sparse matrix of its compressed columns
/// (its compressed rows of a transposed one).
pub(crate) fn matrix_rows(matrix: &MlMatrix) -> Vec<Vec<f64>> {
    match matrix {
        MlMatrix::Dense {
            num_rows,
            num_cols,
            values,
            is_transposed,
        } => {
            let (rows, cols) = (*num_rows as usize, *num_cols as usize);
            (0..rows)
                .map(|i| {
                    (0..cols)
                        .map(|j| {
                            let index = if *is_transposed {
                                i * cols + j
                            } else {
                                i + j * rows
                            };
                            values[index]
                        })
                        .collect()
                })
                .collect()
        }
        MlMatrix::Sparse {
            num_rows,
            num_cols,
            col_ptrs,
            row_indices,
            values,
            is_transposed,
        } => {
            let mut rows = vec![vec![0.0; *num_cols as usize]; *num_rows as usize];
            for (k, bounds) in col_ptrs.windows(2).enumerate() {
                for entry in bounds[0] as usize..bounds[1] as usize {
                    let index = row_indices[entry] as usize;
                    let (i, j) = if *is_transposed {
                        (k, index)
                    } else {
                        (index, k)
                    };
                    rows[i][j] = values[entry];
                }
            }
            rows
        }
    }
}

fn doubles_of(value: &Value) -> Option<Vec<f64>> {
    match value {
        Value::List(items) => items
            .iter()
            .map(|item| match item {
                Value::Double(v) => Some(*v),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn ints_of(value: &Value) -> Option<Vec<i32>> {
    match value {
        Value::List(items) => items
            .iter()
            .map(|item| match item {
                Value::Integer(v) => Some(*v),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn int_of(value: &Value) -> Option<i32> {
    match value {
        Value::Integer(v) => Some(*v),
        _ => None,
    }
}

/// The vector of a collected value of a `VectorUDT` column: the struct of its type (0 sparse,
/// 1 dense), its size and its indices (of a sparse vector), and its values.
pub(crate) fn vector_of_value(value: &Value) -> Option<MlVector> {
    let Value::Struct(fields) = value else {
        return None;
    };
    let fields = fields.values();
    match fields.first()? {
        Value::Byte(1) => Some(MlVector::Dense(doubles_of(fields.get(3)?)?)),
        Value::Byte(0) => Some(MlVector::Sparse {
            size: int_of(fields.get(1)?)?,
            indices: ints_of(fields.get(2)?)?,
            values: doubles_of(fields.get(3)?)?,
        }),
        _ => None,
    }
}

/// The matrix of a collected value of a `MatrixUDT` column: the struct of its type (0 sparse,
/// 1 dense), its shape, its column pointers and row indices (of a sparse matrix), its values and
/// whether it is transposed.
pub(crate) fn matrix_of_value(value: &Value) -> Option<MlMatrix> {
    let Value::Struct(fields) = value else {
        return None;
    };
    let fields = fields.values();
    let num_rows = int_of(fields.get(1)?)?;
    let num_cols = int_of(fields.get(2)?)?;
    let values = doubles_of(fields.get(5)?)?;
    let is_transposed = match fields.get(6)? {
        Value::Bool(v) => *v,
        _ => return None,
    };
    match fields.first()? {
        Value::Byte(1) => Some(MlMatrix::Dense {
            num_rows,
            num_cols,
            values,
            is_transposed,
        }),
        Value::Byte(0) => Some(MlMatrix::Sparse {
            num_rows,
            num_cols,
            col_ptrs: ints_of(fields.get(3)?)?,
            row_indices: ints_of(fields.get(4)?)?,
            values,
            is_transposed,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_rows_lay_out_the_values_as_pyspark_to_array() {
        // [[1, 3], [2, 4]] of each layout.
        let dense = MlMatrix::Dense {
            num_rows: 2,
            num_cols: 2,
            values: vec![1.0, 2.0, 3.0, 4.0],
            is_transposed: false,
        };
        let transposed = MlMatrix::Dense {
            num_rows: 2,
            num_cols: 2,
            values: vec![1.0, 3.0, 2.0, 4.0],
            is_transposed: true,
        };
        let expected = vec![vec![1.0, 3.0], vec![2.0, 4.0]];
        assert_eq!(matrix_rows(&dense), expected);
        assert_eq!(matrix_rows(&transposed), expected);
        // [[0, 5, 0], [6, 0, 7]] of its columns, and of its rows when transposed.
        let sparse = MlMatrix::Sparse {
            num_rows: 2,
            num_cols: 3,
            col_ptrs: vec![0, 1, 2, 3],
            row_indices: vec![1, 0, 1],
            values: vec![6.0, 5.0, 7.0],
            is_transposed: false,
        };
        let sparse_rows = MlMatrix::Sparse {
            num_rows: 2,
            num_cols: 3,
            col_ptrs: vec![0, 1, 3],
            row_indices: vec![1, 0, 2],
            values: vec![5.0, 6.0, 7.0],
            is_transposed: true,
        };
        let expected = vec![vec![0.0, 5.0, 0.0], vec![6.0, 0.0, 7.0]];
        assert_eq!(matrix_rows(&sparse), expected);
        assert_eq!(matrix_rows(&sparse_rows), expected);
    }
}
