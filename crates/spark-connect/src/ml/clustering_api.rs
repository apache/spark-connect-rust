// SPDX-License-Identifier: Apache-2.0
//! The typed accessors, summaries and persistence of the clustering models, over the server-side
//! objects of [`super::object`].
//!
//! Each estimator fits its typed model (`KMeans::fit` a [`KMeansModel`]), whose members are
//! fetches of its server-side object, as PySpark's `_call_java` of each member, and whose
//! summaries implement [`ClusteringSummary`]. Each family has its own file.

use crate::ml::object::MlValue;
use crate::ml::regression_api::{as_i64, unexpected};
use crate::ml::{MlMatrix, MlVector};
use crate::row::Value;
use spark_connect_core::error::{Result, SparkError};

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

/// The rows of `matrix`, as PySpark's `Matrix.toArray` lays out its values: a dense matrix of its
/// values by column (by row of a transposed one), and a sparse matrix of its compressed columns
/// (its compressed rows of a transposed one).
///
/// A matrix that the server sends malformed, with negative dimensions, a number of values that
/// does not fit its shape, column pointers that are not a monotone range of its values, or row
/// indices out of range, is a ValueError, as PySpark's `DenseMatrix` and `SparseMatrix` require.
pub(crate) fn matrix_rows(matrix: &MlMatrix) -> Result<Vec<Vec<f64>>> {
    match matrix {
        MlMatrix::Dense {
            num_rows,
            num_cols,
            values,
            is_transposed,
        } => {
            let (rows, cols) = matrix_dims(matrix, *num_rows, *num_cols)?;
            if rows.checked_mul(cols) != Some(values.len()) {
                return Err(malformed(matrix, "its values do not fill its shape"));
            }
            Ok((0..rows)
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
                .collect())
        }
        MlMatrix::Sparse {
            num_rows,
            num_cols,
            col_ptrs,
            row_indices,
            values,
            is_transposed,
        } => {
            let (rows, cols) = matrix_dims(matrix, *num_rows, *num_cols)?;
            // The compressed dimension is the rows of a transposed matrix, else its columns.
            let (compressed, extent) = if *is_transposed {
                (rows, cols)
            } else {
                (cols, rows)
            };
            let pointers = col_ptrs
                .iter()
                .map(|&p| usize::try_from(p).ok())
                .collect::<Option<Vec<usize>>>();
            let indices = row_indices
                .iter()
                .map(|&i| usize::try_from(i).ok().filter(|&i| i < extent))
                .collect::<Option<Vec<usize>>>();
            let (Some(pointers), Some(indices)) = (pointers, indices) else {
                return Err(malformed(matrix, "a row index is out of range"));
            };
            if col_ptrs.len() != compressed + 1
                || pointers.first() != Some(&0)
                || pointers.last() != Some(&values.len())
                || pointers.windows(2).any(|bounds| bounds[0] > bounds[1])
                || indices.len() != values.len()
            {
                return Err(malformed(
                    matrix,
                    "its column pointers or row indices do not fit",
                ));
            }
            let mut rows_out = vec![vec![0.0; cols]; rows];
            for (k, bounds) in pointers.windows(2).enumerate() {
                for entry in bounds[0]..bounds[1] {
                    let index = indices[entry];
                    let (i, j) = if *is_transposed {
                        (k, index)
                    } else {
                        (index, k)
                    };
                    rows_out[i][j] = values[entry];
                }
            }
            Ok(rows_out)
        }
    }
}

/// The dimensions of `matrix` as counts, which a negative dimension is not.
fn matrix_dims(matrix: &MlMatrix, num_rows: i32, num_cols: i32) -> Result<(usize, usize)> {
    match (usize::try_from(num_rows), usize::try_from(num_cols)) {
        (Ok(rows), Ok(cols)) => Ok((rows, cols)),
        _ => Err(malformed(matrix, "its dimensions are negative")),
    }
}

/// The ValueError of a malformed matrix, of its shape and what is wrong with it.
fn malformed(matrix: &MlMatrix, problem: &str) -> SparkError {
    let (num_rows, num_cols) = match matrix {
        MlMatrix::Dense {
            num_rows, num_cols, ..
        }
        | MlMatrix::Sparse {
            num_rows, num_cols, ..
        } => (num_rows, num_cols),
    };
    SparkError::value_msg(format!(
        "malformed matrix of {num_rows} rows and {num_cols} columns: {problem}"
    ))
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
        assert_eq!(matrix_rows(&dense).unwrap(), expected);
        assert_eq!(matrix_rows(&transposed).unwrap(), expected);
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
        assert_eq!(matrix_rows(&sparse).unwrap(), expected);
        assert_eq!(matrix_rows(&sparse_rows).unwrap(), expected);
    }

    #[test]
    fn matrix_rows_fails_a_malformed_matrix_instead_of_panicking() {
        fn dense(num_rows: i32, num_cols: i32, values: &[f64]) -> MlMatrix {
            MlMatrix::Dense {
                num_rows,
                num_cols,
                values: values.to_vec(),
                is_transposed: false,
            }
        }
        fn sparse(
            (num_rows, num_cols): (i32, i32),
            (col_ptrs, row_indices): (&[i32], &[i32]),
            values: &[f64],
            is_transposed: bool,
        ) -> MlMatrix {
            MlMatrix::Sparse {
                num_rows,
                num_cols,
                col_ptrs: col_ptrs.to_vec(),
                row_indices: row_indices.to_vec(),
                values: values.to_vec(),
                is_transposed,
            }
        }
        let values = [6.0, 5.0, 7.0];
        let malformed = [
            dense(2, 2, &[1.0, 2.0, 3.0]),
            dense(-1, 2, &[]),
            dense(2, -2, &[]),
            // The column pointers of a 2 x 3 matrix have 4 entries, not 3, or do not start at 0.
            sparse((2, 3), (&[0, 1, 3], &[1, 0, 1]), &values, false),
            sparse((2, 3), (&[1, 2, 3, 3], &[1, 0, 1]), &values, false),
            // Not monotone, or not ending at the number of values.
            sparse((2, 3), (&[0, 2, 1, 3], &[1, 0, 1]), &values, false),
            sparse((2, 3), (&[0, 1, 2, 2], &[1, 0, 1]), &values, false),
            // A row index out of range of the rows, negative, or a column index of a transposed
            // matrix out of range of its columns.
            sparse((2, 3), (&[0, 1, 2, 3], &[2, 0, 1]), &values, false),
            sparse((2, 3), (&[0, 1, 2, 3], &[-1, 0, 1]), &values, false),
            sparse((2, 3), (&[0, 1, 3], &[1, 0, 3]), &[5.0, 6.0, 7.0], true),
            // Indices and values of different lengths.
            sparse((2, 3), (&[0, 1, 2, 3], &[1, 0]), &values, false),
            sparse((-2, 3), (&[0, 1, 2, 3], &[1, 0, 1]), &values, false),
        ];
        for matrix in &malformed {
            let error = matrix_rows(matrix).expect_err("a malformed matrix must fail");
            assert!(
                error.to_string().contains("malformed matrix"),
                "unexpected error {error}"
            );
        }
    }
}
