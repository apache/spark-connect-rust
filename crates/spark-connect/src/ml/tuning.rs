// SPDX-License-Identifier: Apache-2.0
//! The splits of the datasets of PySpark's validators of `pyspark.ml.tuning`, which they make on
//! the client:
//! - [`k_fold`]: the folds of `CrossValidator._kFold`, of a random split or of a fold column;
//! - [`train_validation_split`]: the training and validation sets of `TrainValidationSplit._fit`.
//!
//! Each split is of the plans of PySpark's, so that a split of the same seed has the rows of
//! PySpark's.
use super::*;
use crate::column::{lit, lit_double, lit_string};
use crate::functions::{call_udf, col, raise_error, rand, rand_with_seed};
use spark_connect_core::error::Result;

/// `dataset` and its column `<uid>_rand` of `rand(seed)`, as PySpark's
/// `dataset.select("*", F.rand(seed).alias(randCol))`, where `randCol = uid + "_rand"`: of a
/// random seed of no `seed`, as `F.rand(None)`. The name of the column is the second item.
fn with_rand_column(dataset: &DataFrame, seed: Option<i64>, uid: &str) -> (DataFrame, String) {
    let rand_col = format!("{uid}_rand");
    let random = match seed {
        Some(seed) => rand_with_seed(seed),
        None => rand(),
    };
    let df = dataset.select([col("*"), random.alias(&rand_col)]);
    (df, rand_col)
}

/// The folds of `dataset`, as `CrossValidator._kFold` makes them: `num_folds` pairs of a training
/// and a validation DataFrame, for the validator of the uid `uid`.
/// - Of an empty `fold_col`, a random split: the validation set of the fold `i` is the rows whose
///   column `<uid>_rand` of `rand(seed)` is in `[i / num_folds, (i + 1) / num_folds)`, and its
///   training set the others.
/// - Else the fold of each row is its `fold_col`, which must be in `[0, num_folds)`: a fold
///   number out of the range fails the queries of the folds (`raise_error`). A fold of an empty
///   training or validation set is an error, as PySpark raises it, after a query of each.
///
/// `num_folds` of 0 with no `fold_col` is an error (PySpark's `1.0 / nFolds` raises), and fewer
/// than 0 folds are none.
pub fn k_fold(
    dataset: &DataFrame,
    num_folds: i64,
    fold_col: &str,
    seed: Option<i64>,
    uid: &str,
) -> Result<Vec<(DataFrame, DataFrame)>> {
    let mut datasets = Vec::new();
    if fold_col.is_empty() {
        if num_folds == 0 {
            return Err(SparkError::value_msg("float division by zero"));
        }
        let h = 1.0 / num_folds as f64;
        let (df, rand_col) = with_rand_column(dataset, seed, uid);
        for i in 0..num_folds {
            let lower = i as f64 * h;
            let upper = (i + 1) as f64 * h;
            let condition = df
                .col(&rand_col)
                .ge(lit_double(lower))
                .and(df.col(&rand_col).lt(lit_double(upper)));
            let validation = df.filter(condition.clone());
            let train = df.filter(condition.not());
            datasets.push((train, validation));
        }
    } else {
        let message = format!("Fold number must be in range [0, {num_folds}), but got %s");
        let checked = dataset.with_column(
            fold_col,
            crate::column::when(
                lit(0)
                    .le(col(fold_col))
                    .and(col(fold_col).lt(lit(num_folds))),
                col(fold_col),
            )
            .otherwise(raise_error(call_udf(
                "printf",
                [lit_string(&message), col(fold_col)],
            ))),
        );
        for i in 0..num_folds {
            let training = checked.filter(col(fold_col).ne(lit(i)));
            let validation = checked.filter(col(fold_col).eq(lit(i)));
            if training.is_empty()? {
                return Err(SparkError::value_msg(format!(
                    "The training data at fold {i} is empty."
                )));
            }
            if validation.is_empty()? {
                return Err(SparkError::value_msg(format!(
                    "The validation data at fold {i} is empty."
                )));
            }
            datasets.push((training, validation));
        }
    }
    Ok(datasets)
}

/// The training and validation sets of `dataset`, as `TrainValidationSplit._fit` splits it for
/// the validator of the uid `uid`: the validation set is the rows whose column `<uid>_rand` of
/// `rand(seed)` is at least `train_ratio`, and the training set the others.
pub fn train_validation_split(
    dataset: &DataFrame,
    train_ratio: f64,
    seed: Option<i64>,
    uid: &str,
) -> (DataFrame, DataFrame) {
    let (df, rand_col) = with_rand_column(dataset, seed, uid);
    let condition = df.col(&rand_col).ge(lit_double(train_ratio));
    let validation = df.filter(condition.clone());
    let train = df.filter(condition.not());
    (train, validation)
}

#[cfg(test)]
mod tests {
    use super::*;

    // gRPC connects lazily, so the session builds offline, and the splits only build plans.
    fn session() -> SparkSession {
        SparkSession::builder()
            .remote("sc://localhost:15002")
            .get_or_create()
            .expect("session")
    }

    #[test]
    fn random_folds_filter_the_rand_column_of_the_seed_by_the_bounds_of_each_fold() {
        let df = session().range(10).expect("a range");
        let folds = k_fold(&df, 3, "", Some(7), "CrossValidator_1").expect("the folds");
        assert_eq!(folds.len(), 3);
        let (train, validation) = &folds[1];
        let plan = format!("{:?}", validation.plan());
        assert!(plan.contains("CrossValidator_1_rand"), "{plan}");
        assert!(plan.contains("rand"), "{plan}");
        assert!(plan.contains(">="), "{plan}");
        assert!(plan.contains(&format!("{:?}", 1.0 / 3.0)), "{plan}");
        assert!(plan.contains(&format!("{:?}", 2.0 / 3.0)), "{plan}");
        let train_plan = format!("{:?}", train.plan());
        assert!(train_plan.contains("\"not\""), "{train_plan}");
        assert!(k_fold(&df, 0, "", Some(7), "CrossValidator_1").is_err());
        assert!(k_fold(&df, -1, "", Some(7), "CrossValidator_1")
            .expect("no folds")
            .is_empty());
    }

    #[test]
    fn a_train_validation_split_is_of_the_rand_column_at_the_train_ratio() {
        let df = session().range(10).expect("a range");
        let (train, validation) = train_validation_split(&df, 0.75, None, "TVS_1");
        let plan = format!("{:?}", validation.plan());
        assert!(plan.contains("TVS_1_rand"), "{plan}");
        assert!(plan.contains("0.75"), "{plan}");
        assert!(format!("{:?}", train.plan()).contains("\"not\""));
    }
}
