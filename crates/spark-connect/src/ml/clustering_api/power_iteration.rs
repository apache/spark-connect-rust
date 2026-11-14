// SPDX-License-Identifier: Apache-2.0
//! The persistence of Power Iteration Clustering, as PySpark's `PowerIterationClustering` saves
//! and loads it: the params of the server's `PowerIterationClusteringWrapper` transformer, of the
//! uid of the algorithm.

use crate::ml::{read_operator, MlWriter, OperatorType, PowerIterationClustering};
use crate::session::SparkSession;
use spark_connect_core::error::Result;

const POWER_ITERATION_CLUSTERING_WRAPPER: &str =
    "org.apache.spark.ml.clustering.PowerIterationClusteringWrapper";

impl PowerIterationClustering {
    /// The uid of the algorithm (`uid`).
    pub fn uid(&self) -> &str {
        &self.operator.uid
    }

    /// A writer that saves the algorithm, of a session that [`MlWriter::session`] sets, as
    /// `PowerIterationClustering.write()` of PySpark.
    pub fn write(&self) -> MlWriter {
        MlWriter::operator(&self.operator, &self.params)
    }

    /// The algorithm saved at `path`, as `PowerIterationClustering.load` of PySpark.
    pub fn load(session: &SparkSession, path: &str) -> Result<Self> {
        let (operator, params) = read_operator(
            session,
            POWER_ITERATION_CLUSTERING_WRAPPER,
            OperatorType::Transformer,
            path,
        )?;
        Ok(Self { operator, params })
    }
}
