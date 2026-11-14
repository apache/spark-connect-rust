// SPDX-License-Identifier: Apache-2.0
//! The typed models and summaries of the decision tree, random forest and gradient-boosted tree classifiers: their
//! models and the random forest summaries.
use super::*;

/// Register the typed classes of the family.
pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _ = m;
    Ok(())
}
