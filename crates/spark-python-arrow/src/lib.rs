// SPDX-License-Identifier: Apache-2.0

//! Python/Arrow conversion without a dependency on Spark transport or worker execution.
//!
//! The bridge shares Arrow data buffers; scalar and row materialization still allocates Python
//! objects. The caller owns its interpreter, Python classes, session settings, and error policy.
//! UDF-specific coercion and end-of-batch result conversion are explicit in the [`udf`] API.

pub mod bridge;
pub mod scalar;
pub mod udf;
