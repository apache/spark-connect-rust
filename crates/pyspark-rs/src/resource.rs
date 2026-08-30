//! PyO3 wrappers for Spark resource profiles.

use pyo3::prelude::*;
use spark_connect::resource::{
    ExecutorResourceRequests, ResourceProfile, ResourceProfileBuilder, TaskResourceRequests,
};

/// Python wrapper for ExecutorResourceRequests.
#[pyclass(name = "ExecutorResourceRequests", module = "pyspark.resource.profile")]
pub struct PyExecutorResourceRequests {
    pub(crate) inner: ExecutorResourceRequests,
}

#[pymethods]
impl PyExecutorResourceRequests {
    /// Create a new empty ExecutorResourceRequests builder.
    #[new]
    fn new() -> Self {
        PyExecutorResourceRequests {
            inner: ExecutorResourceRequests::new(),
        }
    }

    /// Set the memory requirement in MB. Returns a new instance.
    fn memory(&self, memory_mb: i64) -> Self {
        PyExecutorResourceRequests {
            inner: self.inner.clone().memory(memory_mb),
        }
    }

    /// Set the off-heap memory requirement in MB. Returns a new instance.
    fn off_heap_memory(&self, memory_mb: i64) -> Self {
        PyExecutorResourceRequests {
            inner: self.inner.clone().off_heap_memory(memory_mb),
        }
    }

    /// Set the number of cores. Returns a new instance.
    fn cores(&self, num_cores: i64) -> Self {
        PyExecutorResourceRequests {
            inner: self.inner.clone().cores(num_cores),
        }
    }

    /// Add a custom resource request. Returns a new instance.
    #[pyo3(signature = (name, amount, discovery_script=None, vendor=None))]
    fn resource(
        &self,
        name: &str,
        amount: i64,
        discovery_script: Option<String>,
        vendor: Option<String>,
    ) -> Self {
        PyExecutorResourceRequests {
            inner: self
                .inner
                .clone()
                .resource(name, amount, discovery_script, vendor),
        }
    }
}

/// Python wrapper for TaskResourceRequests.
#[pyclass(name = "TaskResourceRequests", module = "pyspark.resource.profile")]
pub struct PyTaskResourceRequests {
    pub(crate) inner: TaskResourceRequests,
}

#[pymethods]
impl PyTaskResourceRequests {
    /// Create a new empty TaskResourceRequests builder.
    #[new]
    fn new() -> Self {
        PyTaskResourceRequests {
            inner: TaskResourceRequests::new(),
        }
    }

    /// Set the number of CPUs requested per task. Returns a new instance.
    fn cpus(&self, num_cpus: f64) -> Self {
        PyTaskResourceRequests {
            inner: self.inner.clone().cpus(num_cpus),
        }
    }

    /// Add a custom resource request per task. Returns a new instance.
    fn resource(&self, name: &str, amount: f64) -> Self {
        PyTaskResourceRequests {
            inner: self.inner.clone().resource(name, amount),
        }
    }
}

/// Python wrapper for ResourceProfileBuilder.
#[pyclass(name = "ResourceProfileBuilder", module = "pyspark.resource.profile")]
pub struct PyResourceProfileBuilder {
    pub(crate) inner: ResourceProfileBuilder,
}

#[pymethods]
impl PyResourceProfileBuilder {
    /// Create a new empty ResourceProfileBuilder.
    #[new]
    fn new() -> Self {
        PyResourceProfileBuilder {
            inner: ResourceProfileBuilder::new(),
        }
    }

    /// Set the executor resource requests. Returns a new instance.
    fn executor_resources(&self, requests: PyRef<'_, PyExecutorResourceRequests>) -> Self {
        PyResourceProfileBuilder {
            inner: self
                .inner
                .clone()
                .executor_resources(requests.inner.clone()),
        }
    }

    /// Set the task resource requests. Returns a new instance.
    fn task_resources(&self, requests: PyRef<'_, PyTaskResourceRequests>) -> Self {
        PyResourceProfileBuilder {
            inner: self.inner.clone().task_resources(requests.inner.clone()),
        }
    }

    /// Build the ResourceProfile.
    fn build(&self) -> PyResourceProfile {
        PyResourceProfile {
            inner: self.inner.clone().build(),
        }
    }
}

/// Python wrapper for ResourceProfile.
#[pyclass(name = "ResourceProfile", module = "pyspark.resource.profile")]
pub struct PyResourceProfile {
    pub(crate) inner: ResourceProfile,
}

#[pymethods]
impl PyResourceProfile {
    /// Get the profile id if this profile has been registered with the server.
    fn id(&self) -> Option<i32> {
        self.inner.id()
    }
}
