//! Resource profile for specifying executor and task resource requirements.
//!
//! Mirrors `pyspark.resource` and allows building and registering resource profiles
//! on the Spark Connect server.

use spark_connect_proto as proto;
use std::collections::HashMap;

/// Builder for executor resource requests.
///
/// Mirrors `pyspark.resource.ExecutorResourceRequests`. Allows specifying
/// resources requested by executors, including memory, cores, and custom resources.
#[derive(Debug, Clone, Default)]
pub struct ExecutorResourceRequests {
    resources: HashMap<String, proto::ExecutorResourceRequest>,
}

impl ExecutorResourceRequests {
    /// Create a new empty ExecutorResourceRequests builder.
    pub fn new() -> Self {
        ExecutorResourceRequests {
            resources: HashMap::new(),
        }
    }

    /// Set the memory requirement in MB.
    pub fn memory(mut self, memory_mb: i64) -> Self {
        self.resources.insert(
            "memory".to_string(),
            proto::ExecutorResourceRequest {
                resource_name: "memory".to_string(),
                amount: memory_mb,
                discovery_script: None,
                vendor: None,
            },
        );
        self
    }

    /// Set the off-heap memory requirement in MB.
    pub fn off_heap_memory(mut self, memory_mb: i64) -> Self {
        self.resources.insert(
            "offHeap".to_string(),
            proto::ExecutorResourceRequest {
                resource_name: "offHeap".to_string(),
                amount: memory_mb,
                discovery_script: None,
                vendor: None,
            },
        );
        self
    }

    /// Set the number of cores.
    pub fn cores(mut self, num_cores: i64) -> Self {
        self.resources.insert(
            "cores".to_string(),
            proto::ExecutorResourceRequest {
                resource_name: "cores".to_string(),
                amount: num_cores,
                discovery_script: None,
                vendor: None,
            },
        );
        self
    }

    /// Add a custom resource request.
    ///
    /// # Arguments
    /// * `name` - The resource name (e.g., "gpu", "fpga")
    /// * `amount` - The amount of the resource being requested
    /// * `discovery_script` - Optional script to discover the resource on the executor
    /// * `vendor` - Optional vendor name for the resource
    pub fn resource(
        mut self,
        name: &str,
        amount: i64,
        discovery_script: Option<String>,
        vendor: Option<String>,
    ) -> Self {
        self.resources.insert(
            name.to_string(),
            proto::ExecutorResourceRequest {
                resource_name: name.to_string(),
                amount,
                discovery_script,
                vendor,
            },
        );
        self
    }
}

/// Builder for task resource requests.
///
/// Mirrors `pyspark.resource.TaskResourceRequests`. Allows specifying
/// resources requested per task, including cores and custom resources.
#[derive(Debug, Clone, Default)]
pub struct TaskResourceRequests {
    resources: HashMap<String, proto::TaskResourceRequest>,
}

impl TaskResourceRequests {
    /// Create a new empty TaskResourceRequests builder.
    pub fn new() -> Self {
        TaskResourceRequests {
            resources: HashMap::new(),
        }
    }

    /// Set the number of CPUs requested per task.
    pub fn cpus(mut self, num_cpus: f64) -> Self {
        self.resources.insert(
            "cpus".to_string(),
            proto::TaskResourceRequest {
                resource_name: "cpus".to_string(),
                amount: num_cpus,
            },
        );
        self
    }

    /// Add a custom resource request per task.
    ///
    /// # Arguments
    /// * `name` - The resource name (e.g., "gpu", "fpga")
    /// * `amount` - The fractional amount of the resource per task
    pub fn resource(mut self, name: &str, amount: f64) -> Self {
        self.resources.insert(
            name.to_string(),
            proto::TaskResourceRequest {
                resource_name: name.to_string(),
                amount,
            },
        );
        self
    }
}

/// Builder for ResourceProfile.
///
/// Mirrors `pyspark.resource.ResourceProfile`. Allows configuring both executor
/// and task resource requests, then building and registering the profile with Spark.
#[derive(Debug, Clone, Default)]
pub struct ResourceProfileBuilder {
    executor_requests: ExecutorResourceRequests,
    task_requests: TaskResourceRequests,
}

impl ResourceProfileBuilder {
    /// Create a new empty ResourceProfileBuilder.
    pub fn new() -> Self {
        ResourceProfileBuilder {
            executor_requests: ExecutorResourceRequests::new(),
            task_requests: TaskResourceRequests::new(),
        }
    }

    /// Set the executor resource requests.
    pub fn executor_resources(mut self, requests: ExecutorResourceRequests) -> Self {
        self.executor_requests = requests;
        self
    }

    /// Set the task resource requests.
    pub fn task_resources(mut self, requests: TaskResourceRequests) -> Self {
        self.task_requests = requests;
        self
    }

    /// Build the ResourceProfile.
    pub fn build(self) -> ResourceProfile {
        ResourceProfile {
            proto_profile: proto::ResourceProfile {
                executor_resources: self.executor_requests.resources.clone(),
                task_resources: self.task_requests.resources.clone(),
            },
            profile_id: None,
        }
    }
}

/// A Spark ResourceProfile, optionally with a server-assigned id.
///
/// Returned by a ResourceProfileBuilder.build() and registered with the server
/// via SparkSession::build_resource_profile().
#[derive(Debug, Clone)]
pub struct ResourceProfile {
    pub(crate) proto_profile: proto::ResourceProfile,
    pub(crate) profile_id: Option<i32>,
}

impl ResourceProfile {
    /// Get the profile id if this profile has been registered with the server.
    pub fn id(&self) -> Option<i32> {
        self.profile_id
    }

    /// Get a reference to the underlying proto ResourceProfile.
    pub(crate) fn proto(&self) -> &proto::ResourceProfile {
        &self.proto_profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn executor_resource_requests_encode_decode() {
        let reqs = ExecutorResourceRequests::new()
            .memory(2048)
            .cores(4)
            .resource("gpu", 2, None, Some("nvidia".to_string()));

        let profile = proto::ResourceProfile {
            executor_resources: reqs.resources.clone(),
            task_resources: HashMap::new(),
        };

        let encoded = profile.encode_to_vec();
        let decoded = proto::ResourceProfile::decode(encoded.as_slice()).unwrap();

        assert_eq!(decoded.executor_resources.len(), 3);
        assert_eq!(
            decoded.executor_resources.get("memory").unwrap().amount,
            2048
        );
        assert_eq!(decoded.executor_resources.get("cores").unwrap().amount, 4);
        assert_eq!(decoded.executor_resources.get("gpu").unwrap().amount, 2);
        assert_eq!(
            decoded.executor_resources.get("gpu").unwrap().vendor,
            Some("nvidia".to_string())
        );
    }

    #[test]
    fn task_resource_requests_encode_decode() {
        let reqs = TaskResourceRequests::new().cpus(0.5).resource("gpu", 0.25);

        let profile = proto::ResourceProfile {
            executor_resources: HashMap::new(),
            task_resources: reqs.resources.clone(),
        };

        let encoded = profile.encode_to_vec();
        let decoded = proto::ResourceProfile::decode(encoded.as_slice()).unwrap();

        assert_eq!(decoded.task_resources.len(), 2);
        assert_eq!(decoded.task_resources.get("cpus").unwrap().amount, 0.5);
        assert_eq!(decoded.task_resources.get("gpu").unwrap().amount, 0.25);
    }

    #[test]
    fn resource_profile_builder() {
        let executor_reqs = ExecutorResourceRequests::new().memory(4096).cores(8);
        let task_reqs = TaskResourceRequests::new().cpus(1.0);

        let profile = ResourceProfileBuilder::new()
            .executor_resources(executor_reqs)
            .task_resources(task_reqs)
            .build();

        assert!(profile.profile_id.is_none());
        assert_eq!(profile.proto_profile.executor_resources.len(), 2);
        assert_eq!(profile.proto_profile.task_resources.len(), 1);

        let executor_mem = profile
            .proto_profile
            .executor_resources
            .get("memory")
            .unwrap();
        assert_eq!(executor_mem.amount, 4096);

        let task_cpus = profile.proto_profile.task_resources.get("cpus").unwrap();
        assert_eq!(task_cpus.amount, 1.0);
    }

    #[test]
    fn off_heap_memory_and_profile_accessors() {
        let reqs = ExecutorResourceRequests::new()
            .off_heap_memory(1024)
            .cores(2);
        assert_eq!(reqs.resources.get("offHeap").unwrap().amount, 1024);

        let profile = ResourceProfileBuilder::new()
            .executor_resources(reqs)
            .build();
        // id() is None before the profile is registered with the server.
        assert!(profile.id().is_none());
        // proto() exposes the underlying proto with both resource entries.
        assert_eq!(profile.proto().executor_resources.len(), 2);
    }
}
