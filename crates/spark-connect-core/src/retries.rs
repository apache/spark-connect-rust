//! Retry policy and backoff logic.
//!
//! Mirrors `pyspark.sql.connect.client.retries.RetryPolicy` and friends.
//! The backoff computation is deterministic and unit-testable without sleeping.

/// Default maximum cumulative elapsed time for retry exception retries.
pub const DEFAULT_MAX_RETRY_EXCEPTION_ELAPSED_TIME: u64 = 60 * 60; // 1 hour in seconds

/// Describes how retries should be performed.
///
/// Mirrors `pyspark.sql.connect.client.retries.RetryPolicy`.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of retries.
    pub max_retries: Option<u32>,
    /// Initial backoff in milliseconds.
    pub initial_backoff_ms: u64,
    /// Maximum backoff in milliseconds.
    pub max_backoff_ms: Option<u64>,
    /// Multiplier for exponential backoff.
    pub backoff_multiplier: f64,
    /// Random jitter to add to backoff (in milliseconds).
    pub jitter_ms: u64,
    /// Minimum backoff threshold to apply jitter (in milliseconds).
    pub min_jitter_threshold_ms: u64,
    /// Whether to recognize server-provided retry delays.
    pub recognize_server_retry_delay: bool,
    /// Maximum server-provided retry delay (in milliseconds).
    pub max_server_retry_delay_ms: Option<u64>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: Some(15),
            initial_backoff_ms: 1000,
            max_backoff_ms: Some(64000),
            backoff_multiplier: 2.0,
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            recognize_server_retry_delay: false,
            max_server_retry_delay_ms: None,
        }
    }
}

impl RetryPolicy {
    /// Whether a failed RPC is retryable under this policy.
    ///
    /// Mirrors `pyspark.sql.connect.client.retries.DefaultPolicy.can_retry`:
    /// retry transient `UNAVAILABLE`, and `INTERNAL` errors whose message carries
    /// `INVALID_CURSOR.DISCONNECTED` (a mid-stream cursor drop resumed via reattach).
    pub fn can_retry(&self, status: &tonic::Status) -> bool {
        match status.code() {
            tonic::Code::Unavailable => true,
            tonic::Code::Internal => status.message().contains("INVALID_CURSOR.DISCONNECTED"),
            _ => false,
        }
    }
}

/// Stateful retry attempt tracker.
///
/// Mirrors `pyspark.sql.connect.client.retries.RetryPolicyState`.
pub struct RetryPolicyState {
    policy: RetryPolicy,
    attempt: u32,
    next_wait_ms: f64,
}

impl RetryPolicyState {
    pub fn new(policy: RetryPolicy) -> Self {
        Self {
            next_wait_ms: policy.initial_backoff_ms as f64,
            policy,
            attempt: 0,
        }
    }

    /// Compute the wait time before the next retry.
    ///
    /// Returns the number of milliseconds to wait, or None if no more retries are allowed.
    /// This is deterministic and doesn't perform any actual sleeping.
    pub fn next_attempt(&mut self, server_retry_delay_ms: Option<u64>) -> Option<u64> {
        // Check if we've exhausted the retry budget
        if let Some(max) = self.policy.max_retries {
            if self.attempt >= max {
                return None;
            }
        }

        self.attempt += 1;
        let mut wait_time = self.next_wait_ms;

        // Calculate next backoff for future attempts
        if let Some(max_backoff) = self.policy.max_backoff_ms {
            self.next_wait_ms = f64::min(
                max_backoff as f64,
                wait_time * self.policy.backoff_multiplier,
            );
        } else {
            self.next_wait_ms = wait_time * self.policy.backoff_multiplier;
        }

        // Honor server-provided retry delay if configured
        if self.policy.recognize_server_retry_delay {
            if let Some(delay) = server_retry_delay_ms {
                let max_delay = self.policy.max_server_retry_delay_ms.unwrap_or(delay);
                let delay = u64::min(delay, max_delay);
                wait_time = f64::max(wait_time, delay as f64);
            }
        }

        // Add jitter if wait_time meets the threshold
        if wait_time >= self.policy.min_jitter_threshold_ms as f64 {
            wait_time += rand_jitter(self.policy.jitter_ms);
        }

        // Round to whole milliseconds
        Some(wait_time.ceil() as u64)
    }

    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }

    pub fn attempt(&self) -> u32 {
        self.attempt
    }
}

/// A pseudo-random jitter value in `[0, max)` milliseconds.
///
/// Mirrors `random.uniform(0, jitter)` in the reference `RetryPolicyState`. Retry
/// jitter needs no cryptographic strength, so this is a clock-seeded xorshift with
/// no external RNG dependency.
fn rand_jitter(max: u64) -> f64 {
    if max == 0 {
        return 0.0;
    }
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        | 1;
    // xorshift64
    let mut x = seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    let frac = (x >> 11) as f64 / (1u64 << 53) as f64; // uniform in [0, 1)
    frac * (max as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_policy_backoff_sequence() {
        let policy = RetryPolicy {
            max_retries: Some(5),
            initial_backoff_ms: 100,
            max_backoff_ms: Some(1000),
            backoff_multiplier: 2.0,
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            recognize_server_retry_delay: false,
            max_server_retry_delay_ms: None,
        };

        let mut state = RetryPolicyState::new(policy);

        // Sequence should be: 100, 200, 400, 800, 1000 (capped)
        assert_eq!(state.next_attempt(None), Some(100));
        assert_eq!(state.next_attempt(None), Some(200));
        assert_eq!(state.next_attempt(None), Some(400));
        assert_eq!(state.next_attempt(None), Some(800));
        assert_eq!(state.next_attempt(None), Some(1000));
        // Now exhausted
        assert_eq!(state.next_attempt(None), None);
    }

    #[test]
    fn test_can_retry_classification() {
        let p = RetryPolicy::default();
        // Transient UNAVAILABLE is retryable.
        assert!(p.can_retry(&tonic::Status::unavailable("server restarting")));
        // INTERNAL is retryable only for a disconnected cursor (reattach path).
        assert!(p.can_retry(&tonic::Status::internal(
            "INVALID_CURSOR.DISCONNECTED: stream dropped"
        )));
        assert!(!p.can_retry(&tonic::Status::internal("some other internal error")));
        // Other codes are not retryable.
        assert!(!p.can_retry(&tonic::Status::not_found("missing")));
        assert!(!p.can_retry(&tonic::Status::invalid_argument("bad")));
    }

    #[test]
    fn test_jitter_within_bounds() {
        // rand_jitter must stay in [0, max) and vary (not a constant).
        for _ in 0..50 {
            let j = rand_jitter(100);
            assert!((0.0..100.0).contains(&j), "jitter {j} out of range");
        }
        assert_eq!(rand_jitter(0), 0.0);
    }

    #[test]
    fn test_retry_policy_respects_max_retries() {
        let policy = RetryPolicy {
            max_retries: Some(2),
            initial_backoff_ms: 50,
            max_backoff_ms: None,
            backoff_multiplier: 1.0,
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            recognize_server_retry_delay: false,
            max_server_retry_delay_ms: None,
        };

        let mut state = RetryPolicyState::new(policy);
        assert_eq!(state.next_attempt(None), Some(50));
        assert_eq!(state.next_attempt(None), Some(50));
        assert_eq!(state.next_attempt(None), None);
    }

    #[test]
    fn test_retry_policy_no_max_retries() {
        let policy = RetryPolicy {
            max_retries: None,
            initial_backoff_ms: 10,
            max_backoff_ms: Some(100),
            backoff_multiplier: 2.0,
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            recognize_server_retry_delay: false,
            max_server_retry_delay_ms: None,
        };

        let mut state = RetryPolicyState::new(policy);
        // Should be able to retry indefinitely (up to max_backoff)
        for _ in 0..10 {
            assert!(state.next_attempt(None).is_some());
        }
    }

    #[test]
    fn test_server_retry_delay_recognized() {
        let policy = RetryPolicy {
            max_retries: Some(3),
            initial_backoff_ms: 100,
            max_backoff_ms: Some(1000),
            backoff_multiplier: 2.0,
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            recognize_server_retry_delay: true,
            max_server_retry_delay_ms: Some(500),
        };

        let mut state = RetryPolicyState::new(policy);

        // First attempt with server delay > client backoff
        let wait = state.next_attempt(Some(250));
        assert_eq!(wait, Some(250)); // server delay wins

        // Second attempt with server delay < client backoff
        let wait = state.next_attempt(Some(50));
        assert_eq!(wait, Some(200)); // client backoff wins
    }

    #[test]
    fn test_attempt_count() {
        let policy = RetryPolicy {
            max_retries: Some(3),
            initial_backoff_ms: 50,
            max_backoff_ms: None,
            backoff_multiplier: 1.0,
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            recognize_server_retry_delay: false,
            max_server_retry_delay_ms: None,
        };

        let mut state = RetryPolicyState::new(policy);
        assert_eq!(state.attempt(), 0);
        state.next_attempt(None);
        assert_eq!(state.attempt(), 1);
        state.next_attempt(None);
        assert_eq!(state.attempt(), 2);
    }
}
