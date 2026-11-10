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
    /// Maximum cumulative elapsed time for the retries of a `RetryException`, in milliseconds.
    /// Defaults to 1 hour. Set to None to disable the ceiling. Not applied to the retries of a
    /// policy, as the reference's `RetryPolicyState` does not.
    pub max_retry_exception_elapsed_time_ms: Option<u64>,
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
            max_retry_exception_elapsed_time_ms: Some(
                DEFAULT_MAX_RETRY_EXCEPTION_ELAPSED_TIME * 1000,
            ),
        }
    }
}

impl RetryPolicy {
    /// A policy that performs a single attempt with no retries.
    ///
    /// Used by the transport-injection stub, which is a drop-in for the grpcio stub: the
    /// reference client layers its own `GrpcRetryHandler` (configured by the client-side
    /// retry policy) on top, so retrying here too would double-retry and ignore that
    /// policy - e.g. a `max_retries=0` client would still see our 15-retry backoff.
    pub fn no_retries() -> Self {
        Self {
            max_retries: Some(0),
            ..Self::default()
        }
    }

    /// Whether a failed RPC is retryable under this policy.
    ///
    /// Mirrors `pyspark.sql.connect.client.retries.DefaultPolicy.can_retry`:
    /// retry transient `UNAVAILABLE`, and `INTERNAL` errors whose message carries
    /// `INVALID_CURSOR.DISCONNECTED` (a mid-stream cursor drop resumed via reattach).
    /// A connection that closes under a call is `UNAVAILABLE` in grpc-python, so it is retried
    /// too: tonic reports a reset during a call as `UNKNOWN`, and a call queued on the
    /// connection but not sent yet as `CANCELLED`.
    pub fn can_retry(&self, status: &tonic::Status) -> bool {
        match status.code() {
            tonic::Code::Unavailable => true,
            tonic::Code::Internal => status.message().contains("INVALID_CURSOR.DISCONNECTED"),
            tonic::Code::Unknown => is_transport_error(status),
            tonic::Code::Cancelled => is_unsent_request(status),
            _ => false,
        }
    }
}

/// The errors that tonic keeps as the source of a status that it built from a transport error.
/// A status the server sent has none.
fn transport_errors(
    status: &tonic::Status,
) -> impl Iterator<Item = &(dyn std::error::Error + 'static)> + '_ {
    std::iter::successors(std::error::Error::source(status), |error| error.source())
}

/// Whether tonic built the status from a transport error.
fn is_transport_error(status: &tonic::Status) -> bool {
    transport_errors(status).any(|error| error.is::<h2::Error>() || error.is::<hyper::Error>())
}

/// Whether tonic built the status from a request that hyper canceled, which hyper does only for
/// a request that its connection closed before sending. Neither tonic's timeout of a request
/// nor a cancellation that the server sent, both `CANCELLED` too, is one.
fn is_unsent_request(status: &tonic::Status) -> bool {
    transport_errors(status).any(|error| {
        error
            .downcast_ref::<hyper::Error>()
            .is_some_and(hyper::Error::is_canceled)
    })
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
    /// This is deterministic and doesn't perform any actual sleeping. As the reference's
    /// `RetryPolicyState.next_attempt`, only `max_retries` bounds the retries here; the elapsed
    /// time bounds the retries of a `RetryException`, which the client checks itself.
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
            max_retry_exception_elapsed_time_ms: None,
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
    fn test_no_retries_policy_allows_a_single_attempt() {
        // no_retries() (max_retries = 0) means the first/only attempt is never retried,
        // so the very first next_attempt() already returns None.
        let policy = RetryPolicy::no_retries();
        assert_eq!(policy.max_retries, Some(0));
        let mut state = RetryPolicyState::new(policy);
        assert_eq!(state.next_attempt(None), None);
    }

    #[test]
    fn test_next_attempt_adds_jitter_above_threshold() {
        // With jitter enabled and the threshold met, the returned wait is the base
        // backoff plus a (rounded-up) jitter contribution in [0, jitter_ms).
        let policy = RetryPolicy {
            max_retries: Some(3),
            initial_backoff_ms: 100,
            max_backoff_ms: Some(1000),
            backoff_multiplier: 2.0,
            jitter_ms: 40,
            min_jitter_threshold_ms: 10,
            recognize_server_retry_delay: false,
            max_server_retry_delay_ms: None,
            max_retry_exception_elapsed_time_ms: None,
        };
        let mut state = RetryPolicyState::new(policy);
        let w = state.next_attempt(None).expect("first retry allowed");
        // jitter is [0, 40); base 100 + jitter, rounded up, lands in [100, 140].
        assert!(
            (100..=140).contains(&w),
            "expected 100..=140 with jitter, got {w}"
        );
        assert_eq!(state.attempt(), 1);
    }

    #[test]
    fn test_next_attempt_honors_server_retry_delay() {
        // recognize_server_retry_delay=true raises the wait to the server-provided delay
        // (capped by max_server_retry_delay_ms).
        let policy = RetryPolicy {
            max_retries: Some(3),
            initial_backoff_ms: 100,
            max_backoff_ms: Some(10_000),
            backoff_multiplier: 2.0,
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            recognize_server_retry_delay: true,
            max_server_retry_delay_ms: Some(5_000),
            max_retry_exception_elapsed_time_ms: None,
        };
        let mut state = RetryPolicyState::new(policy);
        // Server asks for 2s; that exceeds the 100ms base backoff, so it wins.
        assert_eq!(state.next_attempt(Some(2_000)), Some(2_000));
        // A server delay above the cap is clamped to max_server_retry_delay_ms.
        let w = state
            .next_attempt(Some(9_999))
            .expect("second retry allowed");
        assert!(
            w <= 5_000,
            "server delay must be clamped to the cap, got {w}"
        );
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
            max_retry_exception_elapsed_time_ms: None,
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
            max_retry_exception_elapsed_time_ms: None,
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
            max_retry_exception_elapsed_time_ms: None,
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
            max_retry_exception_elapsed_time_ms: None,
        };

        let mut state = RetryPolicyState::new(policy);
        assert_eq!(state.attempt(), 0);
        state.next_attempt(None);
        assert_eq!(state.attempt(), 1);
        state.next_attempt(None);
        assert_eq!(state.attempt(), 2);
    }

    #[test]
    fn test_elapsed_time_ceiling_does_not_cut_policy_retries() {
        // With max_retry_exception_elapsed_time_ms: Some(0), the elapsed bound of the
        // RetryExceptions has passed at once, but a policy retry is bounded by max_retries
        // alone, as the reference's RetryPolicyState: the retry is allowed.
        let policy = RetryPolicy {
            max_retries: Some(10),
            initial_backoff_ms: 50,
            max_backoff_ms: None,
            backoff_multiplier: 1.0,
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            recognize_server_retry_delay: false,
            max_server_retry_delay_ms: None,
            max_retry_exception_elapsed_time_ms: Some(0),
        };

        let mut state = RetryPolicyState::new(policy);
        assert_eq!(state.next_attempt(None), Some(50));
        assert_eq!(state.attempt(), 1);
    }

    #[test]
    fn test_policy_retries_after_the_elapsed_bound_until_max_retries() {
        // A wait past the elapsed bound: the retries of the policy are still those of max_retries.
        let policy = RetryPolicy {
            max_retries: Some(2),
            initial_backoff_ms: 50,
            max_backoff_ms: None,
            backoff_multiplier: 1.0,
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            recognize_server_retry_delay: false,
            max_server_retry_delay_ms: None,
            max_retry_exception_elapsed_time_ms: Some(1),
        };

        let mut state = RetryPolicyState::new(policy);
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert_eq!(state.next_attempt(None), Some(50));
        assert_eq!(state.next_attempt(None), Some(50));
        assert_eq!(state.next_attempt(None), None);
    }

    #[test]
    fn test_elapsed_time_ceiling_with_retries_allowed() {
        // With max_retry_exception_elapsed_time_ms: Some(u64::MAX), retries should be
        // limited by max_retries, not the elapsed-time ceiling.
        let policy = RetryPolicy {
            max_retries: Some(2),
            initial_backoff_ms: 50,
            max_backoff_ms: None,
            backoff_multiplier: 1.0,
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            recognize_server_retry_delay: false,
            max_server_retry_delay_ms: None,
            max_retry_exception_elapsed_time_ms: Some(u64::MAX),
        };

        let mut state = RetryPolicyState::new(policy);
        assert_eq!(state.next_attempt(None), Some(50));
        assert_eq!(state.next_attempt(None), Some(50));
        // max_retries exhausted
        assert_eq!(state.next_attempt(None), None);
    }

    /// Serves one response over HTTP/2, as a server does, and returns the address to call.
    /// With `reset`, the response is cut off by a connection reset in the middle of a gRPC
    /// message. Otherwise it ends with a status that the server sent: UNKNOWN "boom".
    async fn serve_one_response(reset: bool) -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (socket, _) = listener.accept().await.unwrap();
            if reset {
                // Closing the socket now sends a RST instead of a FIN.
                socket.set_zero_linger().unwrap();
            }
            let mut connection = h2::server::handshake(socket).await.unwrap();
            let (_request, mut respond) = connection.accept().await.unwrap().unwrap();
            let headers = tonic::codegen::http::Response::builder()
                .header("content-type", "application/grpc")
                .body(())
                .unwrap();
            let mut stream = respond.send_response(headers, false).unwrap();
            if reset {
                // The first 2 bytes of a 1000-byte gRPC message.
                let partial = bytes::Bytes::from_static(&[0, 0, 0, 3, 232, 1, 2]);
                stream.send_data(partial, false).unwrap();
            } else {
                let mut trailers = tonic::codegen::http::HeaderMap::new();
                trailers.insert("grpc-status", "2".parse().unwrap());
                trailers.insert("grpc-message", "boom".parse().unwrap());
                stream.send_trailers(trailers).unwrap();
            }
            // Keep the connection driven until the frames are written, then drop it.
            let _ =
                tokio::time::timeout(std::time::Duration::from_millis(200), connection.accept())
                    .await;
        });
        address
    }

    /// The status that ends the first response of a call to `address`.
    async fn status_of_first_response(address: std::net::SocketAddr) -> tonic::Status {
        let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))
            .unwrap()
            .connect()
            .await
            .unwrap();
        let mut stub =
            spark_connect_proto::spark_connect_service_client::SparkConnectServiceClient::new(
                channel,
            );
        let mut responses = stub
            .execute_plan(spark_connect_proto::ExecutePlanRequest::default())
            .await
            .unwrap()
            .into_inner();
        responses
            .message()
            .await
            .expect_err("the response ends with an error")
    }

    #[tokio::test]
    async fn test_can_retry_a_connection_reset_during_a_response() {
        let status = status_of_first_response(serve_one_response(true).await).await;
        // tonic reports the reset as UNKNOWN, with the transport error as its source.
        assert_eq!(status.code(), tonic::Code::Unknown);
        assert!(RetryPolicy::default().can_retry(&status), "{status:?}");
    }

    #[tokio::test]
    async fn test_cannot_retry_an_unknown_the_server_sent() {
        let status = status_of_first_response(serve_one_response(false).await).await;
        assert_eq!(status.code(), tonic::Code::Unknown);
        assert_eq!(status.message(), "boom");
        assert!(!RetryPolicy::default().can_retry(&status));
    }

    /// The error of a request queued on an HTTP/2 connection that closes before sending it, as
    /// a request that a reset of the connection catches before it is sent.
    async fn error_of_a_request_its_connection_closed_before_sending() -> hyper::Error {
        let (io, _server) = tokio::io::duplex(1 << 16);
        let (mut sender, connection) = hyper::client::conn::http2::handshake(
            hyper_util::rt::TokioExecutor::new(),
            hyper_util::rt::TokioIo::new(io),
        )
        .await
        .unwrap();
        let response = sender.send_request(tonic::codegen::http::Request::new(
            tonic::body::Body::empty(),
        ));
        drop(connection);
        response.await.expect_err("the connection closed")
    }

    #[tokio::test]
    async fn test_can_retry_a_request_its_connection_closed_before_sending() {
        let error = error_of_a_request_its_connection_closed_before_sending().await;
        assert!(error.is_canceled(), "{error:?}");
        // tonic reports it as CANCELLED, with hyper's error as its source.
        let status = tonic::Status::from_error(Box::new(error));
        assert_eq!(status.code(), tonic::Code::Cancelled);
        assert!(RetryPolicy::default().can_retry(&status), "{status:?}");
    }

    #[test]
    fn test_cannot_retry_a_cancellation_the_server_sent_or_a_timeout() {
        let p = RetryPolicy::default();
        assert!(!p.can_retry(&tonic::Status::cancelled("operation was canceled")));
        // tonic's timeout of a request is CANCELLED too.
        let timeout = tonic::Status::from_error(Box::new(tonic::TimeoutExpired(())));
        assert_eq!(timeout.code(), tonic::Code::Cancelled);
        assert!(!p.can_retry(&timeout));
    }
}
