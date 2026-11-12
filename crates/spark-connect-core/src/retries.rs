//! Retry policy and backoff logic.
//!
//! Mirrors `pyspark.sql.connect.client.retries.RetryPolicy` and friends.
//! The backoff computation is deterministic and unit-testable without sleeping.

use prost::Message;

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
    /// The values of PySpark's `DefaultPolicy`, which the Scala client's `defaultPolicy()` shares.
    fn default() -> Self {
        Self {
            max_retries: Some(15),
            initial_backoff_ms: 50,
            max_backoff_ms: Some(60000),
            backoff_multiplier: 4.0,
            jitter_ms: 500,
            min_jitter_threshold_ms: 2000,
            recognize_server_retry_delay: true,
            max_server_retry_delay_ms: Some(10 * 60 * 1000),
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
    /// Mirrors `pyspark.sql.connect.client.retries.DefaultPolicy.can_retry`, on the code that
    /// grpc-python gives the failure (see [`grpc_python_code`]): retry transient `UNAVAILABLE`,
    /// and `INTERNAL` errors whose message carries `INVALID_CURSOR.DISCONNECTED` (a mid-stream
    /// cursor drop resumed via reattach). Any failure with a `RetryInfo` detail is retried too,
    /// whatever its code, as the server asks for it.
    pub fn can_retry(&self, status: &tonic::Status) -> bool {
        let transient = match grpc_python_code(status) {
            tonic::Code::Unavailable => true,
            tonic::Code::Internal => cursor_disconnected(status),
            _ => false,
        };
        transient || retry_info(status).is_some()
    }
}

/// The first of the error conditions `conditions` that the message of a failed RPC names.
///
/// The reference client recognizes these conditions by their name in the message of the error
/// (`"INVALID_CURSOR.DISCONNECTED" in str(e)` in `DefaultPolicy.can_retry`, and
/// `error in status.message` in `ExecutePlanResponseReattachableIterator`), not by the
/// `ErrorInfo` detail of the status, so this client does too: every such match goes through
/// here.
fn condition_in_message(
    status: &tonic::Status,
    conditions: &[&'static str],
) -> Option<&'static str> {
    conditions
        .iter()
        .copied()
        .find(|condition| status.message().contains(condition))
}

/// Whether the message of `status` names `INVALID_CURSOR.DISCONNECTED`: another RPC took over
/// the cursor of the response stream, which a reattach resumes.
pub(crate) fn cursor_disconnected(status: &tonic::Status) -> bool {
    condition_in_message(status, &["INVALID_CURSOR.DISCONNECTED"]).is_some()
}

/// The error class of `status` when the server does not know the operation of a reattach, or
/// its session, which the reference reads alike: the initial ExecutePlan never reached the
/// server, unless a response was read already.
pub(crate) fn unknown_operation(status: &tonic::Status) -> Option<&'static str> {
    condition_in_message(
        status,
        &[
            "INVALID_HANDLE.OPERATION_NOT_FOUND",
            "INVALID_HANDLE.SESSION_NOT_FOUND",
        ],
    )
}

/// The `google.rpc.Status` of a failed RPC, from its `grpc-status-details-bin`, whose details
/// hold Any-packed messages.
#[derive(Clone, PartialEq, Message)]
struct StatusDetails {
    #[prost(message, repeated, tag = "3")]
    details: Vec<prost_types::Any>,
}

/// The `google.rpc.RetryInfo` of a failed RPC: how long the server asks the client to wait.
#[derive(Clone, PartialEq, Message)]
struct RetryInfo {
    #[prost(message, optional, tag = "1")]
    retry_delay: Option<prost_types::Duration>,
}

/// The `google.rpc.RetryInfo` detail of a status, if it has one.
fn retry_info(status: &tonic::Status) -> Option<RetryInfo> {
    StatusDetails::decode(status.details())
        .ok()?
        .details
        .into_iter()
        .find(|detail| detail.type_url.rsplit('/').next() == Some("google.rpc.RetryInfo"))
        .and_then(|detail| RetryInfo::decode(detail.value.as_slice()).ok())
}

/// The delay that the server asks for in the `RetryInfo` detail of a status, in milliseconds.
///
/// Mirrors `pyspark.sql.connect.client.retries.extract_retry_delay`. A negative delay is zero.
pub fn retry_delay_ms(status: &tonic::Status) -> Option<u64> {
    let delay = retry_info(status)?.retry_delay?;
    let seconds = u64::try_from(delay.seconds).unwrap_or(0);
    let nanos = u64::try_from(delay.nanos).unwrap_or(0);
    Some(
        seconds
            .saturating_mul(1000)
            .saturating_add(nanos / 1_000_000),
    )
}

/// The code of a failed RPC as grpc-python reports it, which PySpark's retries and errors see.
/// A status that the server sent keeps its code. tonic gives some transport failures other
/// codes: a connection that closes under a call is `UNAVAILABLE` in grpc-python (tonic:
/// `UNKNOWN`), and so are a connection that the server ends with a GOAWAY (tonic: the code of
/// its reason) and a call queued on the connection but not sent yet (tonic: `CANCELLED`). A
/// stream that the server resets has the code of its reason in both, except for the reasons
/// that tonic leaves `UNKNOWN`, such as `STREAM_CLOSED`, which grpc-python reports as `INTERNAL`.
pub fn grpc_python_code(status: &tonic::Status) -> tonic::Code {
    let code = status.code();
    match transport_errors(status).find_map(|error| error.downcast_ref::<h2::Error>()) {
        Some(error) if error.is_reset() => match code {
            tonic::Code::Unknown => tonic::Code::Internal,
            _ => code,
        },
        Some(error) if error.is_go_away() => tonic::Code::Unavailable,
        _ => match code {
            tonic::Code::Unknown if is_transport_error(status) => tonic::Code::Unavailable,
            tonic::Code::Cancelled if is_unsent_request(status) => tonic::Code::Unavailable,
            _ => code,
        },
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

        // Round down to whole milliseconds, as the reference's int()
        Some(wait_time.floor() as u64)
    }

    pub fn policy(&self) -> &RetryPolicy {
        &self.policy
    }

    pub fn attempt(&self) -> u32 {
        self.attempt
    }
}

/// The retries of one call, as PySpark's `Retrying`: the one place that decides whether a
/// failed attempt is made again, and that waits before it is.
///
/// A failure is retried in one of two ways, as in the reference:
///
/// - [`Retrying::backoff`]: a failed RPC that the policy can retry
///   ([`RetryPolicy::can_retry`]) is an attempt of the policy. It waits the backoff of the
///   attempt (or the delay that the server asks for) and is bounded by `max_retries`.
/// - [`Retrying::retry_at_once`]: what the reference raises as a `RetryException` (a stream of
///   an execution past its deadline, or an execution that the server does not know yet) is
///   retried without a wait and outside the attempts of the policy, for up to
///   `max_retry_exception_elapsed_time_ms` from the first of them.
///
/// [`retrying`] is the loop over the attempts of a call. A caller whose attempts are not one
/// call each, as the reader of a reattachable stream, asks the two methods itself.
pub struct Retrying {
    state: RetryPolicyState,
    /// When the first retry made at once was asked for.
    retry_exception_since: Option<std::time::Instant>,
    /// Whether a signal that the wait of the caller takes ends a backoff.
    ends_with_abort: bool,
}

/// What [`Retrying::backoff`] decides for a failed attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backoff {
    /// The wait before the retry is over: the attempt is made again.
    Retry,
    /// The policy cannot retry the failure, or its retries are spent: the failure is the error
    /// of the call.
    GiveUp,
    /// A signal that the wait of the caller takes ended the wait before the retry, which is not
    /// made (see [`crate::runtime::until_abort`]).
    Aborted,
}

impl Retrying {
    /// The retries of a call under `policy`, whose backoffs are waited whole.
    pub fn new(policy: &RetryPolicy) -> Self {
        Self {
            state: RetryPolicyState::new(policy.clone()),
            retry_exception_since: None,
            ends_with_abort: false,
        }
    }

    /// These retries, whose backoffs a signal of the process ends when the wait of the caller
    /// takes it ([`crate::runtime::ABORT_RETRIES`]), as the sleep of PySpark's `Retrying`
    /// raises KeyboardInterrupt. For the calls that a caller waits for; a call in the
    /// background, that no caller waits for, waits its backoffs whole.
    pub fn ending_with_abort(mut self) -> Self {
        self.ends_with_abort = true;
        self
    }

    /// Whether the attempt that failed with `status` is made again under the policy, after the
    /// wait before it, which this awaits. It is not when the policy cannot retry the failure,
    /// or once its retries are spent: `status` is then the error of the call.
    pub async fn backoff(&mut self, status: &tonic::Status) -> Backoff {
        if !self.state.policy().can_retry(status) {
            return Backoff::GiveUp;
        }
        let Some(wait_ms) = self.state.next_attempt(retry_delay_ms(status)) else {
            return Backoff::GiveUp;
        };
        let wait = tokio::time::sleep(std::time::Duration::from_millis(wait_ms));
        if !self.ends_with_abort {
            wait.await;
            return Backoff::Retry;
        }
        match crate::runtime::until_abort(crate::runtime::ABORT_RETRIES, wait).await {
            Some(()) => Backoff::Retry,
            None => Backoff::Aborted,
        }
    }

    /// Whether a failure that the reference retries as a `RetryException` is retried, at once:
    /// until such retries have run for the policy's elapsed-time bound, measured from the first
    /// of them, which this call is when there was none.
    pub fn retry_at_once(&mut self) -> bool {
        let since = *self
            .retry_exception_since
            .get_or_insert_with(std::time::Instant::now);
        !self
            .state
            .policy()
            .max_retry_exception_elapsed_time_ms
            .is_some_and(|ms| since.elapsed() >= std::time::Duration::from_millis(ms))
    }
}

/// Why a call made by [`retrying`] failed, and what an attempt of it fails with.
#[derive(Debug)]
pub enum RetryError<E> {
    /// A failed RPC. From an attempt, it is retried when the policy allows; from [`retrying`],
    /// the policy could not retry it, or its retries are spent.
    Status(tonic::Status),
    /// An error that is not retried: of an attempt, or of the check before a retry.
    Aborted(E),
    /// A signal ended the wait before a retry ([`Backoff::Aborted`]). Only of retries
    /// [`Retrying::ending_with_abort`].
    WaitAborted,
}

/// Makes the attempts of a call under the policy of `retrying` until one succeeds or fails for
/// good: a failed RPC is retried as [`Retrying::backoff`] decides, after its wait.
///
/// What differs between the calls is given rather than built in:
///
/// - `retrying` says whether a signal ends the backoffs ([`Retrying::ending_with_abort`]).
/// - `before_retry` runs after the wait of each retry, before its attempt, and its error ends
///   the call. The calls of a client check there that the client is still open, as the
///   reference's `_stub` raises for a closed one; the release of an execution, which the
///   reference sends through the stub of its iterator, checks nothing.
/// - `attempt` fails with [`RetryError::Aborted`] for an error that is no failed RPC, such as
///   the unreadable source of an upload, which ends the call as it is.
pub async fn retrying<T, E, Fut>(
    mut retrying: Retrying,
    mut before_retry: impl FnMut() -> Result<(), E>,
    mut attempt: impl FnMut() -> Fut,
) -> Result<T, RetryError<E>>
where
    Fut: std::future::Future<Output = Result<T, RetryError<E>>>,
{
    loop {
        match attempt().await {
            Err(RetryError::Status(status)) => {
                match retrying.backoff(&status).await {
                    Backoff::Retry => {}
                    Backoff::GiveUp => return Err(RetryError::Status(status)),
                    Backoff::Aborted => return Err(RetryError::WaitAborted),
                }
                before_retry().map_err(RetryError::Aborted)?;
            }
            result => return result,
        }
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

    /// A status of `code` with a `RetryInfo` detail that asks for a delay of `seconds` and
    /// `nanos`, as the server sends it in `grpc-status-details-bin`.
    fn status_with_retry_info(code: tonic::Code, seconds: i64, nanos: i32) -> tonic::Status {
        let info = RetryInfo {
            retry_delay: Some(prost_types::Duration { seconds, nanos }),
        };
        let detail = prost_types::Any {
            type_url: "type.googleapis.com/google.rpc.RetryInfo".to_string(),
            value: info.encode_to_vec(),
        };
        let details = StatusDetails {
            details: vec![detail],
        };
        tonic::Status::with_details(code, "retry", details.encode_to_vec().into())
    }

    #[test]
    fn test_default_schedule_without_jitter() {
        // The waits of PySpark's DefaultPolicy with the jitter off: 50 ms, quadrupling up to
        // the 60 s maximum, over the 15 retries.
        let policy = RetryPolicy {
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            ..RetryPolicy::default()
        };
        let mut state = RetryPolicyState::new(policy);
        let mut waits = Vec::new();
        while let Some(wait) = state.next_attempt(None) {
            waits.push(wait);
        }
        let mut expected = vec![50, 200, 800, 3200, 12800, 51200];
        expected.extend([60000; 9]);
        assert_eq!(waits, expected);
        assert_eq!(waits.iter().sum::<u64>(), 608250);
    }

    #[test]
    fn test_default_jitter_applies_only_at_or_above_the_threshold() {
        // The waits below the 2 s threshold get no jitter. The 3.2 s wait gets one in [0, 500) ms.
        let mut jittered = Vec::new();
        for _ in 0..20 {
            let mut state = RetryPolicyState::new(RetryPolicy::default());
            assert_eq!(state.next_attempt(None), Some(50));
            assert_eq!(state.next_attempt(None), Some(200));
            assert_eq!(state.next_attempt(None), Some(800));
            let wait = state.next_attempt(None).expect("fourth retry allowed");
            assert!(
                (3200..3700).contains(&wait),
                "expected 3200..3700, got {wait}"
            );
            jittered.push(wait);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(jittered.iter().any(|&wait| wait > 3200), "{jittered:?}");
        // A wait of the max backoff is jittered too, in [60000, 60500).
        let mut state = RetryPolicyState::new(RetryPolicy::default());
        for _ in 0..8 {
            state.next_attempt(None);
        }
        let wait = state.next_attempt(None).expect("last retry allowed");
        assert!(
            (60000..60500).contains(&wait),
            "expected 60000..60500, got {wait}"
        );
    }

    #[test]
    fn test_server_retry_delay_from_retry_info_is_honored() {
        let policy = RetryPolicy {
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            ..RetryPolicy::default()
        };
        // A RetryInfo of 2.5 s is the delay of the retry, above the 50 ms base wait.
        let status = status_with_retry_info(tonic::Code::Unavailable, 2, 500_000_000);
        assert_eq!(retry_delay_ms(&status), Some(2500));
        let mut state = RetryPolicyState::new(policy.clone());
        assert_eq!(state.next_attempt(retry_delay_ms(&status)), Some(2500));
        // A delay above max_server_retry_delay (10 minutes) is capped to it.
        let status = status_with_retry_info(tonic::Code::Unavailable, 3600, 0);
        assert_eq!(retry_delay_ms(&status), Some(3_600_000));
        assert_eq!(state.next_attempt(retry_delay_ms(&status)), Some(600_000));
        // A status without a RetryInfo has no delay, and the base wait applies.
        let status = tonic::Status::unavailable("server restarting");
        assert_eq!(retry_delay_ms(&status), None);
        assert_eq!(state.next_attempt(retry_delay_ms(&status)), Some(800));
        // The delay is ignored when the policy does not recognize it.
        let policy = RetryPolicy {
            recognize_server_retry_delay: false,
            ..policy
        };
        let status = status_with_retry_info(tonic::Code::Unavailable, 2, 0);
        let mut state = RetryPolicyState::new(policy);
        assert_eq!(state.next_attempt(retry_delay_ms(&status)), Some(50));
    }

    #[test]
    fn test_server_retry_delay_with_default_jitter_is_jittered_above_the_threshold() {
        // The 2.5 s delay is above the 2 s threshold, so it gets the jitter of [0, 500) ms.
        let status = status_with_retry_info(tonic::Code::Unavailable, 2, 500_000_000);
        for _ in 0..20 {
            let mut state = RetryPolicyState::new(RetryPolicy::default());
            let wait = state
                .next_attempt(retry_delay_ms(&status))
                .expect("allowed");
            assert!(
                (2500..3000).contains(&wait),
                "expected 2500..3000, got {wait}"
            );
        }
    }

    #[test]
    fn test_can_retry_a_status_with_retry_info_whatever_its_code() {
        // As PySpark's DefaultPolicy, a RetryInfo of a failure makes it retryable.
        let p = RetryPolicy::default();
        assert!(p.can_retry(&status_with_retry_info(
            tonic::Code::ResourceExhausted,
            1,
            0
        )));
        assert!(p.can_retry(&status_with_retry_info(tonic::Code::Internal, 0, 0)));
        // The same codes without the RetryInfo are not retried.
        assert!(!p.can_retry(&tonic::Status::new(tonic::Code::ResourceExhausted, "retry")));
        // Another detail of the status is not a RetryInfo.
        let detail = prost_types::Any {
            type_url: "type.googleapis.com/google.rpc.ErrorInfo".to_string(),
            value: Vec::new(),
        };
        let details = StatusDetails {
            details: vec![detail],
        };
        let status = tonic::Status::with_details(
            tonic::Code::ResourceExhausted,
            "retry",
            details.encode_to_vec().into(),
        );
        assert!(!p.can_retry(&status));
        assert_eq!(retry_delay_ms(&status), None);
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
        // tonic reports the reset as UNKNOWN, with the transport error as its source, and
        // grpc-python as UNAVAILABLE, which the error of the client reports too.
        assert_eq!(status.code(), tonic::Code::Unknown);
        assert_eq!(grpc_python_code(&status), tonic::Code::Unavailable);
        assert!(RetryPolicy::default().can_retry(&status), "{status:?}");
        let error = crate::error::SparkError::from_grpc_status(status);
        assert_eq!(error.grpc_code(), Some(tonic::Code::Unavailable as i32));
    }

    #[tokio::test]
    async fn test_cannot_retry_an_unknown_the_server_sent() {
        let status = status_of_first_response(serve_one_response(false).await).await;
        assert_eq!(status.code(), tonic::Code::Unknown);
        assert_eq!(status.message(), "boom");
        // A status that the server sent keeps its code.
        assert_eq!(grpc_python_code(&status), tonic::Code::Unknown);
        assert!(!RetryPolicy::default().can_retry(&status));
        let error = crate::error::SparkError::from_grpc_status(status);
        assert_eq!(error.grpc_code(), Some(tonic::Code::Unknown as i32));
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
        assert_eq!(grpc_python_code(&status), tonic::Code::Unavailable);
        assert!(RetryPolicy::default().can_retry(&status), "{status:?}");
    }

    /// How a server answers the request headers of a call with one HTTP/2 frame.
    #[derive(Clone, Copy, Debug)]
    enum Answer {
        /// A RST_STREAM of the stream of the call, of this error code.
        Reset(h2::Reason),
        /// A GOAWAY of this error code that processed no stream.
        GoAway(h2::Reason),
    }

    /// Serves one connection in raw HTTP/2 frames, which answers the request of a call with
    /// `answer`, and then reads the frames of the client until it closes the connection: the
    /// frame, not the end of the connection, ends the call.
    async fn serve_one_frame(answer: Answer) -> std::net::SocketAddr {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        fn frame(kind: u8, flags: u8, stream: u32, payload: &[u8]) -> Vec<u8> {
            let length = (payload.len() as u32).to_be_bytes();
            let mut frame = vec![length[1], length[2], length[3], kind, flags];
            frame.extend_from_slice(&stream.to_be_bytes());
            frame.extend_from_slice(payload);
            frame
        }
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut preface = [0; 24];
            socket.read_exact(&mut preface).await.unwrap();
            // SETTINGS.
            socket.write_all(&frame(0x4, 0, 0, &[])).await.unwrap();
            loop {
                let mut head = [0; 9];
                if socket.read_exact(&mut head).await.is_err() {
                    return;
                }
                let mut payload =
                    vec![0; u32::from_be_bytes([0, head[0], head[1], head[2]]) as usize];
                if socket.read_exact(&mut payload).await.is_err() {
                    return;
                }
                let stream = u32::from_be_bytes([head[5], head[6], head[7], head[8]]) & 0x7fff_ffff;
                let reply = match (head[3], answer) {
                    // The acknowledgement of the SETTINGS of the client.
                    (0x4, _) if head[4] & 0x1 == 0 => frame(0x4, 0x1, 0, &[]),
                    (0x1, Answer::Reset(reason)) => {
                        frame(0x3, 0, stream, &u32::from(reason).to_be_bytes())
                    }
                    (0x1, Answer::GoAway(reason)) => {
                        // Of last stream ID 0.
                        let mut payload = vec![0; 4];
                        payload.extend_from_slice(&u32::from(reason).to_be_bytes());
                        frame(0x7, 0, 0, &payload)
                    }
                    _ => continue,
                };
                if socket.write_all(&reply).await.is_err() {
                    return;
                }
            }
        });
        address
    }

    /// The status that ends a call to `address`, before or after its response headers.
    async fn status_of_a_call(address: std::net::SocketAddr) -> tonic::Status {
        let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))
            .unwrap()
            .connect()
            .await
            .unwrap();
        let mut stub =
            spark_connect_proto::spark_connect_service_client::SparkConnectServiceClient::new(
                channel,
            );
        match stub
            .execute_plan(spark_connect_proto::ExecutePlanRequest::default())
            .await
        {
            Err(status) => status,
            Ok(response) => response
                .into_inner()
                .message()
                .await
                .expect_err("the response ends with an error"),
        }
    }

    #[tokio::test]
    async fn test_cannot_retry_a_stream_the_server_resets_as_grpc_python() {
        // The code of the reason, as grpc-python reports it: tonic leaves STREAM_CLOSED,
        // FRAME_SIZE_ERROR and HTTP_1_1_REQUIRED UNKNOWN, which grpc-python reports as INTERNAL.
        // A reset is the answer of the server about the call, not a failure of the connection.
        for (reason, code) in [
            (h2::Reason::NO_ERROR, tonic::Code::Internal),
            (h2::Reason::PROTOCOL_ERROR, tonic::Code::Internal),
            (h2::Reason::INTERNAL_ERROR, tonic::Code::Internal),
            (h2::Reason::STREAM_CLOSED, tonic::Code::Internal),
            (h2::Reason::FRAME_SIZE_ERROR, tonic::Code::Internal),
            (h2::Reason::CANCEL, tonic::Code::Cancelled),
            (h2::Reason::HTTP_1_1_REQUIRED, tonic::Code::Internal),
        ] {
            let status = status_of_a_call(serve_one_frame(Answer::Reset(reason)).await).await;
            assert_eq!(grpc_python_code(&status), code, "{reason:?}: {status:?}");
            assert!(
                !RetryPolicy::default().can_retry(&status),
                "{reason:?}: {status:?}"
            );
        }
        // A stream that the server refused, which it did not process, is retried.
        let refused = serve_one_frame(Answer::Reset(h2::Reason::REFUSED_STREAM)).await;
        let status = status_of_a_call(refused).await;
        assert_eq!(grpc_python_code(&status), tonic::Code::Unavailable);
        assert!(RetryPolicy::default().can_retry(&status), "{status:?}");
    }

    #[tokio::test]
    async fn test_can_retry_a_connection_the_server_ends_with_a_goaway_as_grpc_python() {
        // grpc-python reports a call that a GOAWAY ends as UNAVAILABLE, whatever its reason:
        // tonic gives it the code of the reason, such as INTERNAL for NO_ERROR.
        for reason in [
            h2::Reason::NO_ERROR,
            h2::Reason::INTERNAL_ERROR,
            h2::Reason::ENHANCE_YOUR_CALM,
        ] {
            let status = status_of_a_call(serve_one_frame(Answer::GoAway(reason)).await).await;
            assert_eq!(
                grpc_python_code(&status),
                tonic::Code::Unavailable,
                "{reason:?}: {status:?}"
            );
            assert!(
                RetryPolicy::default().can_retry(&status),
                "{reason:?}: {status:?}"
            );
            let error = crate::error::SparkError::from_grpc_status(status);
            assert_eq!(error.grpc_code(), Some(tonic::Code::Unavailable as i32));
        }
    }

    #[test]
    fn test_cannot_retry_a_cancellation_the_server_sent_or_a_timeout() {
        let p = RetryPolicy::default();
        assert!(!p.can_retry(&tonic::Status::cancelled("operation was canceled")));
        // tonic's timeout of a request is CANCELLED too.
        let timeout = tonic::Status::from_error(Box::new(tonic::TimeoutExpired(())));
        assert_eq!(timeout.code(), tonic::Code::Cancelled);
        assert_eq!(grpc_python_code(&timeout), tonic::Code::Cancelled);
        assert!(!p.can_retry(&timeout));
    }

    #[test]
    fn test_conditions_are_read_from_the_message_as_the_reference_does() {
        // The cursor that another RPC took over, anywhere in the message.
        let disconnected = tonic::Status::internal(
            "[INVALID_CURSOR.DISCONNECTED] The cursor has been disconnected by the server.",
        );
        assert!(cursor_disconnected(&disconnected));
        assert!(!cursor_disconnected(&tonic::Status::internal(
            "[INVALID_CURSOR.POSITION_NOT_FOUND] The cursor position was not found."
        )));
        assert!(!cursor_disconnected(&tonic::Status::internal("")));
        // The match is on the message alone, whatever the code: `can_retry` asks for INTERNAL.
        let not_internal = tonic::Status::invalid_argument("INVALID_CURSOR.DISCONNECTED");
        assert!(cursor_disconnected(&not_internal));
        assert!(!RetryPolicy::default().can_retry(&not_internal));
        assert!(RetryPolicy::default().can_retry(&disconnected));

        // An operation or a session that the server does not know, named as the server names it.
        assert_eq!(
            unknown_operation(&tonic::Status::not_found(
                "[INVALID_HANDLE.OPERATION_NOT_FOUND] Operation not found."
            )),
            Some("INVALID_HANDLE.OPERATION_NOT_FOUND")
        );
        assert_eq!(
            unknown_operation(&tonic::Status::internal(
                "[INVALID_HANDLE.SESSION_NOT_FOUND] Session not found."
            )),
            Some("INVALID_HANDLE.SESSION_NOT_FOUND")
        );
        // The other conditions of a handle are not: a session that changed closes the client.
        for message in [
            "[INVALID_HANDLE.SESSION_CHANGED] The existing session changed.",
            "[INVALID_HANDLE.OPERATION_ABANDONED] Operation was abandoned.",
            "INVALID_HANDLE",
            "operation not found",
            "",
        ] {
            assert_eq!(unknown_operation(&tonic::Status::internal(message)), None);
        }
    }

    /// A policy of `max_retries` retries with waits of 1 ms, so that a test does not wait.
    fn quick_policy(max_retries: u32) -> RetryPolicy {
        RetryPolicy {
            max_retries: Some(max_retries),
            initial_backoff_ms: 1,
            max_backoff_ms: Some(1),
            backoff_multiplier: 1.0,
            jitter_ms: 0,
            min_jitter_threshold_ms: 0,
            recognize_server_retry_delay: false,
            max_server_retry_delay_ms: None,
            max_retry_exception_elapsed_time_ms: None,
        }
    }

    /// Runs [`retrying`] over `outcomes`, one for each attempt in order, and gives its result
    /// with the numbers of attempts made and of checks made before a retry.
    async fn retry_over(
        policy: &RetryPolicy,
        outcomes: Vec<Result<u32, RetryError<&'static str>>>,
    ) -> (Result<u32, RetryError<&'static str>>, usize, usize) {
        let total = outcomes.len();
        let mut outcomes = outcomes.into_iter();
        let mut checks = 0;
        let result = retrying(
            Retrying::new(policy),
            || {
                checks += 1;
                Ok(())
            },
            || {
                let outcome = outcomes.next().expect("no attempt after the last outcome");
                async move { outcome }
            },
        )
        .await;
        (result, total - outcomes.len(), checks)
    }

    fn unavailable<T>() -> Result<T, RetryError<&'static str>> {
        Err(RetryError::Status(tonic::Status::unavailable("restarting")))
    }

    #[tokio::test]
    async fn test_retrying_retries_a_retryable_failure_until_an_attempt_succeeds() {
        let outcomes = vec![unavailable(), unavailable(), Ok(7)];
        let (result, attempts, checks) = retry_over(&quick_policy(5), outcomes).await;
        assert!(matches!(result, Ok(7)), "{result:?}");
        assert_eq!(attempts, 3);
        // The check runs before each retry, and not before the first attempt.
        assert_eq!(checks, 2);
    }

    #[tokio::test]
    async fn test_retrying_makes_one_attempt_of_a_call_that_succeeds() {
        let (result, attempts, checks) = retry_over(&quick_policy(5), vec![Ok(1)]).await;
        assert!(matches!(result, Ok(1)), "{result:?}");
        assert_eq!((attempts, checks), (1, 0));
    }

    #[tokio::test]
    async fn test_retrying_does_not_retry_a_failure_that_the_policy_cannot() {
        let outcomes = vec![
            Err(RetryError::Status(tonic::Status::invalid_argument("bad"))),
            Ok(1),
        ];
        let (result, attempts, checks) = retry_over(&quick_policy(5), outcomes).await;
        match result {
            Err(RetryError::Status(status)) => {
                assert_eq!(status.code(), tonic::Code::InvalidArgument);
                assert_eq!(status.message(), "bad");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!((attempts, checks), (1, 0));
    }

    #[tokio::test]
    async fn test_retrying_fails_with_the_last_failure_once_the_retries_are_spent() {
        let outcomes = vec![
            unavailable(),
            unavailable(),
            Err(RetryError::Status(tonic::Status::unavailable("the last"))),
            Ok(1),
        ];
        // Two retries: three attempts.
        let (result, attempts, checks) = retry_over(&quick_policy(2), outcomes).await;
        match result {
            Err(RetryError::Status(status)) => assert_eq!(status.message(), "the last"),
            other => panic!("{other:?}"),
        }
        assert_eq!((attempts, checks), (3, 2));

        // No retries: the failure of the only attempt.
        let (result, attempts, checks) =
            retry_over(&RetryPolicy::no_retries(), vec![unavailable(), Ok(1)]).await;
        assert!(matches!(result, Err(RetryError::Status(_))), "{result:?}");
        assert_eq!((attempts, checks), (1, 0));
    }

    #[tokio::test]
    async fn test_retrying_does_not_retry_a_call_past_its_deadline() {
        // The deadline of a call is not a failure of the policy: the reference retries it only
        // as a RetryException of a reattachable execution, which `retry_at_once` bounds.
        let outcomes = vec![
            Err(RetryError::Status(tonic::Status::deadline_exceeded(
                "Deadline Exceeded",
            ))),
            Ok(1),
        ];
        let (result, attempts, checks) = retry_over(&quick_policy(5), outcomes).await;
        match result {
            Err(RetryError::Status(status)) => {
                assert_eq!(status.code(), tonic::Code::DeadlineExceeded)
            }
            other => panic!("{other:?}"),
        }
        assert_eq!((attempts, checks), (1, 0));
    }

    #[tokio::test]
    async fn test_retrying_ends_with_the_error_of_an_attempt_that_is_no_failed_rpc() {
        let outcomes = vec![unavailable(), Err(RetryError::Aborted("unreadable")), Ok(1)];
        let (result, attempts, checks) = retry_over(&quick_policy(5), outcomes).await;
        assert!(
            matches!(result, Err(RetryError::Aborted("unreadable"))),
            "{result:?}"
        );
        assert_eq!((attempts, checks), (2, 1));
    }

    #[tokio::test]
    async fn test_retrying_ends_with_the_error_of_the_check_before_a_retry() {
        let mut attempts = 0;
        let result: Result<(), _> = retrying(
            Retrying::new(&quick_policy(5)),
            || Err("closed"),
            || {
                attempts += 1;
                async { unavailable() }
            },
        )
        .await;
        assert!(
            matches!(result, Err(RetryError::Aborted("closed"))),
            "{result:?}"
        );
        // The retry that the check refused is not sent.
        assert_eq!(attempts, 1);
    }

    #[tokio::test]
    async fn test_backoff_waits_the_delay_of_the_attempt_before_a_retry() {
        let policy = RetryPolicy {
            initial_backoff_ms: 40,
            max_backoff_ms: Some(40),
            ..quick_policy(1)
        };
        let mut retrying = Retrying::new(&policy);
        let status = tonic::Status::unavailable("restarting");
        let started = std::time::Instant::now();
        assert_eq!(retrying.backoff(&status).await, Backoff::Retry);
        assert!(started.elapsed() >= std::time::Duration::from_millis(40));
        // The retries are spent: no wait, and no retry.
        let started = std::time::Instant::now();
        assert_eq!(retrying.backoff(&status).await, Backoff::GiveUp);
        assert!(started.elapsed() < std::time::Duration::from_millis(40));
    }

    #[tokio::test]
    async fn test_backoff_does_not_spend_a_retry_on_a_failure_that_is_not_retried() {
        let mut retrying = Retrying::new(&quick_policy(1));
        let missing = tonic::Status::not_found("missing");
        assert_eq!(retrying.backoff(&missing).await, Backoff::GiveUp);
        let late = tonic::Status::deadline_exceeded("Deadline Exceeded");
        assert_eq!(retrying.backoff(&late).await, Backoff::GiveUp);
        // The one retry of the policy is still there.
        let down = tonic::Status::unavailable("down");
        assert_eq!(retrying.backoff(&down).await, Backoff::Retry);
        assert_eq!(retrying.backoff(&down).await, Backoff::GiveUp);
    }

    /// A policy whose one backoff, of a minute, a test does not wait out.
    fn slow_policy() -> RetryPolicy {
        RetryPolicy {
            initial_backoff_ms: 60_000,
            max_backoff_ms: Some(60_000),
            ..quick_policy(1)
        }
    }

    #[test]
    fn test_a_signal_ends_the_backoff_of_retries_that_end_with_an_abort() {
        use crate::runtime::{block_on, with_wait_watch, Watch};
        let down = tonic::Status::unavailable("down");
        let started = std::time::Instant::now();
        let ended = with_wait_watch(std::sync::Arc::new(|| Watch::Abort), || {
            block_on(async {
                let mut retrying = Retrying::new(&slow_policy()).ending_with_abort();
                retrying.backoff(&down).await
            })
        });
        assert_eq!(ended, Backoff::Aborted);
        assert!(started.elapsed() < std::time::Duration::from_secs(30));

        // The retry after the backoff is not made, and its check is not run.
        let mut attempts = 0;
        let result: Result<(), RetryError<&'static str>> =
            with_wait_watch(std::sync::Arc::new(|| Watch::Abort), || {
                block_on(retrying(
                    Retrying::new(&slow_policy()).ending_with_abort(),
                    || panic!("no check before a retry that is not made"),
                    || {
                        attempts += 1;
                        async { unavailable() }
                    },
                ))
            });
        assert!(matches!(result, Err(RetryError::WaitAborted)), "{result:?}");
        assert_eq!(attempts, 1);
    }

    #[test]
    fn test_a_signal_does_not_end_the_backoff_of_retries_that_are_waited_whole() {
        use crate::runtime::{block_on, with_wait_watch, Watch, POLL_INTERVAL};
        // Longer than the poll of the watch, which aborts at its first poll.
        let policy = RetryPolicy {
            initial_backoff_ms: (POLL_INTERVAL * 2).as_millis() as u64,
            max_backoff_ms: None,
            ..quick_policy(1)
        };
        let down = tonic::Status::unavailable("down");
        let ended = with_wait_watch(std::sync::Arc::new(|| Watch::Abort), || {
            block_on(async {
                let mut retrying = Retrying::new(&policy);
                retrying.backoff(&down).await
            })
        });
        assert_eq!(ended, Backoff::Retry);
    }

    #[tokio::test]
    async fn test_retries_at_once_are_bounded_by_their_elapsed_time_and_not_by_the_attempts() {
        let bounded = |ms| RetryPolicy {
            max_retry_exception_elapsed_time_ms: ms,
            ..quick_policy(0)
        };
        // No bound: always, though the policy has no retries.
        let mut retrying = Retrying::new(&bounded(None));
        assert!((0..100).all(|_| retrying.retry_at_once()));
        // A bound of 0 has passed at the first of them.
        assert!(!Retrying::new(&bounded(Some(0))).retry_at_once());
        // The clock starts at the first of them, not when the call started.
        let mut retrying = Retrying::new(&bounded(Some(50)));
        std::thread::sleep(std::time::Duration::from_millis(60));
        assert!(retrying.retry_at_once());
        std::thread::sleep(std::time::Duration::from_millis(60));
        assert!(!retrying.retry_at_once());
        // They do not spend the retries of the policy, nor the policy's theirs.
        let mut retrying = Retrying::new(&RetryPolicy {
            max_retry_exception_elapsed_time_ms: Some(0),
            ..quick_policy(1)
        });
        assert!(!retrying.retry_at_once());
        let down = tonic::Status::unavailable("down");
        assert_eq!(retrying.backoff(&down).await, Backoff::Retry);
    }
}
