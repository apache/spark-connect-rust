// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! The subchannels of the process, as gRPC's global subchannel pool keeps them: the connection to
//! one address, of what the connection is made of besides it ([`SubchannelArgs`]), which every
//! channel of a client whose own resolution gives that address shares (see
//! [`super::pick_first`]).
//!
//! A subchannel has the connectivity state of gRPC's: idle, connecting, ready, or in transient
//! failure. It connects only when a channel asks it to, to its address, of the tunnel of a proxy and
//! the TLS of its arguments, with a deadline of 20 s, or of its backoff when that is longer. In the
//! dialect of grpcio, an attempt that failed leaves it in transient failure until gRPC's backoff
//! from the start of the attempt ends (1 s, then 1.6 times longer up to 120 s, with a jitter of
//! 20 %), and then idle, as gRPC's subchannel; its backoff starts again once a connection that it
//! made ends. A Rust client's subchannel is idle again at once, as tonic's channel connects again
//! at each call that needs a connection.
//!
//! The connection is of a channel of tonic of the subchannel's own, whose connector hands over the
//! connection that the subchannel made ([`connection`]), so that the calls of every channel that
//! picked the subchannel go on the one connection. The end of the connection, as hyper drops it,
//! leaves the subchannel idle and wakes the channels that watch it. A call that tonic sends as the
//! connection ends makes its connector ask for one again, which makes one attempt at most, as a
//! call of the transport made one before (see [`connection`]).

use super::*;

/// The order of the failures of the attempts of the subchannels of the process, by which a channel
/// tells the last failure of its addresses, as gRPC's `pick_first` gives the last that it saw.
static FAILURES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What the connections of a subchannel are made of besides its address, by which gRPC's subchannel
/// pool tells them apart, of the arguments of its channels and their credentials: the origin of the
/// calls (`grpc.default_authority`, of the scheme of the TLS), the keepalive, the tunnel of a proxy
/// and its credentials, the name that TLS verifies and how, local credentials, the dialect, and the
/// call credentials of a channel of its own credentials (of a token), which no other channel shares,
/// as grpcio's channel credentials of a token are of each channel. The roots that TLS trusts are
/// those of the process, which it reads once, the system's store too (see [`super::system_roots`]),
/// as grpcio computes its default roots once: the same of every channel of the process.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SubchannelArgs {
    origin: String,
    keepalive: Option<(Duration, Duration, bool)>,
    tunnel: Option<(String, Option<String>)>,
    tls: Option<(String, bool)>,
    local: bool,
    grpcio: bool,
    credentials: Option<u64>,
}

impl SubchannelArgs {
    /// The arguments of the subchannels of a channel to `target`, of the origin `origin` of its
    /// calls and the keepalive `keepalive`, of call credentials of its own if `own_credentials`.
    pub(crate) fn new(
        target: &Target,
        origin: &str,
        keepalive: Option<(Duration, Duration, bool)>,
        own_credentials: bool,
    ) -> Self {
        /// The identities of the call credentials of the channels of their own.
        static CREDENTIALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Self {
            origin: origin.to_string(),
            keepalive,
            tunnel: target
                .tunnel
                .as_ref()
                .map(|tunnel| (tunnel.authority.clone(), tunnel.authorization.clone())),
            tls: target
                .tls
                .as_ref()
                .map(|tls| (tls.name.clone(), matches!(tls.check, NameCheck::Grpc(_)))),
            local: target.local,
            grpcio: target.grpcio,
            credentials: own_credentials.then(|| CREDENTIALS.fetch_add(1, Ordering::SeqCst)),
        }
    }
}

/// What gRPC's subchannel pool keys a subchannel by: its address, and its arguments.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct SubchannelKey {
    address: Address,
    args: SubchannelArgs,
}

/// The subchannels of a generation of the process (see [`crate::fork`]), of their keys, while a
/// channel holds them, as gRPC's global subchannel pool: a child of a `fork()` has a pool of its
/// own, whose lock no thread of its parent held.
#[derive(Default)]
pub(crate) struct Pool {
    subchannels: Mutex<std::collections::HashMap<SubchannelKey, Weak<Subchannel>>>,
}

/// The pool of the process, of each generation.
static POOL: crate::fork::PerGeneration<Arc<Pool>> = crate::fork::PerGeneration::new();

impl Pool {
    /// The pool of the generation of the process.
    pub(crate) fn of_the_process() -> Arc<Self> {
        Arc::clone(POOL.get_or_init(|_| Arc::default()))
    }

    /// The subchannel of `key` that a channel holds, else a new one of `make`, which is given the
    /// weak reference of the new subchannel.
    fn get(
        &self,
        key: SubchannelKey,
        make: impl FnOnce(&Weak<Subchannel>) -> Subchannel,
    ) -> Arc<Subchannel> {
        let mut subchannels = self.subchannels.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(subchannel) = subchannels.get(&key).and_then(Weak::upgrade) {
            return subchannel;
        }
        subchannels.retain(|_, subchannel| subchannel.strong_count() > 0);
        let subchannel = Arc::new_cyclic(make);
        subchannels.insert(key, Arc::downgrade(&subchannel));
        subchannel
    }

    /// How many subchannels of the pool a channel holds, of the tests.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        let subchannels = self.subchannels.lock().unwrap_or_else(|e| e.into_inner());
        subchannels
            .values()
            .filter(|subchannel| subchannel.strong_count() > 0)
            .count()
    }
}

/// The connectivity state of a subchannel, as gRPC's.
enum SubchannelState {
    /// No connection, nor any attempt: the next request of a channel connects.
    Idle,
    /// An attempt is in progress.
    Connecting,
    /// Connected: the connection, until the channel of tonic takes it.
    Ready(Option<Ready>),
    /// The last attempt failed: no attempt before its backoff ends, as gRPC's subchannel.
    TransientFailure,
}

/// The state of a subchannel, and what it went through, which its channels read.
struct Inner {
    state: SubchannelState,
    backoff: Backoff,
    /// How many of its attempts failed, and the last failure, of its order among the failures of
    /// the process (see [`FAILURES`]).
    failures: u64,
    last_failure: Option<(u64, Failure)>,
    /// How many connections it made: the number of the current one.
    connections: u64,
    /// The task of the attempt in progress, or of the end of the backoff, which ends with the
    /// subchannel.
    task: Option<tokio::task::AbortHandle>,
}

/// What a channel reads of a subchannel at once (see [`Subchannel::snapshot`]).
pub(super) struct Snapshot {
    /// The number of the connection of a ready subchannel.
    pub(super) ready: Option<u64>,
    pub(super) idle: bool,
    pub(super) in_failure: bool,
    pub(super) failures: u64,
    pub(super) last_failure: Option<(u64, Failure)>,
}

/// A connection of the process to one address (see the module).
pub(crate) struct Subchannel {
    address: Address,
    /// What its connections are made of: the tunnel, the TLS, the local credentials and the
    /// dialect of the target of the channel that made it, the same of every channel of its key.
    target: Target,
    /// The runtime of its attempts, that of the channel that made it.
    runtime: tokio::runtime::Handle,
    inner: Mutex<Inner>,
    /// Told of each change of its state: the channels that watch it, of their own wake-ups.
    watchers: Mutex<Vec<Weak<tokio::sync::Notify>>>,
    /// The channel of tonic of its connection (see the module).
    channel: tonic::transport::Channel,
}

impl Drop for Subchannel {
    /// Ends its attempt, and the waits of the connectors of its channel of tonic, which fail then
    /// (see [`connection`]).
    fn drop(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(|e| e.into_inner());
        if let Some(task) = inner.task.take() {
            task.abort();
        }
        let watchers = self.watchers.get_mut().unwrap_or_else(|e| e.into_inner());
        for watcher in watchers.drain(..).filter_map(|watcher| watcher.upgrade()) {
            watcher.notify_one();
        }
    }
}

impl Subchannel {
    /// The subchannel of `address` of `pool`, of the arguments `args` of a channel to `target`, whose
    /// calls are of `endpoint`, on `runtime`: the one that a channel holds, else a new one.
    pub(super) fn of(
        pool: &Pool,
        address: Address,
        args: &SubchannelArgs,
        target: &Target,
        endpoint: &tonic::transport::Endpoint,
        runtime: &tokio::runtime::Handle,
    ) -> Arc<Self> {
        let key = SubchannelKey {
            address: address.clone(),
            args: args.clone(),
        };
        pool.get(key, |weak| {
            // The worker of the channel of tonic runs on the runtime of the subchannel.
            let channel = {
                let _runtime = runtime.enter();
                endpoint.connect_with_connector_lazy(SubchannelConnector(weak.clone()))
            };
            Subchannel {
                address,
                target: target.clone(),
                runtime: runtime.clone(),
                inner: Mutex::new(Inner {
                    state: SubchannelState::Idle,
                    backoff: Backoff::new(),
                    failures: 0,
                    last_failure: None,
                    connections: 0,
                    task: None,
                }),
                watchers: Mutex::new(Vec::new()),
                channel,
            }
        })
    }

    fn inner(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The address of the subchannel.
    pub(super) fn address(&self) -> &Address {
        &self.address
    }

    /// The channel of tonic of its connection, which takes the calls of the channels that picked it.
    pub(super) fn channel(&self) -> tonic::transport::Channel {
        self.channel.clone()
    }

    /// Its state now, and what it went through.
    pub(super) fn snapshot(&self) -> Snapshot {
        let inner = self.inner();
        Snapshot {
            ready: matches!(inner.state, SubchannelState::Ready(_)).then_some(inner.connections),
            idle: matches!(inner.state, SubchannelState::Idle),
            in_failure: matches!(inner.state, SubchannelState::TransientFailure),
            failures: inner.failures,
            last_failure: inner.last_failure.clone(),
        }
    }

    /// The number of its connection, if it is ready.
    pub(super) fn ready(&self) -> Option<u64> {
        let inner = self.inner();
        matches!(inner.state, SubchannelState::Ready(_)).then_some(inner.connections)
    }

    /// Wake `watcher` at each change of its state, until it is dropped or as many
    /// [`Subchannel::unwatch`] of it came: a channel watches a subchannel once for each list of it
    /// that holds it.
    pub(super) fn watch(&self, watcher: &Arc<tokio::sync::Notify>) {
        let mut watchers = self.watchers.lock().unwrap_or_else(|e| e.into_inner());
        watchers.retain(|watching| watching.strong_count() > 0);
        watchers.push(Arc::downgrade(watcher));
    }

    /// Stop one watch of `watcher` (see [`Subchannel::watch`]).
    pub(super) fn unwatch(&self, watcher: &Arc<tokio::sync::Notify>) {
        let mut watchers = self.watchers.lock().unwrap_or_else(|e| e.into_inner());
        let watcher = Arc::downgrade(watcher);
        if let Some(index) = watchers
            .iter()
            .position(|watching| watching.ptr_eq(&watcher))
        {
            watchers.swap_remove(index);
        }
        watchers.retain(|watching| watching.strong_count() > 0);
    }

    /// How many watches the subchannel has, of the tests.
    #[cfg(test)]
    pub(super) fn watches(&self) -> usize {
        let watchers = self.watchers.lock().unwrap_or_else(|e| e.into_inner());
        watchers
            .iter()
            .filter(|watching| watching.strong_count() > 0)
            .count()
    }

    /// Wake the watchers, of a change of the state.
    fn changed(&self) {
        let watchers: Vec<_> = {
            let mut watchers = self.watchers.lock().unwrap_or_else(|e| e.into_inner());
            watchers.retain(|watching| watching.strong_count() > 0);
            watchers.iter().filter_map(Weak::upgrade).collect()
        };
        for watcher in watchers {
            watcher.notify_one();
        }
    }

    /// Connect, if idle, as gRPC's `RequestConnection`: an attempt of its own, which the channels
    /// that watch it see end.
    pub(super) fn request_connection(self: &Arc<Self>) {
        let started = {
            let mut inner = self.inner();
            if !matches!(inner.state, SubchannelState::Idle) {
                return;
            }
            self.start_attempt(&mut inner);
            true
        };
        if started {
            self.changed();
        }
    }

    /// Start an attempt of the subchannel, of `inner`, its state, which is connecting from then on:
    /// in the dialect of grpcio, of gRPC's backoff, whose next delay is the start of the next attempt
    /// after a failure and the deadline of this one, but not less than [`CONNECT_TIMEOUT`].
    fn start_attempt(self: &Arc<Self>, inner: &mut Inner) {
        let start = Instant::now();
        let timeout = self.target.connect_timeout();
        let (deadline, next) = if self.target.grpcio {
            let delay = inner.backoff.next();
            (start + delay.max(timeout), Some(start + delay))
        } else {
            (start + timeout, None)
        };
        let weak = Arc::downgrade(self);
        let target = self.target.clone();
        let address = self.address.clone();
        let attempt = self.runtime.spawn(async move {
            let result = connect_address(&target, &address, deadline).await;
            if let Err(failure) = &result {
                log_failure(target.grpcio, &address, failure);
            }
            if let Some(subchannel) = weak.upgrade() {
                subchannel.attempted(result, next);
            }
        });
        inner.task = Some(attempt.abort_handle());
        inner.state = SubchannelState::Connecting;
    }

    /// The end of an attempt, of `result`, whose failure leaves the subchannel in transient failure
    /// until `next`, if any, as gRPC's, else idle.
    fn attempted(self: &Arc<Self>, result: Result<Ready, Failure>, next: Option<Instant>) {
        {
            let mut inner = self.inner();
            if !matches!(inner.state, SubchannelState::Connecting) {
                return;
            }
            inner.task = None;
            match result {
                Ok(ready) => {
                    inner.connections += 1;
                    inner.state = SubchannelState::Ready(Some(ready));
                }
                Err(failure) => {
                    inner.failures += 1;
                    inner.last_failure = Some((FAILURES.fetch_add(1, Ordering::SeqCst), failure));
                    match next.filter(|next| *next > Instant::now()) {
                        Some(next) => {
                            inner.state = SubchannelState::TransientFailure;
                            let weak = Arc::downgrade(self);
                            let backoff = self.runtime.spawn(async move {
                                tokio::time::sleep_until(next).await;
                                if let Some(subchannel) = weak.upgrade() {
                                    subchannel.backed_off();
                                }
                            });
                            inner.task = Some(backoff.abort_handle());
                        }
                        // Of a Rust client, or of an attempt that took longer than its backoff, as
                        // gRPC's subchannel then goes idle at once.
                        None => inner.state = SubchannelState::Idle,
                    }
                }
            }
        }
        self.changed();
    }

    /// The end of the backoff of a failure: idle, as gRPC's subchannel.
    fn backed_off(&self) {
        {
            let mut inner = self.inner();
            if !matches!(inner.state, SubchannelState::TransientFailure) {
                return;
            }
            inner.task = None;
            inner.state = SubchannelState::Idle;
        }
        self.changed();
    }

    /// The end of the connection of the number `connection`: idle, of a backoff that starts again,
    /// as gRPC's subchannel whose connection ends.
    fn connection_ended(&self, connection: u64) {
        {
            let mut inner = self.inner();
            if !matches!(inner.state, SubchannelState::Ready(_)) || inner.connections != connection
            {
                return;
            }
            inner.state = SubchannelState::Idle;
            inner.backoff = Backoff::new();
        }
        self.changed();
    }

    /// A step of [`connection`], of `awaited`, the attempt that it waits for, if any: the
    /// connection that the channel of tonic takes, of its number, or the failure of the call, once
    /// there is one.
    fn connection_step(
        self: &Arc<Self>,
        awaited: &mut Option<Awaited>,
    ) -> Option<Result<(Ready, u64), ConnectError>> {
        {
            let mut inner = self.inner();
            let counts = Awaited {
                failures: inner.failures,
                connections: inner.connections,
            };
            let (failed, connected) = match *awaited {
                Some(at) => (
                    counts.failures > at.failures,
                    counts.connections > at.connections,
                ),
                None => (false, false),
            };
            match &mut inner.state {
                SubchannelState::Ready(held) => {
                    if let Some(mut ready) = held.take() {
                        // A connection that the subchannel made since the connector asked is
                        // hyper's as it is, of an end that came since too, which then fails the
                        // call of its end; one made before, while it is still open.
                        if connected || ready.alive() {
                            return Some(Ok((ready, counts.connections)));
                        }
                    }
                    if awaited.is_some() {
                        return Some(Err(self.closed()));
                    }
                    // The connection that tonic had ended, or the one held ended before a call
                    // took it: one attempt, of a backoff that starts again.
                    inner.state = SubchannelState::Idle;
                    inner.backoff = Backoff::new();
                    self.start_attempt(&mut inner);
                    *awaited = Some(counts);
                }
                // The attempt failed, and the subchannel is idle again: of a Rust client, or of an
                // attempt that took longer than its backoff.
                SubchannelState::Idle if failed => return Some(Err(self.failed(&inner))),
                SubchannelState::Idle if awaited.is_some() => return Some(Err(self.closed())),
                SubchannelState::Idle => {
                    self.start_attempt(&mut inner);
                    *awaited = Some(counts);
                }
                SubchannelState::Connecting => {
                    awaited.get_or_insert(counts);
                    return None;
                }
                SubchannelState::TransientFailure => return Some(Err(self.failed(&inner))),
            }
        }
        self.changed();
        None
    }

    /// The error of a call of the last failure of the subchannel, of `inner`, its state.
    fn failed(&self, inner: &Inner) -> ConnectError {
        match &inner.last_failure {
            Some((_, failure)) => ConnectError::of(
                &ChannelFailure::Addresses {
                    address: self.address.clone(),
                    failure: failure.clone(),
                },
                self.target.grpcio,
            ),
            None => self.closed(),
        }
    }

    /// The error of a call of the end of the connection of the subchannel (see
    /// [`connection_closed`]).
    fn closed(&self) -> ConnectError {
        connection_closed(self.target.grpcio)
    }
}

/// The counts of the failures and of the connections of a subchannel as the attempt that a
/// connector of its channel of tonic waits for started, or was found in progress (see
/// [`connection`]): a failure or a connection that came after them is of that attempt.
#[derive(Clone, Copy)]
struct Awaited {
    failures: u64,
    connections: u64,
}

/// The connection of the subchannel of `weak`, of its number, for its channel of tonic, which asks
/// for one at its first call and once the one that it had ended (as at a GOAWAY, or of a call that
/// tonic sent as it ended): the connection that the subchannel holds, if it is still open; else the
/// connection of one attempt, which the connector makes, or waits for while one is in progress, as
/// hyper's even if it ended since, so that the call fails of its end, or the failure of that
/// attempt; in transient failure, the last failure at once. So a call makes one attempt at most,
/// as a call of the transport made one before SPARK-62257, and fails of the failure of the
/// attempt, in transient failure too, as gRPC's calls fail once the channel's attempts failed. The
/// connector holds the subchannel only while it reads its state, so that a subchannel that no
/// channel holds any more is freed as its connector waits, which then fails.
async fn connection(weak: Weak<Subchannel>) -> Result<(Ready, u64), ConnectError> {
    let Some(subchannel) = weak.upgrade() else {
        return Err(connection_closed(grpcio_dialect()));
    };
    let grpcio = subchannel.target.grpcio;
    let wake = Arc::new(tokio::sync::Notify::new());
    subchannel.watch(&wake);
    let _watching = Watching {
        subchannel: weak.clone(),
        wake: Arc::clone(&wake),
    };
    drop(subchannel);
    let mut awaited = None;
    loop {
        let woken = wake.notified();
        let Some(subchannel) = weak.upgrade() else {
            return Err(connection_closed(grpcio));
        };
        if let Some(result) = subchannel.connection_step(&mut awaited) {
            return result;
        }
        drop(subchannel);
        woken.await;
    }
}

/// A watch of a subchannel by a connector of its channel of tonic, which ends with it (see
/// [`connection`]).
struct Watching {
    subchannel: Weak<Subchannel>,
    wake: Arc<tokio::sync::Notify>,
}

impl Drop for Watching {
    fn drop(&mut self) {
        if let Some(subchannel) = self.subchannel.upgrade() {
            subchannel.unwatch(&self.wake);
        }
    }
}

/// The error of a call of a connection whose subchannel no channel holds any more, or that ended
/// before the call took it: of the end of a connection, in the dialect of grpcio if `grpcio`.
fn connection_closed(grpcio: bool) -> ConnectError {
    ConnectError {
        text: if grpcio {
            SUBCHANNEL_GONE_GRPCIO
        } else {
            SUBCHANNEL_GONE_RUST
        }
        .into(),
    }
}

#[cfg(test)]
impl Subchannel {
    /// The connection that the subchannel holds, of the tests, as its channel of tonic would take
    /// it, but of no end that it is told of.
    pub(super) fn take_held(&self) -> Option<Ready> {
        match &mut self.inner().state {
            SubchannelState::Ready(held) => held.take(),
            _ => None,
        }
    }

    /// The connection that its channel of tonic takes, of the tests (see [`connection`]): the
    /// handover to hyper, or the text of the failure.
    pub(super) async fn hand_over(self: &Arc<Self>) -> Result<Handover, String> {
        use tonic::codegen::Service;
        let uri = tonic::codegen::http::Uri::from_static("http://localhost");
        SubchannelConnector(Arc::downgrade(self))
            .call(uri)
            .await
            .map(TokioIo::into_inner)
            .map_err(|error| error.to_string())
    }

    /// How many failures and connections it had, of the tests.
    pub(super) fn counts(&self) -> (u64, u64) {
        let inner = self.inner();
        (inner.failures, inner.connections)
    }
}

/// The end of a connection of a subchannel, as hyper drops it (see [`Handover`]): the subchannel is
/// idle from then on.
pub(super) struct ConnectionEnd {
    subchannel: Weak<Subchannel>,
    connection: u64,
}

impl ConnectionEnd {
    /// The server's GOAWAY: the connection takes no new call, and the subchannel is idle, as
    /// gRPC's subchannel whose transport received a GOAWAY, while hyper keeps the connection for
    /// the calls in flight.
    pub(super) fn going_away(&self) {
        if let Some(subchannel) = self.subchannel.upgrade() {
            subchannel.connection_ended(self.connection);
        }
    }
}

impl Drop for ConnectionEnd {
    fn drop(&mut self) {
        if let Some(subchannel) = self.subchannel.upgrade() {
            subchannel.connection_ended(self.connection);
        }
    }
}

/// The text of a connection whose subchannel no channel holds any more, which a call that its
/// channel of tonic took meanwhile fails of: of the end of the connection that it had.
const SUBCHANNEL_GONE_GRPCIO: &str = SOCKET_CLOSED;
const SUBCHANNEL_GONE_RUST: &str = "the connection was closed";

/// The connector of the channel of tonic of a subchannel (see [`connection`]).
#[derive(Clone)]
struct SubchannelConnector(Weak<Subchannel>);

impl tonic::codegen::Service<tonic::codegen::http::Uri> for SubchannelConnector {
    type Response = TokioIo<Handover>;
    type Error = ConnectError;
    type Future = tonic::codegen::BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _: tonic::codegen::http::Uri) -> Self::Future {
        let weak = self.0.clone();
        Box::pin(async move {
            let (ready, connection) = connection(weak.clone()).await?;
            Ok(TokioIo::new(Handover::new(ready).ending(ConnectionEnd {
                subchannel: weak,
                connection,
            })))
        })
    }
}
