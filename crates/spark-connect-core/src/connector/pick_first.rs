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

//! The channel of a client, as gRPC's client channel: a resolver of its own, which resolves the
//! name of its target, and gRPC's `pick_first` over the subchannels of the addresses that it gives
//! (see [`super::subchannel`]), which the channels of the process share.
//!
//! In the dialect of grpcio, as grpcio's channel of PySpark:
//!
//! - The first call resolves the name of the target, which the calls wait for; a failure fails
//!   them at once of its error, and is tried again in the background a backoff after its end, as
//!   gRPC's resolver tries it (1 s, then 1.6 times longer up to 120 s, with a jitter of 20 %).
//! - The channel picks the first subchannel of its addresses that is ready, of any channel of the
//!   process; else it connects to its addresses in turn, as gRPC's `pick_first` with its happy
//!   eyeballs: the next one 250 ms after the previous one while that one is not done, or at once
//!   when it fails, skipping those in transient failure. The calls wait meanwhile, and the first
//!   subchannel that is ready is the channel's, whose connection its calls take; the others are let
//!   go.
//! - Once every subchannel failed, the channel is in transient failure: its calls fail at once of
//!   the last failure, while each subchannel connects again once its backoff ends, until one is
//!   ready, and the channel asks its resolver to resolve the name again each time all of them
//!   failed again.
//! - When the connection of the channel's subchannel ends, the channel asks its resolver to resolve
//!   the name again, and is idle: the next call connects, to the addresses of the last resolution.
//! - The resolver resolves the name again when the channel asks it to, no sooner than 30 s after
//!   the start of its last resolution, as gRPC's (`grpc.dns_min_time_between_resolutions_ms`), in
//!   the background; the channel takes the addresses that it gives as it connects next, or at once
//!   while it connects or failed; while it is ready, it connects to those of a resolution that no
//!   longer gives its address in the background, and moves to them once one is ready, or fails once
//!   all failed. A resolution that fails after one that succeeded leaves the channel with the
//!   addresses of that one, and is tried again a backoff after its end.
//! - After 30 minutes without a call, a channel in transient failure stops connecting and
//!   resolving, as gRPC's channel goes idle, and the next call resolves the name again.
//!
//! A Rust client's channel resolves the name as each call that needs a connection connects, as
//! tonic's channel does, and that call fails of the failure of its connection, as do the calls
//! that waited for it, rather than failing at once while the channel connects in the background.

use super::subchannel::{Pool, Snapshot, Subchannel, SubchannelArgs};
use super::*;

/// What the calls of a channel see of its state (see [`ClientChannel::pick`]).
#[derive(Clone)]
enum Published {
    /// Idle: the next call connects, of the pass `pass`, whose calls wait. `failed`, of a Rust
    /// client, is the failure of the pass before it, of the calls that waited for it.
    Idle {
        pass: u64,
        failed: Option<ChannelFailure>,
    },
    /// The pass `pass` connects.
    Connecting { pass: u64 },
    /// The calls take the connection of the subchannel.
    Ready(Arc<Subchannel>),
    /// The calls fail at once of the failure.
    TransientFailure(ChannelFailure),
    /// The channel is closed.
    Shutdown,
}

impl Published {
    /// Whether the calls see the same of `self` and of `other`.
    fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Idle { pass, failed },
                Self::Idle {
                    pass: other_pass,
                    failed: other_failed,
                },
            ) => pass == other_pass && failed.is_some() == other_failed.is_some(),
            (Self::Connecting { pass }, Self::Connecting { pass: other }) => pass == other,
            (Self::Ready(subchannel), Self::Ready(other)) => Arc::ptr_eq(subchannel, other),
            (Self::TransientFailure(failure), Self::TransientFailure(other)) => {
                failure.text(true) == other.text(true)
            }
            (Self::Shutdown, Self::Shutdown) => true,
            _ => false,
        }
    }
}

/// Why a call of a channel has no connection to take (see [`ClientChannel::pick`]).
pub(crate) struct Unpicked(Why);

/// Why a call has no connection (see [`Unpicked`]).
enum Why {
    /// The channel could not connect, of this failure.
    Failed(ChannelFailure),
    /// The channel was closed.
    Closed,
}

impl Unpicked {
    /// The status of the call that this fails, in the dialect of grpcio if `grpcio`: of a failure,
    /// gRPC's `UNAVAILABLE` of its text, as tonic gives a connector's error; of a closed channel,
    /// grpcio's `CANCELLED` (see `crate::client::ChannelClosed`).
    pub(crate) fn status(&self, grpcio: bool) -> tonic::Status {
        match &self.0 {
            Why::Failed(failure) => tonic::Status::unavailable(failure.text(grpcio)),
            Why::Closed => tonic::Status::cancelled(crate::client::CLOSED_CHANNEL),
        }
    }

    /// The failure of the channel, if it failed, of the tests.
    #[cfg(test)]
    pub(super) fn failure(self) -> Option<ChannelFailure> {
        match self.0 {
            Why::Failed(failure) => Some(failure),
            Why::Closed => None,
        }
    }
}

/// The connectivity state of a channel, as gRPC's.
#[derive(Clone)]
enum Phase {
    Idle,
    Connecting,
    Ready,
    TransientFailure(ChannelFailure),
}

/// The resolver of a channel, as gRPC's polling resolver of a name.
struct Resolver {
    /// Of a resolver made again once the channel idled, whose resolutions before are not taken.
    generation: u64,
    /// The addresses of the last resolution that succeeded.
    addresses: Option<Vec<Address>>,
    /// When the last resolution started, which the next one waits for 30 s from, in the dialect of
    /// grpcio, unless it tries one that failed again.
    last_start: Option<Instant>,
    /// The resolution in progress, if any.
    resolving: Option<tokio::task::AbortHandle>,
    /// Whether the channel asked for a resolution again, as gRPC's re-resolution request.
    requested: bool,
    /// When a resolution that failed is tried again, and the backoff of those tries.
    retry: Option<Instant>,
    backoff: Backoff,
    /// The result of the resolution that ended, which the channel takes next.
    done: Option<Result<Vec<Address>, ChannelFailure>>,
}

impl Resolver {
    fn new(generation: u64) -> Self {
        Self {
            generation,
            addresses: None,
            last_start: None,
            resolving: None,
            requested: false,
            retry: None,
            backoff: Backoff::new(),
            done: None,
        }
    }
}

impl Drop for Resolver {
    fn drop(&mut self) {
        if let Some(resolving) = self.resolving.take() {
            resolving.abort();
        }
    }
}

/// A subchannel of a list of a channel, and what the channel saw of it.
struct Entry {
    subchannel: Arc<Subchannel>,
    /// The wake-up of the channel, which the subchannel wakes at each change of its state.
    watcher: Arc<tokio::sync::Notify>,
    /// How many failures of the subchannel the channel saw.
    seen_failures: u64,
    /// Whether the subchannel failed since it is of the list, or was in transient failure then.
    failed: bool,
}

impl Drop for Entry {
    fn drop(&mut self) {
        self.subchannel.unwatch(&self.watcher);
    }
}

/// What a step of a list of subchannels came to (see [`List::step`]).
enum Outcome {
    /// The subchannel of this index is ready, of the connection of this number.
    Ready(usize, u64),
    /// Every subchannel failed, the last of this failure, as gRPC's `pick_first` reports transient
    /// failure: once at the end of the first pass, and then once each subchannel failed again.
    Failed(ChannelFailure),
    /// Nothing to report.
    Pending,
}

/// The subchannels of the addresses of a resolution, which `pick_first` connects to in turn, as
/// gRPC's subchannel list.
struct List {
    entries: Vec<Entry>,
    /// The index of the subchannel of the attempt of the first pass, of happy eyeballs.
    attempting: usize,
    /// Whether the attempt of that subchannel started.
    started: bool,
    /// When the attempt of the next subchannel starts, while that one is not done.
    timer: Option<Instant>,
    /// Whether every subchannel failed once: the channel is in transient failure, and each
    /// subchannel connects again as it is idle.
    complete: bool,
    /// How many failures of the subchannels came since then.
    failures: u64,
    /// The last failure of a subchannel of the list, of its order among those of the process.
    last_failure: Option<(u64, ChannelFailure)>,
}

impl List {
    /// The list of `subchannels`, which wake `watcher`.
    fn new(subchannels: Vec<Arc<Subchannel>>, watcher: &Arc<tokio::sync::Notify>) -> Self {
        let mut list = Self {
            entries: Vec::with_capacity(subchannels.len()),
            attempting: 0,
            started: false,
            timer: None,
            complete: false,
            failures: 0,
            last_failure: None,
        };
        for subchannel in subchannels {
            subchannel.watch(watcher);
            let snapshot = subchannel.snapshot();
            // As gRPC's `pick_first`, which skips a subchannel in transient failure as it starts.
            if snapshot.in_failure {
                list.note_failure(&subchannel, &snapshot);
            }
            list.entries.push(Entry {
                watcher: Arc::clone(watcher),
                seen_failures: snapshot.failures,
                failed: snapshot.in_failure,
                subchannel,
            });
        }
        list
    }

    /// Keep the last failure of `subchannel`, of `snapshot`, if it is the last of the list.
    fn note_failure(&mut self, subchannel: &Subchannel, snapshot: &Snapshot) {
        let Some((order, failure)) = &snapshot.last_failure else {
            return;
        };
        if self
            .last_failure
            .as_ref()
            .is_some_and(|(last, _)| last > order)
        {
            return;
        }
        self.last_failure = Some((
            *order,
            ChannelFailure::Addresses {
                address: subchannel.address().clone(),
                failure: failure.clone(),
            },
        ));
    }

    /// The last failure of the list.
    fn last_failure(&self) -> ChannelFailure {
        match &self.last_failure {
            Some((_, failure)) => failure.clone(),
            None => ChannelFailure::NoAddress,
        }
    }

    /// A step of `pick_first` over the list, at `now`, of the states of its subchannels now, of
    /// happy eyeballs of `delay`, [`ATTEMPT_DELAY`] (see the module).
    fn step(&mut self, now: Instant, delay: Duration) -> Outcome {
        let snapshots: Vec<Snapshot> = self
            .entries
            .iter()
            .map(|entry| entry.subchannel.snapshot())
            .collect();
        // The first that is ready, of any channel: as gRPC's `pick_first`, which takes a
        // subchannel of the list that is ready as it sees it so.
        if let Some((index, connection)) = snapshots
            .iter()
            .enumerate()
            .find_map(|(index, snapshot)| snapshot.ready.map(|connection| (index, connection)))
        {
            return Outcome::Ready(index, connection);
        }
        let mut failures = 0;
        for index in 0..self.entries.len() {
            let snapshot = &snapshots[index];
            let entry = &mut self.entries[index];
            if snapshot.failures > entry.seen_failures {
                failures += snapshot.failures - entry.seen_failures;
                entry.seen_failures = snapshot.failures;
                entry.failed = true;
                let subchannel = Arc::clone(&entry.subchannel);
                self.note_failure(&subchannel, snapshot);
            }
        }
        if self.complete {
            // As gRPC's `pick_first` once every subchannel failed: each one connects again as it
            // is idle, and each time as many failures as subchannels came, the last is reported.
            for (entry, snapshot) in self.entries.iter().zip(&snapshots) {
                if snapshot.idle {
                    entry.subchannel.request_connection();
                }
            }
            let size = self.entries.len() as u64;
            let before = self.failures;
            self.failures += failures;
            if failures > 0 && self.failures / size > before / size {
                return Outcome::Failed(self.last_failure());
            }
            return Outcome::Pending;
        }
        // The first pass, of happy eyeballs. A subchannel whose attempt started, which is idle of
        // no failure, as of a connection that ended before the channel saw it ready, connects
        // again.
        let started = (self.attempting + usize::from(self.started)).min(self.entries.len());
        for (entry, snapshot) in self.entries[..started].iter().zip(&snapshots) {
            if snapshot.idle && !entry.failed {
                entry.subchannel.request_connection();
            }
        }
        while self.attempting < self.entries.len() {
            let index = self.attempting;
            if self.entries[index].failed {
                self.attempting += 1;
                self.started = false;
                self.timer = None;
                continue;
            }
            if !self.started {
                self.started = true;
                if snapshots[index].idle {
                    self.entries[index].subchannel.request_connection();
                }
                if index + 1 < self.entries.len() {
                    self.timer = Some(now + delay);
                }
            }
            match self.timer {
                Some(at) if now >= at => {
                    self.attempting += 1;
                    self.started = false;
                    self.timer = None;
                }
                _ => break,
            }
        }
        if self.attempting >= self.entries.len() && self.entries.iter().all(|entry| entry.failed) {
            self.complete = true;
            self.failures = 0;
            for (entry, snapshot) in self.entries.iter().zip(&snapshots) {
                if snapshot.idle {
                    entry.subchannel.request_connection();
                }
            }
            return Outcome::Failed(self.last_failure());
        }
        Outcome::Pending
    }

    /// This list, of its subchannel of `index` only, which the channel picked.
    fn of_the_picked(mut self, index: usize) -> Self {
        let picked = self.entries.swap_remove(index);
        self.entries.clear();
        self.entries.push(picked);
        self.timer = None;
        self
    }
}

/// The state of a channel.
struct State {
    /// Whether the channel is open: its client is.
    open: bool,
    /// Whether a call asked the channel to connect.
    exit_idle: bool,
    phase: Phase,
    /// The number of the pass of connections: of the next one while idle, else of the current one.
    pass: u64,
    /// Of a Rust client, the failure of the last pass, while idle.
    failed: Option<ChannelFailure>,
    resolver: Resolver,
    /// The subchannels of the current pass, or the picked one.
    list: Option<List>,
    /// The subchannels of a resolution of a channel that is ready, which replace the picked one
    /// once one is ready.
    pending: Option<List>,
    /// The picked subchannel, and the number of its connection that the channel took.
    picked: Option<(Arc<Subchannel>, u64)>,
    /// The subchannel that was picked, of a channel idle since its connection ended, until the next
    /// pass takes the subchannels of its addresses: so that the pass finds the connection that the
    /// subchannel made meanwhile, as for a call that took it then, rather than a new subchannel of
    /// the address.
    last_picked: Option<Arc<Subchannel>>,
}

impl State {
    /// Idle, of `failed`, the failure of the pass of a Rust client: the next call connects.
    fn go_idle(&mut self, failed: Option<ChannelFailure>) {
        self.list = None;
        self.pending = None;
        self.picked = None;
        self.last_picked = None;
        self.phase = Phase::Idle;
        self.pass += 1;
        self.failed = failed;
    }
}

/// The channel of a client (see the module).
pub(crate) struct ClientChannel {
    /// What it resolves, and what its connections are made of.
    target: Target,
    args: SubchannelArgs,
    /// The endpoint of the channels of tonic of its subchannels.
    endpoint: tonic::transport::Endpoint,
    pool: Arc<Pool>,
    /// The runtime of its work in the background, and of the attempts of its subchannels.
    runtime: tokio::runtime::Handle,
    state: Mutex<State>,
    published: tokio::sync::watch::Sender<Published>,
    /// Wakes the work of the channel: of a call, or of a change of a subchannel that it watches.
    wake: Arc<tokio::sync::Notify>,
    /// When a call last came, for [`IDLE_TIMEOUT`].
    last_call: Mutex<Instant>,
    /// Whether the work of the channel in the background started.
    working: std::sync::atomic::AtomicBool,
    /// The channel of tonic of a target that gRPC refuses (see [`refused_connection`]).
    refused: std::sync::OnceLock<tonic::transport::Channel>,
    /// How long the attempt to an address runs alone before the next one starts,
    /// [`ATTEMPT_DELAY`], but of the tests.
    attempt_delay: Duration,
}

impl ClientChannel {
    /// The channel of a client to `target`, of its subchannels of `args` in `pool`, whose calls are
    /// of `endpoint`, on `runtime`.
    pub(crate) fn new(
        target: Target,
        args: SubchannelArgs,
        endpoint: tonic::transport::Endpoint,
        pool: Arc<Pool>,
        runtime: tokio::runtime::Handle,
    ) -> Arc<Self> {
        Self::of_delay(target, args, endpoint, pool, runtime, ATTEMPT_DELAY)
    }

    /// As [`ClientChannel::new`], of the attempt to an address that runs alone `attempt_delay`
    /// before the next one starts, of the tests.
    #[cfg(test)]
    pub(super) fn with_attempt_delay(
        target: Target,
        args: SubchannelArgs,
        endpoint: tonic::transport::Endpoint,
        pool: Arc<Pool>,
        runtime: tokio::runtime::Handle,
        attempt_delay: Duration,
    ) -> Arc<Self> {
        Self::of_delay(target, args, endpoint, pool, runtime, attempt_delay)
    }

    fn of_delay(
        target: Target,
        args: SubchannelArgs,
        endpoint: tonic::transport::Endpoint,
        pool: Arc<Pool>,
        runtime: tokio::runtime::Handle,
        attempt_delay: Duration,
    ) -> Arc<Self> {
        let (published, _) = tokio::sync::watch::channel(Published::Idle {
            pass: 0,
            failed: None,
        });
        Arc::new(Self {
            target,
            args,
            endpoint,
            pool,
            runtime,
            state: Mutex::new(State {
                open: true,
                exit_idle: false,
                phase: Phase::Idle,
                pass: 0,
                failed: None,
                resolver: Resolver::new(0),
                list: None,
                pending: None,
                picked: None,
                last_picked: None,
            }),
            published,
            wake: Arc::new(tokio::sync::Notify::new()),
            last_call: Mutex::new(Instant::now()),
            working: Default::default(),
            refused: Default::default(),
            attempt_delay,
        })
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether the channel speaks as grpcio (see [`set_grpcio_dialect`]).
    pub(crate) fn grpcio(&self) -> bool {
        self.target.grpcio
    }

    /// The channel of tonic whose connection a call takes: of the picked subchannel, once the
    /// channel connected; or why the call has none (see the module).
    pub(crate) async fn pick(self: &Arc<Self>) -> Result<tonic::transport::Channel, Unpicked> {
        if let Some(ChannelTarget::Refused { code, message }) = &self.target.form {
            return Ok(self.refused_channel(*code, message));
        }
        *self.last_call.lock().unwrap_or_else(|e| e.into_inner()) = Instant::now();
        self.work();
        let mut published = self.published.subscribe();
        // The pass that this call waits for.
        let mut waits_for: Option<u64> = None;
        loop {
            let now = published.borrow_and_update().clone();
            match now {
                Published::Ready(subchannel) => return Ok(subchannel.channel()),
                Published::TransientFailure(failure) => return Err(Unpicked(Why::Failed(failure))),
                Published::Connecting { pass } => waits_for = Some(pass),
                Published::Idle { pass, failed } => {
                    if let (Some(waited), Some(failure)) = (waits_for, failed) {
                        if waited + 1 == pass {
                            return Err(Unpicked(Why::Failed(failure)));
                        }
                    }
                    waits_for = Some(pass);
                    self.state().exit_idle = true;
                    self.wake.notify_one();
                }
                Published::Shutdown => return Err(Unpicked(Why::Closed)),
            }
            if published.changed().await.is_err() {
                return Err(Unpicked(Why::Closed));
            }
        }
    }

    /// The channel of tonic of a target that gRPC refuses, of the status of `code` and `message`.
    fn refused_channel(&self, code: i32, message: &str) -> tonic::transport::Channel {
        self.refused
            .get_or_init(|| {
                let _runtime = self.runtime.enter();
                self.endpoint.connect_with_connector_lazy(RefusedConnector {
                    code,
                    message: message.to_string(),
                    grpcio: self.grpcio(),
                })
            })
            .clone()
    }

    /// Close the channel, as grpcio's channel that its client closes: it lets go of its
    /// subchannels, whose connections end once no other channel holds them, and does nothing more.
    pub(crate) fn shutdown(&self) {
        let mut state = self.state();
        state.open = false;
        state.list = None;
        state.pending = None;
        state.picked = None;
        state.last_picked = None;
        state.resolver = Resolver::new(state.resolver.generation + 1);
        self.publish(&state);
        drop(state);
        self.wake.notify_one();
    }

    /// Start the work of the channel in the background, once.
    fn work(self: &Arc<Self>) {
        if self.working.swap(true, Ordering::SeqCst) {
            return;
        }
        let weak = Arc::downgrade(self);
        let wake = Arc::clone(&self.wake);
        self.runtime.spawn(async move {
            loop {
                let next = {
                    let Some(channel) = weak.upgrade() else {
                        return;
                    };
                    match channel.step() {
                        Some(next) => next,
                        None => return,
                    }
                };
                match next {
                    Some(at) => {
                        tokio::select! {
                            _ = wake.notified() => {}
                            _ = tokio::time::sleep_until(at) => {}
                        }
                    }
                    None => wake.notified().await,
                }
            }
        });
    }

    /// A step of the work of the channel (see the module): the next time that it is due, if any,
    /// or None once the channel is closed.
    fn step(self: &Arc<Self>) -> Option<Option<Instant>> {
        let now = Instant::now();
        let mut state = self.state();
        if !state.open {
            return None;
        }
        let grpcio = self.grpcio();
        let last_call = *self.last_call.lock().unwrap_or_else(|e| e.into_inner());
        // As gRPC's channel, which idles after 30 minutes without a call, and makes its resolver
        // again then.
        if grpcio
            && matches!(state.phase, Phase::TransientFailure(_))
            && now.saturating_duration_since(last_call) >= IDLE_TIMEOUT
        {
            state.go_idle(None);
            state.resolver = Resolver::new(state.resolver.generation + 1);
        }
        if let Some(result) = state.resolver.done.take() {
            self.resolved(&mut state, result, now);
        }
        if let Some((subchannel, connection)) = &state.picked {
            if subchannel.ready() != Some(*connection) {
                self.lost(&mut state);
            }
        }
        if std::mem::take(&mut state.exit_idle) && matches!(state.phase, Phase::Idle) {
            self.exit_idle(&mut state, now);
        }
        self.pick_first(&mut state, now);
        let mut next = self.resolve_when_due(&mut state, now);
        for list in [&state.list, &state.pending].into_iter().flatten() {
            next = earliest(next, list.timer);
        }
        if grpcio && matches!(state.phase, Phase::TransientFailure(_)) {
            next = earliest(next, Some(last_call + IDLE_TIMEOUT));
        }
        self.publish(&state);
        Some(next)
    }

    /// Make the calls see the state of the channel.
    fn publish(&self, state: &State) {
        let now = if !state.open {
            Published::Shutdown
        } else {
            match &state.phase {
                Phase::Idle => Published::Idle {
                    pass: state.pass,
                    failed: state.failed.clone(),
                },
                Phase::Connecting => Published::Connecting { pass: state.pass },
                Phase::Ready => match &state.picked {
                    Some((subchannel, _)) => Published::Ready(Arc::clone(subchannel)),
                    None => Published::Connecting { pass: state.pass },
                },
                Phase::TransientFailure(failure) => Published::TransientFailure(failure.clone()),
            }
        };
        let mut now = Some(now);
        self.published.send_if_modified(|published| {
            let now = now.take().expect("published once");
            if published.same(&now) {
                return false;
            }
            *published = now;
            true
        });
    }

    /// Connect, of a call of an idle channel: to the addresses of the last resolution, or of a
    /// resolution first, of the first call of a channel of grpcio and of each pass of a Rust
    /// client.
    fn exit_idle(self: &Arc<Self>, state: &mut State, now: Instant) {
        state.phase = Phase::Connecting;
        state.failed = None;
        match &state.resolver.addresses {
            Some(addresses) if self.grpcio() => {
                let addresses = addresses.clone();
                self.connect_to(state, addresses);
            }
            _ => self.start_resolution(state, now),
        }
    }

    /// Resolve the name of the target in the background, from `now`.
    fn start_resolution(self: &Arc<Self>, state: &mut State, now: Instant) {
        if state.resolver.resolving.is_some() {
            return;
        }
        state.resolver.requested = false;
        state.resolver.retry = None;
        state.resolver.last_start = Some(now);
        let generation = state.resolver.generation;
        let weak = Arc::downgrade(self);
        let target = self.target.clone();
        let resolving = self.runtime.spawn(async move {
            let result = resolution(&target).await;
            if let Some(channel) = weak.upgrade() {
                {
                    let mut state = channel.state();
                    if state.resolver.generation != generation {
                        return;
                    }
                    state.resolver.resolving = None;
                    state.resolver.done = Some(result);
                }
                channel.wake.notify_one();
            }
        });
        state.resolver.resolving = Some(resolving.abort_handle());
    }

    /// Ask the resolver to resolve the name again, as gRPC's `pick_first` asks for a
    /// re-resolution: of the dialect of grpcio only, unless a resolution is in progress or one that
    /// failed is to be tried again, which comes anyway.
    fn request_resolution(&self, state: &mut State) {
        if self.grpcio() && state.resolver.resolving.is_none() && state.resolver.retry.is_none() {
            state.resolver.requested = true;
        }
    }

    /// Start the resolution that is due, of a failure tried again, or of a request no sooner than
    /// 30 s after the start of the last resolution: the time of the next one, if it is not due yet.
    fn resolve_when_due(self: &Arc<Self>, state: &mut State, now: Instant) -> Option<Instant> {
        if !self.grpcio() || state.resolver.resolving.is_some() {
            return None;
        }
        let due = match state.resolver.retry {
            Some(retry) => retry,
            None if state.resolver.requested => state
                .resolver
                .last_start
                .map_or(now, |start| start + MIN_TIME_BETWEEN_RESOLUTIONS),
            None => return None,
        };
        if now < due {
            return Some(due);
        }
        self.start_resolution(state, now);
        None
    }

    /// The end of a resolution, of `result`.
    fn resolved(
        self: &Arc<Self>,
        state: &mut State,
        result: Result<Vec<Address>, ChannelFailure>,
        now: Instant,
    ) {
        let failure = match result {
            Ok(addresses) if !addresses.is_empty() => {
                state.resolver.backoff = Backoff::new();
                state.resolver.addresses = Some(addresses.clone());
                if !matches!(state.phase, Phase::Idle) {
                    self.connect_to(state, addresses);
                }
                return;
            }
            Ok(_) => ChannelFailure::NoAddress,
            Err(failure) => failure,
        };
        if !self.grpcio() {
            // The failure of the pass of a Rust client, of its calls.
            state.go_idle(Some(failure));
            return;
        }
        // As gRPC's resolver, which tries a resolution that failed again a backoff after its end,
        // and its `pick_first`, which keeps the addresses of the last one that succeeded.
        state.resolver.retry = Some(now + state.resolver.backoff.next());
        if state.resolver.addresses.is_none() && !matches!(state.phase, Phase::Idle) {
            state.list = None;
            state.pending = None;
            state.picked = None;
            state.phase = Phase::TransientFailure(failure);
        }
    }

    /// Connect to the subchannels of `addresses`, of a new resolution: at once, as gRPC's
    /// `pick_first`, but of a channel that is ready, which goes on until one of them is (see the
    /// module).
    fn connect_to(&self, state: &mut State, mut addresses: Vec<Address>) {
        let mut seen = Vec::with_capacity(addresses.len());
        addresses.retain(|address| {
            if seen.contains(address) {
                return false;
            }
            seen.push(address.clone());
            true
        });
        let subchannels = addresses
            .into_iter()
            .map(|address| {
                Subchannel::of(
                    &self.pool,
                    address,
                    &self.args,
                    &self.target,
                    &self.endpoint,
                    &self.runtime,
                )
            })
            .collect();
        let list = List::new(subchannels, &self.wake);
        state.last_picked = None;
        if state.picked.is_some() {
            state.pending = Some(list);
            return;
        }
        // A failure of the resolver, of no subchannel, ends with it; that of the subchannels stays,
        // as gRPC's `pick_first` reports no change while it connects in transient failure.
        let of_the_resolver = state.list.is_none();
        state.pending = None;
        state.list = Some(list);
        if of_the_resolver || !matches!(state.phase, Phase::TransientFailure(_)) {
            state.phase = Phase::Connecting;
        }
    }

    /// The connection of the picked subchannel ended (see the module).
    fn lost(&self, state: &mut State) {
        let last_picked = state.picked.take().map(|(subchannel, _)| subchannel);
        if self.grpcio() {
            self.request_resolution(state);
            if let Some(pending) = state.pending.take() {
                state.phase = if pending.complete {
                    Phase::TransientFailure(pending.last_failure())
                } else {
                    Phase::Connecting
                };
                state.list = Some(pending);
                return;
            }
        }
        state.go_idle(None);
        state.last_picked = last_picked;
    }

    /// A step of `pick_first` over the subchannels of the channel (see [`List::step`]).
    fn pick_first(&self, state: &mut State, now: Instant) {
        if let Some(pending) = &mut state.pending {
            match pending.step(now, self.attempt_delay) {
                Outcome::Ready(index, connection) => {
                    let pending = state.pending.take().expect("a pending list");
                    self.picked(state, pending, index, connection);
                }
                Outcome::Failed(failure) => {
                    // As gRPC's `pick_first`, which takes the addresses of the resolution all the
                    // same, and lets go of the connection that it has.
                    state.list = state.pending.take();
                    state.picked = None;
                    state.phase = Phase::TransientFailure(failure);
                    self.request_resolution(state);
                }
                Outcome::Pending => {}
            }
        }
        if state.picked.is_some() {
            return;
        }
        let Some(list) = &mut state.list else {
            return;
        };
        match list.step(now, self.attempt_delay) {
            Outcome::Ready(index, connection) => {
                let list = state.list.take().expect("a list");
                self.picked(state, list, index, connection);
            }
            Outcome::Failed(failure) if self.grpcio() => {
                state.phase = Phase::TransientFailure(failure);
                self.request_resolution(state);
            }
            Outcome::Failed(failure) => state.go_idle(Some(failure)),
            Outcome::Pending => {}
        }
    }

    /// The channel picked the subchannel of `index` of `list`, of its connection of the number
    /// `connection`: its calls take it, and the other subchannels of the list are let go.
    fn picked(&self, state: &mut State, list: List, index: usize, connection: u64) {
        let list = list.of_the_picked(index);
        state.picked = Some((Arc::clone(&list.entries[0].subchannel), connection));
        state.list = Some(list);
        state.pending = None;
        state.phase = Phase::Ready;
    }

    /// The addresses of the subchannels of the channel, of the tests: those of its list, in their
    /// order, and whether one of them is picked.
    #[cfg(test)]
    pub(super) fn subchannel_addresses(&self) -> (Vec<String>, bool) {
        let state = self.state();
        let addresses = state
            .list
            .iter()
            .flat_map(|list| list.entries.iter())
            .map(|entry| entry.subchannel.address().to_string())
            .collect();
        (addresses, state.picked.is_some())
    }

    /// The subchannels of the list of the channel, of the tests.
    #[cfg(test)]
    pub(super) fn subchannels(&self) -> Vec<Arc<Subchannel>> {
        let state = self.state();
        state
            .list
            .iter()
            .flat_map(|list| list.entries.iter())
            .map(|entry| Arc::clone(&entry.subchannel))
            .collect()
    }

    /// The picked subchannel, of the tests.
    #[cfg(test)]
    pub(super) fn picked_subchannel(&self) -> Option<Arc<Subchannel>> {
        self.state()
            .picked
            .as_ref()
            .map(|(subchannel, _)| Arc::clone(subchannel))
    }

    /// Whether the channel is in transient failure, of the tests.
    #[cfg(test)]
    pub(super) fn in_transient_failure(&self) -> bool {
        matches!(self.state().phase, Phase::TransientFailure(_))
    }

    /// Whether the channel is idle, of the tests.
    #[cfg(test)]
    pub(super) fn is_idle(&self) -> bool {
        matches!(self.state().phase, Phase::Idle)
    }
}

impl Drop for ClientChannel {
    /// Ends the work of the channel in the background, which waits for a wake-up.
    fn drop(&mut self) {
        self.wake.notify_one();
    }
}

/// The earliest of `a` and `b`.
fn earliest(a: Option<Instant>, b: Option<Instant>) -> Option<Instant> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    }
}

/// The addresses of `target`, of a lookup now (see [`Target::resolve`]), within the connect
/// deadline, as tonic's connect timeout bounded the lookup, but through a DNS server, whose lookup
/// ends of the timeout of gRPC's resolution itself.
pub(super) async fn resolution(target: &Target) -> Result<Vec<Address>, ChannelFailure> {
    let resolved = if target.dns_server().is_some() {
        target.resolve().await
    } else {
        match tokio::time::timeout(target.connect_timeout(), target.resolve()).await {
            Ok(resolved) => resolved,
            Err(_) => Err(ResolveFailure::Timeout),
        }
    };
    resolved.map_err(|failure| ChannelFailure::Resolve {
        name: target.name(),
        failure,
    })
}

/// The connector of the channel of tonic of a target that gRPC refuses (see
/// [`refused_connection`]).
#[derive(Clone)]
struct RefusedConnector {
    code: i32,
    message: String,
    grpcio: bool,
}

impl tonic::codegen::Service<tonic::codegen::http::Uri> for RefusedConnector {
    type Response = TokioIo<Handover>;
    type Error = ConnectError;
    type Future = tonic::codegen::BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _: tonic::codegen::http::Uri) -> Self::Future {
        let this = self.clone();
        Box::pin(async move {
            match refused_connection(this.code, &this.message).await {
                Ok(ready) => Ok(TokioIo::new(Handover::new(ready))),
                Err(failure) => Err(ConnectError::of(&failure, this.grpcio)),
            }
        })
    }
}
