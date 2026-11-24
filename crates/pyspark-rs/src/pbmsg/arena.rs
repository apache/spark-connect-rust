//! The native messages of the runtime ([`Native`]) as the values of fields hold them: [`NRef`], a
//! count of the references of a message, as an `Arc` holds one, in slabs of the messages that one
//! parse makes ([`Arena`]). A parse then takes one allocation for each [`SLAB`] messages rather than
//! one for each, and the messages of a slab are released at once, of one allocation, once no value
//! holds any of them: as Python reads the messages of a parse, whose objects take their fields, the
//! allocator of the process got a small hole back for each message, between the values of the
//! fields that its object took, and many such holes slowed every later allocation of the process
//! down. The first slabs of a parse are smaller ([`FIRST_SLAB`]), so that a parse of few messages
//! holds little more than its messages. A message that a copy makes is an allocation of its own,
//! as of an `Arc`.

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::cell::Cell;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{fence, AtomicUsize, Ordering};
use std::sync::Mutex;

use super::message::Slots;
use super::native::Native;

/// The messages of a slab of an [`Arena`], once its first slabs are full (see [`FIRST_SLAB`]).
pub(crate) const SLAB: usize = 64;

/// The messages of the first slab of an [`Arena`]: each slab that follows has room for twice as
/// many as the one before, up to [`SLAB`], so that a parse of few messages, which a value may hold
/// for long, holds no more than as many nodes again as its messages.
const FIRST_SLAB: usize = 4;

/// A message, and the references of it.
struct Node {
    refs: AtomicUsize,
    /// The slab of the node; none of a node that is an allocation of its own ([`NRef::new`]).
    slab: Option<NonNull<Header>>,
    native: ManuallyDrop<Native>,
}

/// The header of a slab, which its nodes follow.
struct Header {
    /// The nodes of the slab that are alive, and one more while an arena takes nodes of it: a slab
    /// of an arena counts all its nodes alive from the start, of no count for each node taken, and
    /// those that the arena did not take are given back when it takes no node of it any longer.
    live: AtomicUsize,
    /// The nodes that the slab has room for.
    capacity: usize,
}

/// The offset of the first node of a slab, after its header, of any capacity.
const NODES: usize = std::mem::size_of::<Header>().next_multiple_of(std::mem::align_of::<Node>());

/// The layout of a slab of `capacity` nodes.
fn layout(capacity: usize) -> Layout {
    let nodes = Layout::array::<Node>(capacity).expect("the layout of a slab");
    let (layout, offset) = Layout::new::<Header>()
        .extend(nodes)
        .expect("the layout of a slab");
    debug_assert_eq!(offset, NODES);
    layout.pad_to_align()
}

/// The slabs of [`SLAB`] nodes released, kept for the arenas of the parses that follow, up to
/// [`KEPT`]: the allocator of the process gives the memory of the slabs released at the top of its
/// heap back to the system, so that each parse of a large message took it again, of a fault of each
/// page (for a parse of 100,000 messages, as many faults as of the objects of its messages). The
/// smaller first slabs of an arena are released as they are.
struct Kept(Vec<NonNull<Header>>);

// SAFETY: a slab kept is free memory, which no node or arena holds.
unsafe impl Send for Kept {}

static KEPT_SLABS: Mutex<Kept> = Mutex::new(Kept(Vec::new()));

/// The slabs that [`KEPT_SLABS`] keeps at most: 64 MiB of them, of the messages of a parse of
/// some 600,000 messages.
const KEPT: usize = (64 << 20) / (SLAB * std::mem::size_of::<Node>());

/// A slab of an arena: a slab kept, else a new one, of `live` references.
fn arena_slab(live: usize) -> NonNull<Header> {
    let kept = KEPT_SLABS.lock().unwrap_or_else(|e| e.into_inner()).0.pop();
    match kept {
        Some(slab) => {
            // SAFETY: the slab is free, of the layout of `SLAB` nodes.
            unsafe { slab.as_ref().live.store(live, Ordering::Relaxed) };
            slab
        }
        None => new_slab(SLAB, live),
    }
}

/// A new slab of `capacity` nodes, of `live` references.
fn new_slab(capacity: usize, live: usize) -> NonNull<Header> {
    let layout = layout(capacity);
    // SAFETY: the layout is of a header and of at least one node, so of a size that is not zero.
    let header = unsafe { alloc(layout) }.cast::<Header>();
    let Some(header) = NonNull::new(header) else {
        handle_alloc_error(layout)
    };
    // SAFETY: `header` is a new allocation of the layout, aligned for its header.
    unsafe {
        header.as_ptr().write(Header {
            live: AtomicUsize::new(live),
            capacity,
        })
    };
    header
}

/// The node `index` of the slab `slab`.
///
/// # Safety
///
/// `slab` is a slab of more than `index` nodes.
unsafe fn node_at(slab: NonNull<Header>, index: usize) -> NonNull<Node> {
    // SAFETY: the caller's; the slab is alive while a node or an arena holds it.
    unsafe {
        let first = slab.as_ptr().cast::<u8>().add(NODES).cast::<Node>();
        NonNull::new_unchecked(first.add(index))
    }
}

/// Release a reference of the slab `slab`: the slab is released with its last.
///
/// # Safety
///
/// The caller holds a reference of the slab, which it no longer uses.
unsafe fn release(slab: NonNull<Header>) {
    // SAFETY: the caller's.
    unsafe { release_many(slab, 1) }
}

/// Release `count` references of the slab `slab`: the slab is released with its last.
///
/// # Safety
///
/// The caller holds `count` references of the slab, which it no longer uses.
unsafe fn release_many(slab: NonNull<Header>, count: usize) {
    // SAFETY: the caller's.
    let header = unsafe { slab.as_ref() };
    if header.live.fetch_sub(count, Ordering::Release) != count {
        return;
    }
    // As `Arc` releases its value: every use of the slab by another thread happens before.
    fence(Ordering::Acquire);
    if header.capacity == SLAB {
        let mut kept = KEPT_SLABS.lock().unwrap_or_else(|e| e.into_inner());
        if kept.0.len() < KEPT {
            kept.0.push(slab);
            return;
        }
    }
    let layout = layout(header.capacity);
    // SAFETY: the slab was allocated of this layout, and no node or arena holds it any longer.
    unsafe { dealloc(slab.as_ptr().cast(), layout) };
}

/// Release the node `node`, whose message was dropped or taken: a reference of its slab, else the
/// node itself, of an allocation of its own.
///
/// # Safety
///
/// No reference holds the node any longer, and its message was dropped or taken.
unsafe fn release_node(node: NonNull<Node>) {
    // SAFETY: the caller's; a node of no slab is a box of its own (`NRef::new`).
    unsafe {
        match (*node.as_ptr()).slab {
            Some(slab) => release(slab),
            None => drop(Box::from_raw(node.as_ptr())),
        }
    }
}

/// A native message held by the value of a field, of a count of its references, as an `Arc`
/// holds it: the message is released with its last reference, and its slab with the last of its
/// messages.
pub(crate) struct NRef(NonNull<Node>);

// SAFETY: as `Arc<Native>`: the message is shared of its references only, and `Native` is `Send`
// and `Sync`.
unsafe impl Send for NRef {}
// SAFETY: as above.
unsafe impl Sync for NRef {}

/// A native message is `Send` and `Sync`, as an `NRef` takes it to be.
#[allow(dead_code)]
fn native_is_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<Native>();
}

impl NRef {
    /// The message `native`, of an allocation of its own, as of an `Arc`.
    pub(crate) fn new(native: Native) -> NRef {
        let node = Box::new(Node {
            refs: AtomicUsize::new(1),
            slab: None,
            native: ManuallyDrop::new(native),
        });
        NRef(NonNull::from(Box::leak(node)))
    }

    /// The message `native` in the node `index` of `slab`, of one reference.
    ///
    /// # Safety
    ///
    /// The node is of the slab, and free; the caller counted it in the slab's `live`.
    unsafe fn write(slab: NonNull<Header>, index: usize, native: Native) -> NRef {
        // SAFETY: the caller's.
        unsafe {
            let node = node_at(slab, index);
            node.as_ptr().write(Node {
                refs: AtomicUsize::new(1),
                slab: Some(slab),
                native: ManuallyDrop::new(native),
            });
            NRef(node)
        }
    }

    fn node(&self) -> &Node {
        // SAFETY: the node is alive while a reference of it is.
        unsafe { self.0.as_ref() }
    }

    /// Whether `a` and `b` are the same message.
    pub(crate) fn ptr_eq(a: &NRef, b: &NRef) -> bool {
        a.0 == b.0
    }

    /// The references of the message.
    pub(crate) fn strong_count(this: &NRef) -> usize {
        this.node().refs.load(Ordering::Acquire)
    }

    /// The message, to change it, if no other reference holds it.
    pub(crate) fn get_mut(this: &mut NRef) -> Option<&mut Native> {
        if this.node().refs.load(Ordering::Acquire) != 1 {
            return None;
        }
        // SAFETY: `this` is the only reference of the message, and the caller holds it mutably.
        Some(unsafe { &mut (*this.0.as_ptr()).native })
    }

    /// The set fields and the unknown fields of the message, taken from it, if no other reference
    /// holds it (the rest of it then released); else `this`. As [`NRef::try_unwrap`] of the
    /// fields, of no copy of the message out of its node.
    #[cfg(test)]
    pub(crate) fn into_parts(this: NRef) -> Result<(Slots, Vec<u8>), NRef> {
        Self::into_parts_releasing(this, &mut Releases::default())
    }

    /// [`NRef::into_parts`], of the node released with `releases` (see [`Releases`]).
    pub(crate) fn into_parts_releasing(
        this: NRef,
        releases: &mut Releases,
    ) -> Result<(Slots, Vec<u8>), NRef> {
        // `this` is the only reference, which no other thread can copy meanwhile.
        if this.node().refs.load(Ordering::Acquire) != 1 {
            return Err(this);
        }
        let this = ManuallyDrop::new(this);
        // SAFETY: no other reference holds the message; its fields are taken, the rest of it is
        // dropped once, and its node released.
        unsafe {
            let node = this.0.as_ptr();
            let native: &mut ManuallyDrop<Native> = &mut (*node).native;
            let fields = &mut native.fields;
            let parts = (
                std::mem::take(&mut fields.slots),
                std::mem::take(&mut fields.unknown),
            );
            ManuallyDrop::drop(&mut (*node).native);
            releases.release(this.0);
            Ok(parts)
        }
    }

    /// Whether the message is a node of a slab of a parse (see [`Arena`]), else an allocation of
    /// its own.
    pub(crate) fn in_slab(this: &NRef) -> bool {
        this.node().slab.is_some()
    }

    /// Move the message out of its slab into an allocation of its own, as [`NRef::new`] makes
    /// one, if no other reference holds it: it then keeps the slab alive no longer, which is
    /// released with the last of its other messages.
    pub(crate) fn rebox(this: &mut NRef) {
        let node = this.node();
        let Some(slab) = node.slab else {
            return;
        };
        if node.refs.load(Ordering::Acquire) != 1 {
            return;
        }
        // SAFETY: `this` is the only reference of the message, which the caller holds mutably:
        // the message is taken out of its node once, `this` set to the new node (of no drop of the
        // old one), and the old node released as a node whose message was taken.
        unsafe {
            let native = ManuallyDrop::take(&mut (*this.0.as_ptr()).native);
            let fresh = ManuallyDrop::new(NRef::new(native));
            this.0 = fresh.0;
            release(slab);
        }
    }

    /// The message itself, if no other reference holds it; else `this`.
    pub(crate) fn try_unwrap(this: NRef) -> Result<Native, NRef> {
        if this
            .node()
            .refs
            .compare_exchange(1, 0, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return Err(this);
        }
        let this = ManuallyDrop::new(this);
        // SAFETY: no reference holds the message any longer; it is read once, and its node is
        // released, not dropped.
        unsafe {
            let node = this.0.as_ptr();
            let native = ManuallyDrop::take(&mut (*node).native);
            release_node(this.0);
            Ok(native)
        }
    }
}

impl Clone for NRef {
    fn clone(&self) -> NRef {
        // As `Arc`: a count that could overflow is of leaked references, which abort.
        if self.node().refs.fetch_add(1, Ordering::Relaxed) > isize::MAX as usize {
            std::process::abort();
        }
        NRef(self.0)
    }
}

impl Deref for NRef {
    type Target = Native;

    fn deref(&self) -> &Native {
        &self.node().native
    }
}

impl Drop for NRef {
    fn drop(&mut self) {
        // The only reference, which no other can copy meanwhile (an `NRef` has no weak
        // references), of no atomic write; else as `Arc`.
        let refs = &self.node().refs;
        if refs.load(Ordering::Acquire) != 1 {
            if refs.fetch_sub(1, Ordering::Release) != 1 {
                return;
            }
            fence(Ordering::Acquire);
        }
        // SAFETY: this was the last reference of the message, which is dropped once, and its
        // node released.
        unsafe {
            ManuallyDrop::drop(&mut (*self.0.as_ptr()).native);
            release_node(self.0);
        }
    }
}

/// The releases of the nodes whose messages a read takes, one after another (see
/// [`NRef::into_parts_releasing`]): counted for the slab of the last of them, and released at once
/// when a node of another slab follows or the releases are dropped, of one atomic operation for the
/// nodes of a slab that the read takes in a row rather than one for each.
#[derive(Default)]
pub(crate) struct Releases {
    slab: Option<NonNull<Header>>,
    count: usize,
}

impl Releases {
    /// Release the node `node`: count it, of a slab, else release it now.
    ///
    /// # Safety
    ///
    /// As of [`release_node`]: no reference holds the node any longer, and its message was dropped
    /// or taken.
    unsafe fn release(&mut self, node: NonNull<Node>) {
        // SAFETY: the caller's; the slab of a node counted is alive until the releases are.
        unsafe {
            match (*node.as_ptr()).slab {
                Some(slab) if self.slab == Some(slab) => self.count += 1,
                Some(slab) => {
                    self.flush();
                    self.slab = Some(slab);
                    self.count = 1;
                }
                None => release_node(node),
            }
        }
    }

    /// Release the nodes counted.
    fn flush(&mut self) {
        if let Some(slab) = self.slab.take() {
            // SAFETY: the nodes counted of the slab are no longer held (see `release`).
            unsafe { release_many(slab, self.count) };
        }
    }
}

impl Drop for Releases {
    fn drop(&mut self) {
        self.flush();
    }
}

/// The slabs of the messages of a parse: each message that it makes takes the next node of its
/// slab, and a full slab is followed by a new one ([`Arena::alloc`]), of twice the room, up to
/// [`SLAB`].
pub(crate) struct Arena {
    slab: Cell<Option<NonNull<Header>>>,
    next: Cell<usize>,
    /// The nodes of the slab, none before the first.
    capacity: Cell<usize>,
}

impl Arena {
    pub(crate) fn new() -> Arena {
        Arena {
            slab: Cell::new(None),
            next: Cell::new(0),
            capacity: Cell::new(0),
        }
    }

    /// The message `native`, in the next node of the slab of the arena.
    pub(crate) fn alloc(&self, native: Native) -> NRef {
        let slab = match self.slab.get() {
            Some(slab) if self.next.get() < self.capacity.get() => slab,
            _ => self.next_slab(),
        };
        let index = self.next.get();
        self.next.set(index + 1);
        // SAFETY: the node is free, and counted alive since the slab was made.
        unsafe { NRef::write(slab, index, native) }
    }

    /// The slab that follows the slab of the arena, which is full, of twice its room, up to
    /// [`SLAB`] (a slab kept then, see [`KEPT_SLABS`]).
    #[cold]
    #[inline(never)]
    fn next_slab(&self) -> NonNull<Header> {
        self.give_back();
        let capacity = (self.capacity.get() * 2).clamp(FIRST_SLAB, SLAB);
        // All the nodes of the slab, and the arena's own reference.
        let slab = if capacity == SLAB {
            arena_slab(SLAB + 1)
        } else {
            new_slab(capacity, capacity + 1)
        };
        self.slab.set(Some(slab));
        self.next.set(0);
        self.capacity.set(capacity);
        slab
    }

    /// Give back the references of the slab of the arena: its own, and those of the nodes that it
    /// did not take.
    fn give_back(&self) {
        if let Some(slab) = self.slab.take() {
            // SAFETY: the arena held these references, and takes no node of the slab any longer.
            unsafe { release_many(slab, self.capacity.get() - self.next.get() + 1) };
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        self.give_back();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbmsg::message::Fields;
    use crate::pbmsg::pool::pool;

    /// The tests of the slabs kept, which another test of slabs of [`SLAB`] nodes would change.
    static SERIAL: Mutex<()> = Mutex::new(());

    /// A native `Literal` of no field, of the unknown fields `unknown`, which tell it apart.
    fn native(unknown: Vec<u8>) -> Native {
        let desc = pool()
            .message("spark.connect.Expression.Literal")
            .expect("Literal");
        Native::new(
            Fields {
                desc,
                slots: Slots::default(),
                unknown,
            },
            0,
        )
    }

    fn live(slab: NonNull<Header>) -> usize {
        // SAFETY: of a slab that a reference of the test or an arena holds.
        unsafe { slab.as_ref().live.load(Ordering::Acquire) }
    }

    fn capacity(slab: NonNull<Header>) -> usize {
        // SAFETY: as above.
        unsafe { slab.as_ref().capacity }
    }

    fn slab_of(this: &NRef) -> NonNull<Header> {
        this.node().slab.expect("a node of a slab")
    }

    /// Each slab that `refs` or `arena` holds counts the nodes of it that `refs` holds, and, of the
    /// slab of the arena, the arena itself and the nodes that it did not take.
    fn check(refs: &[Option<NRef>], arena: Option<&Arena>) {
        let mut counts: Vec<(NonNull<Header>, usize)> = Vec::new();
        let mut count = |slab, n| match counts.iter_mut().find(|(s, _)| *s == slab) {
            Some((_, c)) => *c += n,
            None => counts.push((slab, n)),
        };
        for r in refs.iter().flatten() {
            count(slab_of(r), 1);
        }
        if let Some(slab) = arena.and_then(|a| a.slab.get()) {
            let arena = arena.unwrap();
            count(slab, arena.capacity.get() - arena.next.get() + 1);
        }
        for (slab, count) in counts {
            assert_eq!(live(slab), count);
        }
    }

    /// Each message of an arena is the one that it was given, of one reference, in slabs of twice
    /// the room of the one before, up to [`SLAB`], and a slab counts the messages of it that are
    /// alive, and the arena while it takes nodes of it, whatever the order in which they are
    /// released: before the arena or after it, of a clone released by another thread, of
    /// `try_unwrap` or of `into_parts`, which take a message only of no other reference.
    #[test]
    fn a_slab_counts_its_messages_alive_in_any_order_of_release() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let arena = Arena::new();
        let count = 200;
        let mut refs: Vec<Option<NRef>> = (0..count)
            .map(|i| Some(arena.alloc(native((i as u32).to_le_bytes().to_vec()))))
            .collect();
        let mut slabs: Vec<NonNull<Header>> = Vec::new();
        for (i, r) in refs.iter().enumerate() {
            let r = r.as_ref().unwrap();
            assert_eq!(r.fields.unknown, (i as u32).to_le_bytes());
            assert_eq!(NRef::strong_count(r), 1);
            if slabs.last() != Some(&slab_of(r)) {
                slabs.push(slab_of(r));
            }
        }
        let capacities: Vec<usize> = slabs.iter().map(|s| capacity(*s)).collect();
        assert_eq!(capacities, [4, 8, 16, 32, 64, 64, 64]);
        check(&refs, Some(&arena));
        assert_eq!(live(slabs[6]), 200 - 188 + (64 - 12) + 1);

        // Clones that another thread releases, and a clone that is the only one left.
        let clones: Vec<NRef> = refs[..100].iter().map(|r| r.clone().unwrap()).collect();
        assert!(clones.iter().all(|c| NRef::strong_count(c) == 2));
        std::thread::spawn(move || drop(clones)).join().unwrap();
        let kept = refs[1].clone();
        refs[1] = None;
        assert_eq!(NRef::strong_count(kept.as_ref().unwrap()), 1);
        refs.push(kept);
        check(&refs, Some(&arena));

        // `try_unwrap` and `into_parts` of a shared message give it back; of the only one, take it.
        let shared = refs[2].take().unwrap();
        let other = shared.clone();
        let shared = NRef::try_unwrap(shared).map(|_| ()).unwrap_err();
        let shared = NRef::into_parts(shared).map(|_| ()).unwrap_err();
        drop(other);
        let (_, unknown) = NRef::into_parts(shared).map_err(|_| ()).unwrap();
        assert_eq!(unknown, 2u32.to_le_bytes());
        let unwrapped = NRef::try_unwrap(refs[3].take().unwrap())
            .map_err(|_| ())
            .unwrap();
        assert_eq!(unwrapped.fields.unknown, 3u32.to_le_bytes());
        check(&refs, Some(&arena));

        // The messages of the first slabs released before the arena, those of the last after it.
        for r in &mut refs[..60] {
            *r = None;
        }
        check(&refs, Some(&arena));
        drop(arena);
        check(&refs, None);
        assert_eq!(live(slabs[6]), 12);
        for r in &mut refs[189..200] {
            *r = None;
        }
        assert_eq!(live(slabs[6]), 1);
        // A mutable reference of the only reference of a message, of none of a shared one.
        let mut last = refs[188].take().unwrap();
        NRef::get_mut(&mut last).unwrap().fields.unknown.push(1);
        let copy = last.clone();
        assert!(NRef::get_mut(&mut last).is_none());
        assert!(NRef::ptr_eq(&last, &copy));
        drop((last, copy));
        check(&refs, None);
        drop(refs);
    }

    /// The releases of the nodes of a read count those of a slab in a row, and release them at
    /// once when a node of another slab follows, or when they are dropped: the slab counts them
    /// until then.
    #[test]
    fn releases_count_the_nodes_of_a_slab_in_a_row() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let arena = Arena::new();
        let mut refs: Vec<Option<NRef>> = (0..12)
            .map(|i| Some(arena.alloc(native(vec![i]))))
            .collect();
        drop(arena);
        let (first, second) = (
            slab_of(refs[0].as_ref().unwrap()),
            slab_of(refs[4].as_ref().unwrap()),
        );
        assert_eq!((live(first), live(second)), (4, 8));
        let mut releases = Releases::default();
        for i in [4, 5, 6] {
            let (_, unknown) = NRef::into_parts_releasing(refs[i].take().unwrap(), &mut releases)
                .map_err(|_| ())
                .unwrap();
            assert_eq!(unknown, [i as u8]);
        }
        assert_eq!(live(second), 8);
        let one = NRef::new(native(vec![99]));
        let other = one.clone();
        let one = NRef::into_parts_releasing(one, &mut releases)
            .map(|_| ())
            .unwrap_err();
        drop(other);
        NRef::into_parts_releasing(one, &mut releases)
            .map_err(|_| ())
            .unwrap();
        assert_eq!(live(second), 8);
        NRef::into_parts_releasing(refs[0].take().unwrap(), &mut releases)
            .map_err(|_| ())
            .unwrap();
        assert_eq!((live(first), live(second)), (4, 5));
        drop(releases);
        assert_eq!(live(first), 3);
        check(&refs, None);
        drop(refs);
    }

    /// A message of a slab that no other reference holds moves out of it into an allocation of
    /// its own (`NRef::rebox`), released from the slab, and reads as it did; of another reference,
    /// it stays.
    #[test]
    fn a_message_of_its_own_moves_out_of_its_slab() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let arena = Arena::new();
        let mut refs: Vec<NRef> = (0..3).map(|i| arena.alloc(native(vec![i]))).collect();
        drop(arena);
        let slab = slab_of(&refs[0]);
        assert_eq!(live(slab), 3);
        let shared = refs[1].clone();
        NRef::rebox(&mut refs[1]);
        assert!(NRef::in_slab(&refs[1]) && NRef::ptr_eq(&refs[1], &shared));
        drop(shared);
        NRef::rebox(&mut refs[1]);
        assert!(!NRef::in_slab(&refs[1]));
        assert_eq!(refs[1].fields.unknown, [1]);
        assert_eq!(live(slab), 2);
        NRef::rebox(&mut refs[1]);
        assert!(!NRef::in_slab(&refs[1]));
        drop(refs);
    }

    /// A message of its own ([`NRef::new`]) is of no slab, and is released with its last
    /// reference, of any thread, or taken of `try_unwrap` and `into_parts` of the only one.
    #[test]
    fn a_message_of_its_own_is_released_with_its_last_reference() {
        let one = NRef::new(native(vec![1]));
        assert!(one.node().slab.is_none());
        let clone = one.clone();
        assert_eq!(NRef::strong_count(&one), 2);
        std::thread::spawn(move || drop(clone)).join().unwrap();
        let one = NRef::try_unwrap(one).map_err(|_| ()).unwrap();
        assert_eq!(one.fields.unknown, [1]);
        let two = NRef::new(native(vec![2]));
        let other = two.clone();
        let two = NRef::into_parts(two).map(|_| ()).unwrap_err();
        drop(other);
        let (_, unknown) = NRef::into_parts(two).map_err(|_| ()).unwrap();
        assert_eq!(unknown, [2]);
        drop(NRef::new(native(vec![3])));
    }

    /// A slab of [`SLAB`] nodes of an arena released is kept for the arenas that follow, which
    /// take its nodes again; the first slabs of an arena, which are smaller, are not kept.
    #[test]
    fn an_arena_takes_a_slab_released_again() {
        let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let first_slabs = 4 + 8 + 16 + 32;
        let take = |arena: &Arena| -> Vec<NRef> {
            (0..=first_slabs)
                .map(|i| arena.alloc(native(vec![i as u8])))
                .collect()
        };
        let arena = Arena::new();
        let messages = take(&arena);
        let slab = slab_of(&messages[first_slabs]);
        assert_eq!(capacity(slab), SLAB);
        drop((arena, messages));
        let arena = Arena::new();
        let messages = take(&arena);
        assert_eq!(slab_of(&messages[first_slabs]), slab);
        assert_eq!(live(slab), SLAB + 1);
        drop((arena, messages));
    }
}
