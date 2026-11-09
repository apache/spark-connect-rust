"""pytest plugin: apply the known-failure manifest to the official suite run against the drop-in.

The drop-in mode of run_official_tests.py (``--dropin``) runs the official test files
against our own ``pyspark`` (the Rust-backed drop-in), so unlike rust_transport_plugin.py
it must NOT load the extension or inject a transport. This plugin only deselects (or, for
the drift check, selects only) the tests named in the environment, by node-id suffix, the
same way rust_transport_plugin.py does:

    RUST_PARITY_DESELECT     - newline-separated "Class::method" suffixes to drop
    RUST_PARITY_SELECT_ONLY  - newline-separated suffixes to keep (the drift check)
"""

import os


def _suffixes(raw):
    return [s.strip() for s in raw.splitlines() if s.strip()]


def pytest_collection_modifyitems(config, items):
    select_only = os.environ.get("RUST_PARITY_SELECT_ONLY", "")
    deselect = os.environ.get("RUST_PARITY_DESELECT", "")
    suffixes = _suffixes(select_only or deselect)
    if not suffixes:
        return

    def matches(nid):
        # "...::Class::method" or a parametrized "...::Class::method[param]".
        return any(nid.endswith(s) or f"::{s}[" in nid for s in suffixes)

    keep, drop = [], []
    for item in items:
        hit = matches(item.nodeid)
        # select-only keeps the hits and drops the rest; deselect does the opposite.
        if hit == bool(select_only):
            keep.append(item)
        else:
            drop.append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep
