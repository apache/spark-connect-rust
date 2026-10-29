# Arrow handoff repeated-call RSS control

This is a local diagnostic for PR #108, not a performance benchmark or proof of
leak freedom. Both the mock gRPC server and the native client run in one Python
process, so the measurement includes their allocations and allocator retention.

On 2026-10-29, the same 28-call probe was run in fresh processes on master
`d884e34` (IPC handoff) and the review revision of #108 based on `fe79d54`
(C Stream handoff; production changes since that head are comments only).
Both extensions were rebuilt with the same release Cargo configuration and Rust
1.99.1, then loaded using the same Python 3.12 / PyArrow 26 environment on macOS
arm64. Master ran first. These are single runs on a shared desktop.

Each call receives 128 one-MiB numeric batches from the same mock service,
validates the 16,777,216-row table's endpoints, deletes the table, forces Python
GC, and samples process RSS. The first four calls are warmup. All calls passed
validation.

| Handoff | First measured RSS (MiB) | Last measured RSS (MiB) | Measured minimum–maximum (MiB) |
| --- | ---: | ---: | ---: |
| Master IPC | 1129.23 | 3976.06 | 1129.23–3976.06 |
| PR C Stream | 418.81 | 569.47 | 418.81–569.47 |

Growth is not unique to the C Stream change; the master control grew much more
in this run. This does not identify the source of retention, isolate client
allocations, or establish a general memory advantage. The earlier C Stream run
reported 416.27 → 481.66 MiB; variation between runs remains unexplained. A claim
of flat post-GC RSS is not supported.

To reproduce, run the following from the PR checkout with its tests available,
setting `PYTHONPATH` to `<PR checkout>/python`, the directory containing `pyspark`.
For the master control, build master and set `PYTHONPATH` to `<master checkout>/python`;
keep this test harness and all other environment settings the same. `PROTOC`
and `PROTOC_INCLUDE` must identify an installed protocol compiler and includes.
The baseline IPC implementation does not include the new test file, so the
working directory must remain the PR checkout for both runs.

```python
import gc
import json
import runpy
import tempfile
from pathlib import Path

import numpy as np
import psutil
import pyarrow as pa

namespace = runpy.run_path("python/tests/test_arrow_handoff.py")

class TempFactory:
    def mktemp(self, name):
        return Path(tempfile.mkdtemp(prefix=name))

protocol = namespace["protocol"].__wrapped__(TempFactory())
batch = pa.record_batch(
    [pa.array(np.arange(131072, dtype=np.int64))], names=["value"]
)
process = psutil.Process()
samples = []
with namespace["_dataframe"]([batch] * 128, protocol=protocol) as (frame, session):
    for iteration in range(28):
        table = frame.toArrow()
        assert table.num_rows == 16777216
        assert table.column(0)[0].as_py() == 0
        assert table.column(0)[-1].as_py() == 131071
        del table
        gc.collect()
        samples.append(process.memory_info().rss / 1048576)
print(json.dumps({"warmup": 4, "post_gc_rss_mib": samples}))
```
