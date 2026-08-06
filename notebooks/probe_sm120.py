"""Stage-by-stage probe for the sm_120 (Blackwell) FlexAttention hang.

Two RTX 5090 hosts hung silently at the first training forward (2026-08-06): the log stopped after
"[val] cached N batches", GPU util stayed at 0%, and no traceback came out for 1.8 hours. The first
forward is where the trunk builds its masks, where ``flex_is_usable`` compiles FlexAttention, and
where ``compile_trunk`` compiles the model forward. The same code is clean on sm_86 and sm_89.

There is no ssh to a probe box, so the log is the only channel. This script therefore does the
diagnosis itself:

* it walks the suspect stages one at a time and prints a marker with a timer before each one, so
  the hang point is the last marker in the log;
* a watchdog dumps the stack of EVERY thread every ``HAL_PROBE_WATCHDOG`` seconds, so a hang names
  the function it sits in instead of only the stage;
* the same watchdog lists the child processes with their CPU time, which separates a compiler that
  works (``ptxas`` burning CPU) from a worker pool that deadlocks (idle children).

Run it under a timeout so a hang exits instead of billing forever:

    timeout 600 uv run notebooks/probe_sm120.py; echo probe_rc=$?

rc 0 = all stages pass. rc 124 = the stage after the last printed marker hangs.

Env knobs, to run the fix candidates without a code change:

    HAL_PROBE_WATCHDOG=45           # seconds between thread dumps (0 = off)
    HAL_PROBE_EAGER_BLOCK_MASK=1    # do not torch.compile create_block_mask
    HAL_PROBE_BACKEND=aot_eager     # torch.compile backend for the trunk (default inductor)
    TORCHINDUCTOR_COMPILE_THREADS=1 # read by inductor; serializes the async compile pool
"""

# %%
import faulthandler
import os
import sys
import threading
import time

_T0 = time.monotonic()


def stage(name: str) -> None:
    print(f"[probe +{time.monotonic() - _T0:7.1f}s] {name}", flush=True)


def child_processes() -> list[str]:
    """``comm`` and CPU seconds of every descendant, read from /proc (no ``ps`` in the image)."""
    rows = []
    ticks = os.sysconf("SC_CLK_TCK")
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or entry.name == str(os.getpid()):
            continue
        try:
            with open(f"/proc/{entry.name}/stat") as f:
                fields = f.read().rsplit(") ", 1)[1].split()
            with open(f"/proc/{entry.name}/comm") as f:
                comm = f.read().strip()
        except FileNotFoundError, ProcessLookupError, IndexError, PermissionError:
            continue
        cpu = (int(fields[11]) + int(fields[12])) / ticks  # utime + stime, fields shifted by comm
        rows.append(f"{entry.name}:{comm}:state={fields[0]}:cpu={cpu:.1f}s")
    return rows


def start_watchdog(period: float) -> None:
    """Dump all thread stacks and the child table every ``period`` seconds, forever."""
    faulthandler.dump_traceback_later(period, repeat=True, exit=False)

    def report() -> None:
        while True:
            time.sleep(period)
            print(f"[watchdog +{time.monotonic() - _T0:7.1f}s] children: {' '.join(child_processes())}", flush=True)

    threading.Thread(target=report, daemon=True).start()


watchdog_period = float(os.environ.get("HAL_PROBE_WATCHDOG", "45"))
if watchdog_period > 0:
    start_watchdog(watchdog_period)

stage("import torch")
import torch  # noqa: E402

stage("device facts")
print("[probe] torch:", torch.__version__, "cuda:", torch.version.cuda, flush=True)
try:
    import triton

    print("[probe] triton:", triton.__version__, flush=True)
except ImportError:
    print("[probe] triton: NOT INSTALLED", flush=True)
print("[probe] device:", torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)
props = torch.cuda.get_device_properties(0)
print("[probe] sm count:", props.multi_processor_count, "vram GiB:", round(props.total_memory / 2**30, 1), flush=True)
print("[probe] cpu_count:", os.cpu_count(), "affinity:", len(os.sched_getaffinity(0)), flush=True)
for var in sorted(k for k in os.environ if k.startswith(("TORCHINDUCTOR", "TRITON", "TORCH_LOGS", "HAL_PROBE"))):
    print(f"[probe] env {var}={os.environ[var]}", flush=True)

stage("import trunk")
from hal.training import trunk as trunk_mod  # noqa: E402
from hal.training.trunk import Trunk  # noqa: E402
from hal.training.trunk import TrunkConfig  # noqa: E402
from hal.training.trunk import block_mask  # noqa: E402
from hal.training.trunk import flex_is_usable  # noqa: E402

if os.environ.get("HAL_PROBE_EAGER_BLOCK_MASK") == "1":
    from torch.nn.attention.flex_attention import create_block_mask

    trunk_mod._create_block_mask = create_block_mask
    print("[probe] create_block_mask: eager (torch.compile removed)", flush=True)

B, L, d_model, n_heads = 32, 512, 256, 4
head_dim = d_model // n_heads
pad = torch.zeros(B, dtype=torch.long, device="cuda")

stage("block_mask build (compiles create_block_mask)")
mask = block_mask(pad, L, 128)

stage("eager flex_attention forward (uncompiled kernel)")
from torch.nn.attention.flex_attention import flex_attention  # noqa: E402

q, k, v = (torch.zeros(B, n_heads, L, head_dim, device="cuda", requires_grad=True) for _ in range(3))
flex_attention(q, k, v, block_mask=mask).sum().backward()
torch.cuda.synchronize()

stage("compiled flex_attention forward (compiles the Triton kernel)")
out = trunk_mod._flex_attention(q, k, v, block_mask=mask)
torch.cuda.synchronize()

stage("compiled flex_attention backward (compiles the bwd Triton kernel)")
out.sum().backward()
torch.cuda.synchronize()

stage("flex_is_usable probe")
print("[probe] flex usable:", flex_is_usable("cuda"), flush=True)

stage("trunk construction")
cfg = TrunkConfig(d_model=d_model, n_layers=8, n_heads=n_heads, L_ctx=L, attn_window=128)
trunk = Trunk(cfg).cuda()

stage("eager trunk forward+backward (flex kernels, uncompiled module)")
x = torch.randn(B, L, d_model, device="cuda")
y = trunk(x, pad)
y.sum().backward()
torch.cuda.synchronize()
print("[probe] eager path OK, attn_path:", trunk.attn_path, flush=True)

backend = os.environ.get("HAL_PROBE_BACKEND", "inductor")
stage(f"torch.compile'd trunk forward (backend={backend}) — the training path")
trunk.zero_grad()
compiled = torch.compile(trunk, backend=backend, dynamic=False)
y = compiled(x, pad)
torch.cuda.synchronize()

stage("compiled trunk backward")
y.sum().backward()
torch.cuda.synchronize()

stage("bf16 autocast compiled step (full training likeness)")
trunk.zero_grad()
with torch.autocast("cuda", dtype=torch.bfloat16):
    y = compiled(x, pad)
y.float().sum().backward()
torch.cuda.synchronize()

stage("ALL STAGES PASS")
faulthandler.cancel_dump_traceback_later()
sys.exit(0)
