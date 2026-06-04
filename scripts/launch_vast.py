"""Queue for a vast.ai GPU under a price ceiling and run a training command on it.

The instance is fire-and-forget: this launcher pushes the current git SHA, waits
for an offer that clears the hardware bar, rents it, and injects the SHA + the
(base64'd) training command. The box then clones that SHA, trains, and tears
*itself* down — destroy on success (checkpoints are already in R2, logs in W&B),
stop on failure (for inspection). See docker/on-start.sh.

    python scripts/launch_vast.py                         # search-only: print offers, rent nothing
    python scripts/launch_vast.py --dry-run -- uv run experiments/001_flow_matching_baseline.py
    python scripts/launch_vast.py --max-price 0.80 -- uv run experiments/001_flow_matching_baseline.py --cfg.max-steps 100000

Secrets (R2 + W&B) are NOT passed by this launcher — they live as vast *account*
env-vars (Console → Account → Environment Vars) and inject into the box out-of-band,
so they never land in the instance's extra_env. The launcher only verifies they're
present. The host needs no secrets, just a vast API key (~/.config/vastai). An
optional GITHUB_TOKEN is used solely to pull a private ghcr image (the repo is public).
"""

import base64
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

import tyro
from loguru import logger
from vastai import VastAI

GHCR_USER = "ericyuegu"
# on-start.sh runs *before* the box clones the repo, so it can't ship via the git
# SHA like the rest of the code. Rather than bake it into the image (which strands
# the box on a stale copy until the next rebuild), we read it from the working tree
# and pass its contents inline as the onstart command — so editing it takes effect
# on the next launch, no rebuild.
ONSTART_PATH = Path(__file__).resolve().parents[1] / "docker" / "on-start.sh"

# Hardware bar. Field names are vast's query vocabulary (see `help(VastAI.search_offers)`),
# not the result-dict keys, which sometimes differ (dlperf_usd -> dlperf_per_dphtotal).
# The price cap lives in --max-price (impatience knob), substituted in at query time.
#   inet_down       : Mbps down
#   inet_down_cost  : $/GB egress -> $10/TB == 0.01
#   dlperf_usd      : DLPerf per $/hr
FILTERS = (
    "num_gpus=1",
    "total_flops>=28",  # >= 28 TFLOPS
    "dlperf>20",
    "dlperf_usd>70",  # DLPerf per $/hr
    "reliability>0.96",  # > 96%
    "inet_down>300",  # > 300 Mbps down
    "inet_down_cost<=0.01",  # <= $10/TB down
    "cuda_max_good>=13",  # host driver must support the cuda13 image; older drivers
    # silently fail torch CUDA init (err 804 forward-compat on consumer GPUs) -> CPU run
    "rentable=true",
)
ORDER = "dlperf_usd-"  # sort by DLPerf/$/hr, descending

# R2 + W&B secrets are NOT passed via `-e` (that lands them in the instance's
# extra_env, visible in `show instance` and to the host). They live as vast *account*
# env-vars (Console → Account → Environment Vars), injected out-of-band; the launcher
# only verifies they exist. The box recovers them in docker/on-start.sh.
REQUIRED_ACCOUNT_VARS = (
    "AWS_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_BUCKET",
    "WANDB_API_KEY",
)


# The ssbm ISO the box fetches at runtime (~1.4 GB); folded into the one-time download
# estimate alongside the MDS. The ~9 GB docker image pull is separate and not counted.
ISO_GB = 1.4


def storage_dph(offer: dict, disk: int) -> float:
    """$/hr to hold `disk` GB at this offer's storage rate (``storage_cost`` is $/GB/month)."""
    return offer.get("storage_cost", 0.0) * disk / 720.0


def effective_dph(offer: dict, disk: int) -> float:
    """Real $/hr once the disk is provisioned. The search-time ``dph_total`` is essentially
    the GPU/base rate — the disk we ask for at create time isn't in it — so a cheap GPU with
    a pricey 500 GB disk can quietly cost ~4x its quoted rate. This is what --max-price gates."""
    return offer["dph_total"] + storage_dph(offer, disk)


def download_cost(offer: dict, data_gb: float) -> float:
    """One-time ingress $ to pull the MDS dataset (+ ISO) at this offer's $/GB ``inet_down_cost``."""
    return (data_gb + ISO_GB) * offer["inet_down_cost"]


def upload_cost(offer: dict, upload_gb: float) -> float:
    """One-time egress $ to stream checkpoints out to R2 at this offer's $/GB ``inet_up_cost``."""
    return upload_gb * offer["inet_up_cost"]


def amortized_dph(offer: dict, *, disk: int, data_gb: float, upload_gb: float, run_hours: float) -> float:
    """Effective $/hr with the one-time transfer costs (download + upload) spread over the
    expected ``run_hours``. Adding a flat $ to a $/hr rate only makes sense once amortized, so
    a shorter run carries a heavier per-hour transfer tax."""
    one_time = download_cost(offer, data_gb) + upload_cost(offer, upload_gb)
    return effective_dph(offer, disk) + one_time / run_hours


def value_metric(offer: dict, *, disk: int, data_gb: float, upload_gb: float, run_hours: float) -> float:
    """Ranking key (eff$/dlperf/hr): amortized effective $/hr per unit DLPerf. Lower is better
    bang-for-buck — folds the GPU+disk rate and the one-time download+upload into a single
    perf-normalized cost. This is what offers are sorted on."""
    return amortized_dph(offer, disk=disk, data_gb=data_gb, upload_gb=upload_gb, run_hours=run_hours) / offer["dlperf"]


def build_query(max_price: float, disk: int, min_vram: int) -> str:
    # dph_total<max_price is a safe coarse prefilter: effective_dph >= dph_total, so any
    # offer clearing the effective cap also clears this. The real (disk-inclusive) cap is
    # enforced client-side in search(), since storage_cost*disk isn't expressible here.
    q = [*FILTERS, f"disk_space>={disk}", f"dph_total<{max_price}"]
    if min_vram > 0:  # vast `gpu_ram` query is in GB; 0 = no VRAM floor
        q.append(f"gpu_ram>={min_vram}")
    return " ".join(q)


def search(
    vast: VastAI,
    *,
    max_price: float,
    limit: int,
    disk: int,
    min_vram: int,
    data_gb: float,
    upload_gb: float,
    run_hours: float,
) -> list[dict]:
    """Offers whose *effective* $/hr (GPU + provisioned disk) clears --max-price, ranked by the
    value metric (eff$/dlperf/hr, transfers folded in) — best bang-for-buck first."""
    offers = vast.search_offers(query=build_query(max_price, disk, min_vram), order=ORDER, limit=limit)
    qualifying = [o for o in offers if effective_dph(o, disk) <= max_price]
    qualifying.sort(
        key=lambda o: value_metric(o, disk=disk, data_gb=data_gb, upload_gb=upload_gb, run_hours=run_hours)
    )
    return qualifying


def print_offers(offers: list[dict], *, disk: int, data_gb: float, upload_gb: float, run_hours: float) -> None:
    if not offers:
        logger.warning("No offers clear --max-price once disk storage is counted. Raise it or shrink --disk.")
        return
    logger.info(
        f"{len(offers)} offer(s) within effective $/hr cap, best value first (disk={disk}GB, {run_hours:.0f}h run):"
    )
    header = (
        f"{'id':>10}  {'gpu':16} {'gpu$/hr':>8} {'disk$/hr':>8} {'eff$/hr':>8} "
        f"{'dl$':>6} {'ul$':>6} {'dlperf':>7} {'$/dlp/hr':>9} {'down':>9} {'rel':>5}"
    )
    print(header)
    print("-" * len(header))
    for o in offers:
        print(
            f"{o['id']:>10}  {o['gpu_name']:16} {o['dph_total']:8.3f} {storage_dph(o, disk):8.3f} "
            f"{effective_dph(o, disk):8.3f} {download_cost(o, data_gb):6.2f} {upload_cost(o, upload_gb):6.2f} "
            f"{o['dlperf']:7.1f} {value_metric(o, disk=disk, data_gb=data_gb, upload_gb=upload_gb, run_hours=run_hours):9.4f} "
            f"{o['inet_down']:7.0f}mbps {o['reliability']:5.3f}"
        )


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _account_env_keys(vast: VastAI) -> set[str]:
    """Names of the env-vars configured on the vast account (values not needed)."""
    d = vast.show_env_vars()
    if isinstance(d, dict):
        rows = d.get("results") or d.get("env_vars")
        if rows is not None:
            return {r.get("key") or r.get("name") for r in rows if isinstance(r, dict)}
        return set(d.keys())  # flat {name: value}
    if isinstance(d, list):
        return {r.get("key") or r.get("name") for r in d if isinstance(r, dict)}
    return set()


def preflight(vast: VastAI) -> tuple[str, str | None]:
    """Ensure the run is reproducible and credentialed before spending money.

    Returns (sha, github_token_or_none). Exits with a clear message on a dirty tree,
    an unpushed SHA we can't push, or missing account secrets. Secrets come from vast
    account env-vars (not the host, not `-e`); the GitHub token is optional (only used
    to pull a private ghcr image).
    """
    if _git("status", "--porcelain"):
        raise SystemExit("working tree is dirty — commit before launching (the box runs the pushed SHA).")
    sha = _git("rev-parse", "HEAD")
    if not _git("branch", "-r", "--contains", sha):
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        logger.info(f"{sha[:10]} not on origin; pushing {branch}")
        subprocess.run(["git", "push", "origin", branch], check=True)

    missing = [v for v in REQUIRED_ACCOUNT_VARS if v not in _account_env_keys(vast)]
    if missing:
        raise SystemExit(
            f"vast account env-vars missing {missing}. Add them (Console → Account → Environment "
            "Vars, or `vastai create env-var <name> <value>`) so they inject into the box without "
            "leaking into extra_env."
        )
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return sha, token


def queue(
    vast: VastAI,
    *,
    max_price: float,
    limit: int,
    disk: int,
    min_vram: int,
    data_gb: float,
    upload_gb: float,
    run_hours: float,
    poll_interval_s: int,
) -> list[dict]:
    """Poll the market until at least one offer clears the bar; return them ranked."""
    while True:
        offers = search(
            vast,
            max_price=max_price,
            limit=limit,
            disk=disk,
            min_vram=min_vram,
            data_gb=data_gb,
            upload_gb=upload_gb,
            run_hours=run_hours,
        )
        if offers:
            return offers
        logger.info(f"no offer <= ${max_price:.2f}/hr clears the bar; retrying in {poll_interval_s}s")
        time.sleep(poll_interval_s)


def _instance_env(*, sha: str, train_cmd: str) -> dict[str, str]:
    # Only non-secret per-run vars go through `-e` (these are visible in extra_env).
    # Secrets come from the vast account env-vars; see REQUIRED_ACCOUNT_VARS.
    return {
        "HAL_GIT_SHA": sha,
        "HAL_TRAIN_CMD_B64": base64.b64encode(train_cmd.encode()).decode(),
    }


def onstart_cmd() -> str:
    """Current on-start.sh contents, run under bash regardless of the shell vast
    invokes onstart with. Passed inline (not baked) so edits ship without a rebuild."""
    return f"bash -c {shlex.quote(ONSTART_PATH.read_text())}"


def launch(
    vast: VastAI,
    offer: dict,
    *,
    image: str,
    disk: int,
    env: dict[str, str],
    token: str | None,
    timeout_s: int,
    keep_alive: bool,
) -> int | None:
    """Rent the offer and poll to `running`, returning its id. On failure-to-launch
    (stuck/dead before `running`), destroy to avoid a leaked billing instance and
    return ``None`` so the caller can fail over to the next offer — unless
    ``keep_alive``, in which case leave the box up (booting or for inspection) and
    return its id. The ~9 GB image can take a while to pull+extract on a slow box,
    hence a generous timeout."""
    login = {"login": f"-u {GHCR_USER} -p {token} ghcr.io"} if token else {}  # ghcr auth only if image is private
    inst = vast.create_instance(
        id=offer["id"],
        image=image,
        disk=disk,
        env=env,
        onstart_cmd=onstart_cmd(),
        runtype="ssh_proxy",  # proxy SSH; avoids needing direct_port_count in the bar
        **login,
    )
    iid = inst["new_contract"]
    logger.info(f"created instance {iid} on offer {offer['id']} ({offer['gpu_name']}); polling to running")

    def give_up(why: str) -> int | None:
        if keep_alive:
            logger.warning(
                f"{why}; left up (--keep-alive). Monitor: vastai logs {iid} ; destroy: vastai destroy instance {iid}"
            )
            return iid
        logger.error(f"{why}; tearing down instance {iid}, failing over to next offer")
        vast.destroy_instance(id=iid)
        return None

    deadline = time.time() + timeout_s
    terminal_reads = 0  # consecutive polls showing a terminal status
    while True:
        try:
            status = vast.show_instance(id=iid).get("actual_status")
        except Exception as e:  # a transient API blip must not crash the poll
            logger.warning(f"poll: show_instance({iid}) failed ({e!r}); retrying")
            status = None
        if status == "running":
            return iid
        # Never tear down on a *single* bad read: vast reports "unknown" transiently
        # during state transitions, and an API hiccup yields None — either alone would
        # otherwise destroy a box that's merely still loading. Require a terminal status
        # to persist across two polls (a real host death holds; a blip clears).
        if status in {"exited", "offline", "unknown"}:
            terminal_reads += 1
            if terminal_reads >= 2:
                return give_up(f"instance {iid} died before running ({status})")
        else:
            terminal_reads = 0
        if time.time() > deadline:
            return give_up(f"instance {iid} stuck in {status!r} after {timeout_s}s")
        time.sleep(10)


@dataclass(frozen=True)
class Args:
    cmd: tyro.conf.Positional[list[str]] = field(default_factory=list)
    """Training command to run on the box, after `--` (e.g. `-- uv run experiments/001_...py`). Empty ⇒ search-only."""
    max_price: float = 1.10
    """Hard *effective* $/hr ceiling — GPU base PLUS the provisioned --disk's storage cost
    (vast bills disk separately; it's not in the search-time dph_total and can exceed the
    GPU). The impatience knob: lower waits longer for a cheaper box."""
    image: str = "ghcr.io/ericyuegu/hal:cuda13"
    """Image vast pulls (public ghcr; GITHUB_TOKEN only if you make it private)."""
    disk: int = 500
    """Container disk in GB (fixed at create; dies with the instance). Sized to hold the
    whole prod MDS on disk (~380 GB decompressed) plus the image + ISO, so shards cache
    once with no eviction/re-download churn across epochs; cache_limit (cfg) caps just
    under as a disk-full guard. Offers are filtered to disk_space >= this."""
    min_vram: int = 0
    """Minimum GPU VRAM in GB (vast `gpu_ram`); 0 = no floor. Memory-heavy runs need this
    so the $/perf ranking doesn't land them on a small card that OOMs (the launcher picks
    best DLPerf/$ first, which is usually a 12-16 GB card)."""
    limit: int = 10
    """How many offers to fetch/print."""
    poll_interval_s: int = 30
    """Seconds between market polls while queueing."""
    timeout_s: int = 1200
    """How long to wait for `running` — the ~9 GB image can pull slowly on a cheap box."""
    dry_run: bool = False
    """Run preflight + one search and print exactly what would be sent, without renting."""
    keep_alive: bool = False
    """Debug: leave the box up on crash/finish (no self stop/destroy) so you can SSH in."""
    data_gb: float = 40.0
    """Estimated GB the box downloads once at startup (the MDS dataset; the ~1.4 GB ISO is
    added on top). Priced at the offer's $/GB ingress and amortized into the ranking metric.
    Default ≈ the prod MDS; set lower for the dev/smaller sets."""
    upload_gb: float = 1.0
    """Estimated GB the box uploads over the run (checkpoints streamed to R2). Priced at the
    offer's $/GB egress (inet_up_cost) and amortized into the ranking metric."""
    run_hours: float = 10.0
    """Expected run length, used only to amortize the one-time download+upload costs into the
    per-hour ranking metric (eff$/dlperf/hr = (effective $/hr + one-time$/run_hours) / dlperf).
    A shorter run makes the transfer tax weigh more. Does not gate anything."""


def main(args: Args) -> None:
    vast = VastAI()

    search_kw = dict(
        max_price=args.max_price,
        limit=args.limit,
        disk=args.disk,
        min_vram=args.min_vram,
        data_gb=args.data_gb,
        upload_gb=args.upload_gb,
        run_hours=args.run_hours,
    )
    print_kw = dict(disk=args.disk, data_gb=args.data_gb, upload_gb=args.upload_gb, run_hours=args.run_hours)

    if not args.cmd:
        offers = search(vast, **search_kw)
        print_offers(offers, **print_kw)
        logger.info("search-only (pass a training command after `--` to launch). Nothing rented.")
        return

    sha, token = preflight(vast)
    train_cmd = shlex.join(args.cmd)
    env = _instance_env(sha=sha, train_cmd=train_cmd)
    if args.keep_alive:
        env["HAL_KEEP_ALIVE"] = "1"

    if args.dry_run:
        offers = search(vast, **search_kw)
        print_offers(offers, **print_kw)
        login = f"'-u {GHCR_USER} -p *** ghcr.io'" if token else "none (public image)"
        logger.info(f"[dry-run] image={args.image} disk={args.disk}GB runtype=ssh_proxy login={login}")
        logger.info(f"[dry-run] env (non-secret; secrets come from vast account env-vars)={env}")
        logger.info(f"[dry-run] onstart=<inline {ONSTART_PATH.name}, {len(ONSTART_PATH.read_text())} bytes>")
        logger.info(f"[dry-run] HAL_GIT_SHA={sha}")
        logger.info(f"[dry-run] train cmd: {train_cmd}")
        return

    offers = queue(vast, **search_kw, poll_interval_s=args.poll_interval_s)
    print_offers(offers, **print_kw)
    # Try offers best-first; a stuck/dead box (e.g. a host that can't provision the
    # disk or pull the image in time) fails over to the next rather than killing the run.
    for offer in offers:
        iid = launch(
            vast,
            offer,
            image=args.image,
            disk=args.disk,
            env=env,
            token=token,
            timeout_s=args.timeout_s,
            keep_alive=args.keep_alive,
        )
        if iid is not None:
            break
    else:
        raise SystemExit("every ranked offer failed to launch — rerun (the market may have healthier hosts).")

    eff = effective_dph(offer, args.disk)
    logger.success(
        f"instance {iid} ({offer['gpu_name']}, ${eff:.3f}/hr effective = "
        f"${offer['dph_total']:.3f} GPU + ${storage_dph(offer, args.disk):.3f} disk[{args.disk}GB]) "
        f"on SHA {sha[:10]}"
    )
    dl = download_cost(offer, args.data_gb)
    ul = upload_cost(offer, args.upload_gb)
    logger.info(
        f"one-time transfer ≈ ${dl + ul:.2f} (↓${dl:.2f} {args.data_gb:.0f}GB MDS+ISO, "
        f"↑${ul:.2f} {args.upload_gb:.0f}GB ckpts); amortized over {args.run_hours:.0f}h in the ranking"
    )
    logger.info(f"booting: clone -> uv sync -> fetch -> train. Watch W&B + `vastai logs {iid}` for progress.")
    logger.info(f"peek:  {vast.ssh_url(id=iid) or f'vastai ssh-url {iid}'}  (tail /opt/hal/train.log)")
    teardown = (
        "kept up regardless (--keep-alive); destroy manually"
        if args.keep_alive
        else "self-destructs on success / self-stops on failure"
    )
    logger.info(f"teardown: box {teardown}.")


if __name__ == "__main__":
    main(tyro.cli(Args))
