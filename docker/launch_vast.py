"""Queue for a vast.ai GPU under a price ceiling and run a training command on it.

The instance is fire-and-forget: this launcher pushes the current git SHA, waits
for an offer that clears the hardware bar, rents it, and injects the SHA +
credentials + the (base64'd) training command. The box then clones that SHA,
trains, and tears *itself* down — destroy on success (checkpoints are already in
R2, logs in W&B), stop on failure (for inspection). See docker/on-start.sh.

    python docker/launch_vast.py                         # search-only: print offers, rent nothing
    python docker/launch_vast.py --dry-run -- uv run experiments/001_flow_matching_baseline.py
    python docker/launch_vast.py --max-price 0.80 -- uv run experiments/001_flow_matching_baseline.py --cfg.max-steps 100000

Requires host env: AWS_* (+ AWS_BUCKET=hal), WANDB_API_KEY, and GITHUB_TOKEN
(scopes: repo + read:packages — used to pull the private ghcr image and clone
the private repo on the box).
"""

import base64
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from dataclasses import field

import tyro
from loguru import logger
from vastai import VastAI

GHCR_USER = "ericyuegu"

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
    "dlperf_usd>90",  # DLPerf per $/hr
    "reliability>0.96",  # > 96%
    "inet_down>300",  # > 300 Mbps down
    "inet_down_cost<=0.01",  # <= $10/TB down
    "rentable=true",
)
ORDER = "dlperf_usd-"  # sort by DLPerf/$/hr, descending

# R2 + W&B creds forwarded to the box. GITHUB_TOKEN is handled separately (it also
# logs in to ghcr) and the per-run vars (SHA, command) are added at create time.
CRED_VARS = ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_BUCKET", "WANDB_API_KEY")


def build_query(max_price: float) -> str:
    return " ".join((*FILTERS, f"dph_total<{max_price}"))


def search(vast: VastAI, *, max_price: float, limit: int) -> list[dict]:
    return vast.search_offers(query=build_query(max_price), order=ORDER, limit=limit)


def print_offers(offers: list[dict]) -> None:
    if not offers:
        logger.warning("No offers match the bar. Raise --max-price or relax a FILTERS constraint.")
        return
    logger.info(f"{len(offers)} offer(s), best DLPerf/$/hr first:")
    header = f"{'id':>10}  {'gpu':16} {'dlperf/$':>9} {'$/hr':>7} {'tflops':>7} {'down':>9} {'$/GB':>8} {'rel':>5}"
    print(header)
    print("-" * len(header))
    for o in offers:
        print(
            f"{o['id']:>10}  {o['gpu_name']:16} {o['dlperf_per_dphtotal']:9.0f} "
            f"{o['dph_total']:7.3f} {o['total_flops']:7.0f} {o['inet_down']:7.0f}mbps "
            f"{o['inet_down_cost']:8.4f} {o['reliability']:5.3f}"
        )


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def preflight() -> tuple[str, str, dict[str, str]]:
    """Ensure the run is reproducible and credentialed before spending money.

    Returns (sha, github_token, cred_env). Exits with a clear message on a dirty
    tree, an unpushed SHA we can't push, or any missing credential.
    """
    if _git("status", "--porcelain"):
        raise SystemExit("working tree is dirty — commit before launching (the box runs the pushed SHA).")
    sha = _git("rev-parse", "HEAD")
    if not _git("branch", "-r", "--contains", sha):
        branch = _git("rev-parse", "--abbrev-ref", "HEAD")
        logger.info(f"{sha[:10]} not on origin; pushing {branch}")
        subprocess.run(["git", "push", "origin", branch], check=True)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    missing = [v for v in CRED_VARS if not os.environ.get(v)]
    if not token:
        missing.append("GITHUB_TOKEN (or GH_TOKEN)")
    if missing:
        raise SystemExit(f"missing required host env vars: {missing}")
    return sha, token, {v: os.environ[v] for v in CRED_VARS}


def queue(vast: VastAI, *, max_price: float, limit: int, poll_interval_s: int) -> list[dict]:
    """Poll the market until at least one offer clears the bar; return them ranked."""
    while True:
        offers = search(vast, max_price=max_price, limit=limit)
        if offers:
            return offers
        logger.info(f"no offer <= ${max_price:.2f}/hr clears the bar; retrying in {poll_interval_s}s")
        time.sleep(poll_interval_s)


def _instance_env(creds: dict[str, str], *, token: str, sha: str, train_cmd: str) -> dict[str, str]:
    return {
        **creds,
        "GITHUB_TOKEN": token,
        "HAL_GIT_SHA": sha,
        "HAL_TRAIN_CMD_B64": base64.b64encode(train_cmd.encode()).decode(),
    }


def launch(
    vast: VastAI,
    offer: dict,
    *,
    image: str,
    disk: int,
    env: dict[str, str],
    token: str,
    timeout_s: int,
) -> int:
    """Rent the offer, poll to `running`, destroy on any failure-to-launch."""
    inst = vast.create_instance(
        id=offer["id"],
        image=image,
        disk=disk,
        env=env,
        login=f"-u {GHCR_USER} -p {token} ghcr.io",  # pull the private ghcr image
        onstart_cmd="bash /usr/local/bin/on-start.sh",
        runtype="ssh_proxy",  # proxy SSH; avoids needing direct_port_count in the bar
    )
    iid = inst["new_contract"]
    logger.info(f"created instance {iid} on offer {offer['id']} ({offer['gpu_name']}); polling to running")
    deadline = time.time() + timeout_s
    try:
        while True:
            status = vast.show_instance(id=iid).get("actual_status")
            if status == "running":
                return iid
            if status in {"exited", "offline", "unknown"}:
                raise RuntimeError(f"instance {iid} died before running: {status}")
            if time.time() > deadline:
                raise TimeoutError(f"instance {iid} stuck in {status!r} after {timeout_s}s")
            time.sleep(10)
    except BaseException:
        logger.error(f"tearing down instance {iid} after launch failure")
        vast.destroy_instance(id=iid)
        raise


@dataclass(frozen=True)
class Args:
    cmd: tyro.conf.Positional[list[str]] = field(default_factory=list)
    """Training command to run on the box, after `--` (e.g. `-- uv run experiments/001_...py`). Empty ⇒ search-only."""
    max_price: float = 1.10
    """Hard $/hr ceiling — the impatience knob. Lower waits longer for a cheaper box."""
    image: str = "ghcr.io/ericyuegu/hal:cuda13"
    """Image vast pulls (private ghcr; authed via GITHUB_TOKEN)."""
    disk: int = 100
    """Container disk in GB (fixed at create; dies with the instance)."""
    limit: int = 10
    """How many offers to fetch/print."""
    poll_interval_s: int = 30
    """Seconds between market polls while queueing."""
    timeout_s: int = 900
    """How long to wait for the instance to reach `running`."""
    dry_run: bool = False
    """Run preflight + one search and print exactly what would be sent, without renting."""


def main(args: Args) -> None:
    vast = VastAI()

    if not args.cmd:
        print_offers(search(vast, max_price=args.max_price, limit=args.limit))
        logger.info("search-only (pass a training command after `--` to launch). Nothing rented.")
        return

    sha, token, creds = preflight()
    train_cmd = shlex.join(args.cmd)
    env = _instance_env(creds, token=token, sha=sha, train_cmd=train_cmd)

    if args.dry_run:
        offers = search(vast, max_price=args.max_price, limit=args.limit)
        print_offers(offers)
        safe_env = {
            k: (v if k in {"HAL_GIT_SHA", "AWS_BUCKET", "AWS_ENDPOINT_URL"} else "***") for k, v in env.items()
        }
        logger.info(
            f"[dry-run] image={args.image} disk={args.disk}GB runtype=ssh_proxy login='-u {GHCR_USER} -p *** ghcr.io'"
        )
        logger.info(f"[dry-run] env={safe_env}")
        logger.info("[dry-run] onstart='bash /usr/local/bin/on-start.sh'")
        logger.info(f"[dry-run] HAL_GIT_SHA={sha}")
        logger.info(f"[dry-run] train cmd: {train_cmd}")
        return

    offers = queue(vast, max_price=args.max_price, limit=args.limit, poll_interval_s=args.poll_interval_s)
    print_offers(offers)
    offer = offers[0]
    iid = launch(vast, offer, image=args.image, disk=args.disk, env=env, token=token, timeout_s=args.timeout_s)

    logger.success(f"instance {iid} running ({offer['gpu_name']}, ${offer['dph_total']:.3f}/hr) on SHA {sha[:10]}")
    ssh = vast.ssh_url(id=iid)
    logger.info("booting: clone -> uv sync -> fetch -> train. Watch W&B for progress.")
    logger.info(f"peek:  {ssh or f'vastai ssh-url {iid}'}  (tail /opt/hal/train.log)")
    logger.info("the box self-destructs on success / self-stops on failure; no teardown needed from here.")


if __name__ == "__main__":
    main(tyro.cli(Args))
