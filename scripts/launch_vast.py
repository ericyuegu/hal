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


def build_query(max_price: float, disk: int) -> str:
    return " ".join((*FILTERS, f"disk_space>={disk}", f"dph_total<{max_price}"))


def search(vast: VastAI, *, max_price: float, limit: int, disk: int) -> list[dict]:
    return vast.search_offers(query=build_query(max_price, disk), order=ORDER, limit=limit)


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


def queue(vast: VastAI, *, max_price: float, limit: int, disk: int, poll_interval_s: int) -> list[dict]:
    """Poll the market until at least one offer clears the bar; return them ranked."""
    while True:
        offers = search(vast, max_price=max_price, limit=limit, disk=disk)
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
) -> int:
    """Rent the offer and poll to `running`. On failure-to-launch (stuck/dead before
    `running`), destroy to avoid a leaked billing instance — unless ``keep_alive``,
    in which case leave it (booting or for inspection) and return its id. The ~9 GB
    image can take a while to pull+extract on a slow box, hence a generous timeout."""
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

    def give_up(why: str) -> int:
        if keep_alive:
            logger.warning(
                f"{why}; left up (--keep-alive). Monitor: vastai logs {iid} ; destroy: vastai destroy instance {iid}"
            )
            return iid
        logger.error(f"{why}; tearing down instance {iid}")
        vast.destroy_instance(id=iid)
        raise SystemExit(f"launch failed: {why}")

    deadline = time.time() + timeout_s
    while True:
        status = vast.show_instance(id=iid).get("actual_status")
        if status == "running":
            return iid
        if status in {"exited", "offline", "unknown"}:
            return give_up(f"instance {iid} died before running ({status})")
        if time.time() > deadline:
            return give_up(f"instance {iid} stuck in {status!r} after {timeout_s}s")
        time.sleep(10)


@dataclass(frozen=True)
class Args:
    cmd: tyro.conf.Positional[list[str]] = field(default_factory=list)
    """Training command to run on the box, after `--` (e.g. `-- uv run experiments/001_...py`). Empty ⇒ search-only."""
    max_price: float = 1.10
    """Hard $/hr ceiling — the impatience knob. Lower waits longer for a cheaper box."""
    image: str = "ghcr.io/ericyuegu/hal:cuda13"
    """Image vast pulls (public ghcr; GITHUB_TOKEN only if you make it private)."""
    disk: int = 200
    """Container disk in GB (fixed at create; dies with the instance). Sized for the image
    + ISO + the streamed shard cache: StreamingDataset keeps every downloaded shard (no
    cache_limit), and the prod MDS is ~20 GB. Offers are filtered to disk_space >= this."""
    limit: int = 10
    """How many offers to fetch/print."""
    poll_interval_s: int = 30
    """Seconds between market polls while queueing."""
    timeout_s: int = 600
    """How long to wait for `running` — the ~9 GB image can pull slowly on a cheap box."""
    dry_run: bool = False
    """Run preflight + one search and print exactly what would be sent, without renting."""
    keep_alive: bool = False
    """Debug: leave the box up on crash/finish (no self stop/destroy) so you can SSH in."""


def main(args: Args) -> None:
    vast = VastAI()

    if not args.cmd:
        print_offers(search(vast, max_price=args.max_price, limit=args.limit, disk=args.disk))
        logger.info("search-only (pass a training command after `--` to launch). Nothing rented.")
        return

    sha, token = preflight(vast)
    train_cmd = shlex.join(args.cmd)
    env = _instance_env(sha=sha, train_cmd=train_cmd)
    if args.keep_alive:
        env["HAL_KEEP_ALIVE"] = "1"

    if args.dry_run:
        offers = search(vast, max_price=args.max_price, limit=args.limit, disk=args.disk)
        print_offers(offers)
        login = f"'-u {GHCR_USER} -p *** ghcr.io'" if token else "none (public image)"
        logger.info(f"[dry-run] image={args.image} disk={args.disk}GB runtype=ssh_proxy login={login}")
        logger.info(f"[dry-run] env (non-secret; secrets come from vast account env-vars)={env}")
        logger.info(f"[dry-run] onstart=<inline {ONSTART_PATH.name}, {len(ONSTART_PATH.read_text())} bytes>")
        logger.info(f"[dry-run] HAL_GIT_SHA={sha}")
        logger.info(f"[dry-run] train cmd: {train_cmd}")
        return

    offers = queue(
        vast, max_price=args.max_price, limit=args.limit, disk=args.disk, poll_interval_s=args.poll_interval_s
    )
    print_offers(offers)
    offer = offers[0]
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

    logger.success(f"instance {iid} ({offer['gpu_name']}, ${offer['dph_total']:.3f}/hr) on SHA {sha[:10]}")
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
