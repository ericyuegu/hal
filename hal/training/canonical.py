"""libmelee gamestate dict → flat MDS-shaped per-port columns.

The model's training-time batches are MDS rows: flat ``{p1_*, p2_*}`` dicts
of per-frame fields. At inference, ``Session.step()`` hands back a nested
canonical gamestate dict (from libmelee's post-frame). This helper bridges
the two so a model's ``ControllerSource`` can stitch its rolling-history
buffer using the same column schema it trained on.

Mirrors the per-frame extraction in :func:`hal.sim.trajectory.from_capture`;
both pull each field through ``wire.canonical_post_field`` over the shared
``wire.POST_FIELD_SUFFIXES``, so the two can never drift apart.
"""

from hal.wire import MASK_FLOAT
from hal.wire import POST_FIELD_SUFFIXES
from hal.wire import canonical_post_field


def flatten_canonical_frame(frame: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for libmelee_port, prefix in ((1, "p1"), (2, "p2")):
        pd = frame["ports"].get(libmelee_port)
        if pd is None:
            continue
        post = pd["leader"]["post"]
        for suffix in POST_FIELD_SUFFIXES:
            out[f"{prefix}_{suffix}"] = canonical_post_field(post, suffix)
        # Nana follower: gamestate only, mirroring extract._extract_nana. Real values for
        # Ice Climbers; MASK_FLOAT for everyone else (preprocess flags NaN as masked). Keeps
        # the closed-loop obs column set identical to the nana-carrying MDS the model trains on.
        follower = pd.get("follower")
        follower_post = follower["post"] if follower is not None else None
        for suffix in POST_FIELD_SUFFIXES:
            out[f"{prefix}_nana_{suffix}"] = (
                canonical_post_field(follower_post, suffix) if follower_post is not None else MASK_FLOAT
            )
    # Matchup conditioning (SCHEMA_VERSION 4): the driver injects per-match stage + per-port
    # character (constants, the libmelee Stage/Character values that match the training columns).
    # Absent unless drive_vec injected them, so non-conditioned experiments are unaffected.
    matchup = frame.get("_matchup")
    if matchup is not None:
        out["stage"] = matchup["stage"]
        for libmelee_port, prefix in ((1, "p1"), (2, "p2")):
            if libmelee_port in frame["ports"]:
                out[f"{prefix}_character"] = matchup["character"][libmelee_port]
    return out
