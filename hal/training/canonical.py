"""libmelee gamestate dict → flat MDS-shaped per-port columns.

The model's training-time batches are MDS rows: flat ``{p1_*, p2_*}`` dicts
of per-frame fields. At inference, ``Session.step()`` hands back a nested
canonical gamestate dict (from libmelee's post-frame). This helper bridges
the two so a model's ``ControllerSource`` can stitch its rolling-history
buffer using the same column schema it trained on.

Mirrors the per-frame extraction in :func:`hal.sim.trajectory.from_capture`
but emits one row's worth of fields rather than accumulating arrays.
"""


def flatten_canonical_frame(frame: dict) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for libmelee_port, prefix in ((1, "p1"), (2, "p2")):
        pd = frame["ports"].get(libmelee_port)
        if pd is None:
            continue
        post = pd["leader"]["post"]
        pos = post["position"]
        out[f"{prefix}_position_x"] = float(pos["x"])
        out[f"{prefix}_position_y"] = float(pos["y"])
        out[f"{prefix}_percent"] = float(post["percent"])
        out[f"{prefix}_shield"] = float(post["shield"])
        out[f"{prefix}_stock"] = int(post["stock"])
        out[f"{prefix}_direction"] = float(post["direction"])
        out[f"{prefix}_action"] = int(post["action"])
        # libmelee names it state_age; MDS calls the same field action_frame.
        out[f"{prefix}_action_frame"] = float(post.get("state_age") or 0.0)
        out[f"{prefix}_hitlag_left"] = float(post.get("hitlag_left") or 0.0)
        out[f"{prefix}_jumps_used"] = int(post.get("jumps_used") or 0)
        out[f"{prefix}_airborne"] = int(post.get("airborne") or 0)
        out[f"{prefix}_hurtbox_state"] = int(post.get("hurtbox_state") or 0)
    return out
