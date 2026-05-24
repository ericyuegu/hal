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
    return out
