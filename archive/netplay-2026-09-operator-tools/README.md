# September 2026 operator tools

Historical source from the ignored `runs/netplay/` directories, copied without
changes on October 3, 2026. `sha256.json` records the original file hashes.
No credential values, tokens, account exports, policy files or recordings are
included. Originals and campaign evidence remain in `runs/netplay/`.

These scripts target an earlier queue protocol. Do not use them with protocol 4.
The campaign must be ported and tested before it resumes. In particular, its
terminal-job and one-job-per-game assumptions do not match continuous sessions.
The old deployment helper only permitted roster changes; it is not a general
policy restore command.

Maintained replacements:

- `hal/netplay_service/credentials.py`: encrypted credential store and admin environment.
- `scripts/netplay_admin.py`: run the current admin CLI with that store.
- `deploy/netplay/README.md`: deployment and recovery procedures.

The archived chat tool documents the prior token refresh and EventSub flow.
It is research evidence, not a supported command. Do not put these files on
PYTHONPATH or import them into maintained code.
