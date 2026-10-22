# O64 final evaluation protocol

Evaluate only the final update-256 student against its frozen O64 pi_ref.
Both players use D=18, R=2, u=0.95, lambda=1 and masked identity (player id 0).
The reference is the O64 step-0098304 checkpoint with SHA-256
`e6d6303fbf25993c768b1e86fecbb350433a2fb229235b556334e9eb6ae868cf`.

Use the unchanged [production-evaluation.json](production-evaluation.json):
32 weighted training matchup draws, six stages, seed 10017 and both port
orientations, for 384 MSL games. Each game begins at four stocks and zero
damage and ends at stockout or eight minutes. Keep paired seed/start confidence
intervals and the existing event metrics. Consume the held-out set before play
and preserve partial results after failure. Do not use these games to choose
capacity, a checkpoint or lambda.

The runner records this protocol and its grid hash before training. The exact
evaluation command is constructed by `scripts.o65_production.evaluation_command`.
