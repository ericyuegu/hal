# Attention heads and token positions

Use the two existing Muon measurements at update 57,344 and the preselected
replay example zero. Do not sample another replay, change a model, or run a
new forward pass. The saved query is the most recent input token, position
255. Each input token represents one frame; key position zero is the oldest.

For each layer and head, find the key position with the largest saved
attention weight. Retain its weight. Compare these positions between heads
within the same layer and input. Count distinct positions and unordered
pairs of heads whose maximum occurs at the same position. Do not average
positions across replays. These counts are for one input, not population
estimates or measures of independent information.

Check every saved distribution for normalization and a unique maximum.
Reject ties instead of assigning an arbitrary token position. The preliminary
inspection found no tied maxima in the two saved arrays.

Show all 16 heads at once, with a layer selector. Display the half-rate
continuation first and the original Muon run second. Start at layer zero;
do not choose the initial layer for its appearance. Hover details must name
the head, key token position, distance from the query in frames, and actual
attention weight. Coincident key positions must remain visible as separate
heads. A maximum-weight marker does not mean all attention goes to that token.

Source artifacts are the frozen `attention_muon_control/attention_detail.json`
and `attention_example0.npz`. Their SHA-256 values are respectively
`ce30744586fcc6bd5c828dc13f9c1fa930b014465ec157ee5ab506646359f6e2` and
`1518b5581f7beda9a431d2f4c9c7e22c6712c62a350a8fdf135f37c7d8ec38be`.
Keep these and all earlier figures unchanged. Save the derived data,
source snapshot, and new figure in `artifacts/attention_head_positions/`.
