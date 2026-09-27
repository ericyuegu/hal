# O51 16-layer proxy preflight evidence

The fresh O51 v3 measurements ran on B200 workers from Git commit `2cdab7e`:

- compute: Modal app `ap-gMdNqDbedrG559Q8mIiFn8`;
- direct-source loader: Modal app `ap-EodIF3SYKxt1j3f04myOnN`.

The compute benchmark measured 20 compiled updates after three warmup updates.
The loader benchmark measured 50 batches after eight warmup batches. It read
the U0 prefix views from all 44 existing MDS sources. It produced 9,930 distinct
replays, no within-batch duplicates, and no reuse within the 16-batch cooldown.

The raw-byte ratio, exact-resume result, and host telemetry come from the
unchanged direct-source loader evidence in
[`preflight-base-b512-w16-p64.json`](preflight-base-b512-w16-p64.json). The GPU
memory fraction uses the fresh 35.193 GiB peak reservation and the worker's
191,503,138,816-byte device capacity.
