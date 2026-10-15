# Half-Muon attention-control inventory

The shortest valid Muon control is W&B run `7dgv85x8` at boundary update
57,344. It is not an independent run from scratch. It forks `p1fyyp1z` at
update 24,576, restores its training state, and halves only the Muon learning
rate schedule. No better independent Muon baseline exists in the bounded exact
architecture, schema-7, full-44-source-v8 shortlist.

## Artifact ready for analysis

- model-only export:
  `/home/ericgu/src/hal/runs/mechinterp-20260911-scratch/7dgv85x8-step-0057344.model.pt`;
- export SHA-256:
  `2ad0c0f8015c75235def1e8d591a7350145e64a389fca5af63b39b53e6516be2`;
- adjacent metadata SHA-256:
  `9d27c8d84778696423fb6617350e5136886d1e6b01418596117dd5871922b653`;
- preserved source checkpoint SHA-256:
  `89c941d64f6407eed34bc6b9ad3ccc1466aace67d48f0ef45978bcb7575a4b08`;
- R2 ETag: `"9f4346d792838a73cb5fe6303758dfc5-213"`.

The source was loaded only with `weights_only=True`. The pickle scan reported
two project globals: `PhysicalRow` and `RingSlotDescriptor`. Both are frozen,
slotted locator dataclasses. Their loader-file SHA-256 is `c098ac84…` in both
the logged v11 source and the current worktree. The exported payload has the
exact ten fields required by the public O50 loader. It contains 168 tensors,
with the same keys, shapes, and dtypes as the parent update-57,344 export. The
embedded player vocabulary, 44 source manifests, source selection, schema, and
v8 loader protocol all match.

The canonical export SHA changed before analysis because I replaced an inline
serialization with the preserved safe-exporter output. The source checkpoint
did not change. The current export is now frozen. It is bit-exact to the raw
source for all 168 model tensors and has identical config and data identity.
A surviving earlier inline archive is preserved in the disk-backed scratch
directory with SHA-256 `ab4e9277…`; it is also bit-exact to the current export.
The initially predeclared `6cbd8755…` archive was overwritten before it was
backed up and is not available. No attention measurement had started. The JSON
inventory records this provenance error, the exact parity checks, and both
surviving archive identities.

## Run identity and limits

The checkpoint is O50 v5, d1024/L16/h16/context256, with Muon master learning
rate 0.014 and multiplier 0.5. Its parent checkpoint identity is
`p1fyyp1z` update 24,576, SHA-256 `6d8855b5…`.

Fresh `scan_history` families use `global_step`, not W&B `_step`. The 1,867
training rows cover 24,600 through 71,225. All requested values are finite:
gradient norm is 0.2483–0.3380, the 25-update clip fraction is 0–0.04, and
training loss is 1.3291–1.4221. At 57,344, validation NLL is 1.3879 and the
96-boot gameplay result is +0.1645 stocks/min with zero crashes.

This supports “no observed numerical divergence through 71,225.” It does not
establish permanent or gameplay health. The last gameplay evaluation is at
65,536. W&B labels the process `crashed`, the configured target is 131,072,
and no `final.pt` exists.

## Code and treatment differences

The branch logged source artifacts v10 and v11. The first attempt saved the
unchanged fork boundary, then failed before a branch optimizer update while
stacking diagnostic scalars split across CPU and CUDA. V11 changes only that
diagnostic transfer: each scalar moves to CPU before `torch.stack`. The W&B
training history begins at 24,600 under v11.

Compared with the parent update-57,344 raw config, the branch adds its
continuation diagnostics, half-Muon multiplier, and four parent-lineage fields;
its experiment ID changes from v4 to v5. No architecture, objective, optimizer
master setting, representation, vocabulary, or data field differs.

The exact inventory, R2 checkpoint list, raw-config differences, retrieval
command, hashes, and known lineage gaps are in
`artifacts/attention_muon_control/inventory.json`.
