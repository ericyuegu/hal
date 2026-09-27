import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

root = Path.cwd()
if (root / 'experiments/059_muon_action_sequence.py').exists():
    test_path = root / 'tests/experiments/test_059_muon_action_sequence.py'
else:
    test_path = root / 'tests/experiments/test_059_muon_history_decoder.py'
spec = importlib.util.spec_from_file_location('local_contract_test', test_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from hal.sim.rollout import ObservationRow
from hal.sim.vec import Slot
exp = module.exp
torch.manual_seed(19)
cfg = module._tiny_cfg(batch_size=1)
model = (exp.make_model(cfg) if hasattr(exp, "make_model") else exp.GPT(cfg)).eval()
flat, stats = module._live_inputs()
policy = exp.make_policy(model, stats, cfg, decode_seed=3, device='cpu', delay_frames=2, replan_interval_frames=2)
slot = Slot(0, 2)
neutral = exp.NEUTRAL_ACTION.copy()
outputs = []
first = policy.plan_rows({slot: [ObservationRow(10, flat, neutral, reset=True)]})[slot]
outputs.append(first.copy())
prior = first
for step in range(1, 26):
    rows = [ObservationRow(10 + (step - 1) * 2 + i, flat, prior[i - 1].copy()) for i in (1, 2)]
    plan = policy.plan_rows({slot: rows})[slot]
    outputs.append(plan.copy())
    prior = plan
payload = {
    'source': root.name,
    'sha256': hashlib.sha256(np.stack(outputs).astype(np.float32).tobytes()).hexdigest(),
    'plans': np.stack(outputs).astype(np.float32).tolist(),
}
print(json.dumps(payload))
