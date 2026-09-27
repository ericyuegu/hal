import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--candidate", action="store_true")
args = parser.parse_args()
sys.path.insert(0, str(args.root))
name = "hal059_candidate" if args.candidate else "hal059_reference"
filename = "059_muon_action_sequence.py" if args.candidate else "059_muon_history_decoder.py"
path = args.root / "experiments" / filename
spec = importlib.util.spec_from_file_location(name, path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
cfg = module.proxy_config()
torch.manual_seed(29)
model = module.make_model(cfg) if args.candidate else module.GPT(cfg)
parameter_names = [(name, tuple(parameter.shape)) for name, parameter in model.named_parameters()]
parameters = hashlib.sha256()
for name, parameter in model.named_parameters():
    parameters.update(name.encode())
    parameters.update(parameter.detach().cpu().numpy().tobytes())
rng_after_model = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
torch.manual_seed(73)
context = module.synthetic_context(cfg, 1, torch.device("cpu"))
model.eval()
with torch.inference_mode():
    hidden = model.forward_dense(context.features, context.ctx_pad)
    observed = model.codec.quantize(module.stack_actions(context.features))[:, -1]
    returns = torch.tensor([20.0])
    present = torch.tensor([True])
    logits, actions = model.temporal.rollout_conditioned_logits(
        hidden, observed, returns, present, ctx_pad=context.ctx_pad
    )
result = {
    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    "parameter_names": parameter_names,
    "state_keys": list(model.state_dict()),
    "parameter_sha256": parameters.hexdigest(),
    "rng_after_model": rng_after_model,
}
args.output.with_suffix(".json").write_text(json.dumps(result))
torch.save({"hidden": hidden, "actions": actions, "logits": logits}, args.output.with_suffix(".pt"))
print(json.dumps({key: value for key, value in result.items() if key not in ("parameter_names", "state_keys")}))
