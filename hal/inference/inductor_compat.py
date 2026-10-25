"""Torch 2.11.0 / Triton 3.6.0 reduction read-back compatibility.

Inductor can store a reduction intermediate and reload it with a different
thread layout in the same kernel. Triton requires an explicit block barrier.
Without it, the o65 action head can return unnormalized log probabilities.
Remove this adapter when an upstream generator passes the retained cold/warm
cache regression and the captured B200 actor case without the patch.
"""

import importlib.metadata

import sympy
import torch
from torch._inductor.codegen.common import CSEVariable
from torch._inductor.codegen.triton import TritonKernel
from torch._inductor.codegen.triton import TritonScheduling
from torch.compiler import config as compiler_config

CACHE_KEY_TAG = "hal-inductor-reduction-barrier-v1"


class _SynchronizedTritonKernel(TritonKernel):
    def load(self, name: str, index: sympy.Expr) -> CSEVariable:
        if name in self.cse.invalidated_stores:
            # A prior reduction loop emitted the store. Its read-back may use
            # values owned by other threads: triton-lang/triton#10924.
            self.body.writeline("tl.debug_barrier()")
        return super().load(name, index)


def install_reduction_barrier() -> None:
    """Install once, before CUDA compilation or compiler-artifact loading."""
    if str(torch.__version__).split("+", 1)[0] != "2.11.0" or importlib.metadata.version("triton") != "3.6.0":
        raise RuntimeError("reduction barrier requires tested Torch 2.11.0 / Triton 3.6.0")
    if TritonScheduling.kernel_type is _SynchronizedTritonKernel:
        if compiler_config.cache_key_tag != CACHE_KEY_TAG:
            raise RuntimeError("reduction barrier compiler cache tag changed after installation")
        return
    if TritonScheduling.kernel_type is not TritonKernel or compiler_config.cache_key_tag not in ("", CACHE_KEY_TAG):
        raise RuntimeError("another Triton generator or compiler cache tag is installed")
    # The generator hook is global in Torch 2.11. The tag also invalidates its
    # local FX/AOT caches, whose keys do not otherwise include this subclass.
    compiler_config.cache_key_tag = CACHE_KEY_TAG
    TritonScheduling.kernel_type = _SynchronizedTritonKernel
