"""GPU evaluation of O65's existing JSON/BLAKE2b action counters."""

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import triton
import triton.language as tl
from torch import Tensor

if TYPE_CHECKING:
    from builtins import int as _Const

    type _Constants = tuple[int, ...]
    type _Schedule = tuple[tuple[int, ...], ...]
else:
    _Const = _Constants = _Schedule = tl.constexpr


@triton.jit
def _rotate(value, count: _Const):
    return (value >> count) | (value << (64 - count))


@triton.jit
def _mix(a, b, c, d, x, y):
    a = a + b + x
    d = _rotate(d ^ a, 32)
    c = c + d
    b = _rotate(b ^ c, 24)
    a = a + b + y
    d = _rotate(d ^ a, 16)
    c = c + d
    b = _rotate(b ^ c, 63)
    return a, b, c, d


@triton.jit
def _word(prefix, row, length, offset, factor, position, WIDTH: _Const):
    word = tl.load(prefix + row * (WIDTH // 8) + position // 8)
    digits = tl.where(offset >= 10, 2, 1)
    first = tl.where(digits == 2, offset // 10, offset) + 48
    word = word | tl.where(length // 8 == position // 8, first.to(tl.uint64) << (8 * (length % 8)), 0)
    word = word | tl.where(
        (digits == 2) & ((length + 1) // 8 == position // 8),
        (offset % 10 + 48).to(tl.uint64) << (8 * ((length + 1) % 8)),
        0,
    )
    for extra in tl.static_range(3):
        at = length + digits + extra
        value = tl.where(extra == 0, 44, tl.where(extra == 1, factor + 48, 93)).to(tl.uint64)
        word = word | tl.where(at // 8 == position // 8, value << (8 * (at % 8)), 0)
    return word


@triton.jit(do_not_specialize=["BATCH"])
def _uniforms(prefix, metadata, output, BATCH, WIDTH: _Const, BLOCKS: _Const, TILE: _Const):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    live = index < 16 * BATCH
    row = index % BATCH
    group = index // BATCH % 4
    factor = tl.where(group == 0, 3, tl.where(group == 1, 1, tl.where(group == 2, 0, 2)))
    length = tl.load(metadata + 2 * row)
    offset = tl.load(metadata + 2 * row + 1) + index // (4 * BATCH) + 1
    message_length = length + tl.where(offset >= 10, 2, 1) + 3
    iv: _Constants = (
        0x6A09E667F3BCC908,
        0xBB67AE8584CAA73B,
        0x3C6EF372FE94F82B,
        0xA54FF53A5F1D36F1,
        0x510E527FADE682D1,
        0x9B05688C2B3E6C1F,
        0x1F83D9ABFB41BD6B,
        0x5BE0CD19137E2179,
    )
    sigma: _Schedule = (
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        (14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3),
        (11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4),
        (7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8),
        (9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13),
        (2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9),
        (12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11),
        (13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10),
        (6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5),
        (10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        (14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3),
    )
    # BLAKE2b digest_size=8, fanout=depth=1, person=b"hal-o65-rng-v1".
    h0 = tl.full((TILE,), iv[0] ^ 0x01010008, tl.uint64)
    h1 = tl.full((TILE,), iv[1], tl.uint64)
    h2 = tl.full((TILE,), iv[2], tl.uint64)
    h3 = tl.full((TILE,), iv[3], tl.uint64)
    h4 = tl.full((TILE,), iv[4], tl.uint64)
    h5 = tl.full((TILE,), iv[5], tl.uint64)
    h6 = tl.full((TILE,), iv[6] ^ 0x2D35366F2D6C6168, tl.uint64)
    h7 = tl.full((TILE,), iv[7] ^ 0x000031762D676E72, tl.uint64)
    for block in range(BLOCKS):
        words = ()
        for word in tl.static_range(16):
            words += (_word(prefix, row, length, offset, factor, 128 * block + 8 * word, WIDTH),)
        v0, v1, v2, v3, v4, v5, v6, v7 = h0, h1, h2, h3, h4, h5, h6, h7
        v8 = tl.full((TILE,), iv[0], tl.uint64)
        v9 = tl.full((TILE,), iv[1], tl.uint64)
        v10 = tl.full((TILE,), iv[2], tl.uint64)
        v11 = tl.full((TILE,), iv[3], tl.uint64)
        v12 = tl.full((TILE,), iv[4], tl.uint64) ^ tl.minimum(message_length, 128 * (block + 1)).to(tl.uint64)
        v13 = tl.full((TILE,), iv[5], tl.uint64)
        v14 = tl.full((TILE,), iv[6], tl.uint64) ^ tl.where(
            message_length <= 128 * (block + 1), 0xFFFFFFFFFFFFFFFF, 0
        ).to(tl.uint64)
        v15 = tl.full((TILE,), iv[7], tl.uint64)
        for round in tl.static_range(12):
            v0, v4, v8, v12 = _mix(v0, v4, v8, v12, words[sigma[round][0]], words[sigma[round][1]])
            v1, v5, v9, v13 = _mix(v1, v5, v9, v13, words[sigma[round][2]], words[sigma[round][3]])
            v2, v6, v10, v14 = _mix(v2, v6, v10, v14, words[sigma[round][4]], words[sigma[round][5]])
            v3, v7, v11, v15 = _mix(v3, v7, v11, v15, words[sigma[round][6]], words[sigma[round][7]])
            v0, v5, v10, v15 = _mix(v0, v5, v10, v15, words[sigma[round][8]], words[sigma[round][9]])
            v1, v6, v11, v12 = _mix(v1, v6, v11, v12, words[sigma[round][10]], words[sigma[round][11]])
            v2, v7, v8, v13 = _mix(v2, v7, v8, v13, words[sigma[round][12]], words[sigma[round][13]])
            v3, v4, v9, v14 = _mix(v3, v4, v9, v14, words[sigma[round][14]], words[sigma[round][15]])
        active = 128 * block < message_length
        h0 = tl.where(active, h0 ^ v0 ^ v8, h0)
        h1 = tl.where(active, h1 ^ v1 ^ v9, h1)
        h2 = tl.where(active, h2 ^ v2 ^ v10, h2)
        h3 = tl.where(active, h3 ^ v3 ^ v11, h3)
        h4 = tl.where(active, h4 ^ v4 ^ v12, h4)
        h5 = tl.where(active, h5 ^ v5 ^ v13, h5)
        h6 = tl.where(active, h6 ^ v6 ^ v14, h6)
        h7 = tl.where(active, h7 ^ v7 ^ v15, h7)
    tl.store(output + index, (h0 >> 11).to(tl.float64) * 2.0**-53, live)


def action_uniforms(seed: int, counters: Sequence[tuple[int, int, int, int]], device: torch.device) -> Tensor:
    """Return [4 offsets, 4 codec groups, B] with the scalar counter's exact bits.

    Counters contain (logical stream, episode, frame, delay). The decimal JSON
    prefix is serialized once per stream; the GPU expands offset and factor.
    """
    if device.type != "cuda" or not counters:
        raise ValueError("batched action counters require CUDA and a nonempty batch")
    if type(seed) is not int or any(type(value) is not int for row in counters for value in row):
        raise ValueError("action counters must contain integers")
    if any(not 0 <= row[3] <= 95 for row in counters):
        raise ValueError("action counter delays must be in [0, 95]")
    prefixes = [
        (json.dumps((seed, "action", stream, episode, frame), separators=(",", ":"))[:-1] + ",").encode()
        for stream, episode, frame, _ in counters
    ]
    longest = max(map(len, prefixes))
    width = triton.cdiv(longest + 5, 128) * 128
    packed = np.zeros((len(counters), width), dtype=np.uint8)
    for row, prefix in enumerate(prefixes):
        packed[row, : len(prefix)] = np.frombuffer(prefix, dtype=np.uint8)
    metadata = torch.tensor(
        [(len(prefix), counter[3]) for prefix, counter in zip(prefixes, counters, strict=True)]
    ).to(device, non_blocking=True)
    output = torch.empty(4, 4, len(counters), dtype=torch.float64, device=device)
    _uniforms[(triton.cdiv(output.numel(), 128),)](
        torch.as_tensor(packed.view(np.uint64)).to(device, non_blocking=True),
        metadata,
        output,
        len(counters),
        width,
        triton.cdiv(longest + 5, 128),
        128,
    )
    return output
