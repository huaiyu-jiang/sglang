# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# Adapted from flashinfer-ai/flashinfer v0.7.0 (commit 4d75a33f19aa),
# flashinfer/gemm/kernels/dense_bf16_gemm_warp_splitk.py.
#
# TEMPORARY COPY. FlashInfer ships this kernel (``run_warp_splitk_dense``, also
# reachable through ``flashinfer.gemm.bf16_gemm_sm100`` with the cute-dsl
# runners), but its ``validate_inputs`` only accepts a bf16 ``out``. Functional
# differences here: an FP32 output path (K3's merged MoE front projection keeps
# router-logit precision with it), K split over several activation tensors, N
# split into row regions with their own output tensor and dtype, SM90 support,
# and compiled variants persisted through the CuTe AOT cache. Delete this module
# and call FlashInfer's kernel once a FlashInfer release covers that contract.
"""BF16 GEMM with cp.async and four compute warps holding split-K partials.

Coverage: M in [1, 32]; K a multiple of 128 spanning at most 64 K tiles; N a
multiple of 16; BF16 activations and weights; BF16 or FP32 outputs; SM90 and
SM10x. ``run_warp_splitk_dense_multi_io`` reads K from up to four activation
tensors and writes N to up to four output tensors in one launch.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Iterable, Sequence

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import msgspec
import torch as _torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

from sglang.kernels.jit.cute_aot_cache import get_jit_cache

_MMA_SHAPE = (16, 8, 16)
_COMPUTE_WARPS = 4
_A_LOADER_WARPS = 2
_A_LOADER_THREADS = _A_LOADER_WARPS * 32
_MAX_M = 32
# Ring depth cap: cold-read bandwidth saturates around 12 stages (96 KB in
# flight), so deeper rings only enlarge the tactic space.
_MAX_STAGES = 16
# The consumer's K loop is fully unrolled; bound the tile count so JIT time and
# code size stay proportionate (64 tiles of 256 elements = K 16384).
_MAX_K_TILES = 64
_SUPPORTED_OUTPUT_TILES = (16, 32)
_SUPPORTED_TOKEN_TILES = (8, 16, 32)
_SUPPORTED_K_TILES = (128, 256)
_SUPPORTED_B_LOADER_WARPS = (1, 2)
# Arbitrary; each slot adds one kernel parameter, and unused slots are never read.
_MAX_K_SEGMENTS = 4
_MAX_OUTPUT_REGIONS = 4
# Per-SM shared memory is the per-CTA maximum plus the 1 KB the driver reserves
# for every resident CTA; residency estimates must include that reservation.
_CTA_RESERVED_SMEM = 1024
_MAILBOX_GROUP_VALUES = 32
_MAILBOX_PADDED_GROUP_VALUES = 34


# Pad each mailbox group by two FP32 words to reduce bank conflicts.
def _partial_offset(feature, token, token_tile: int):
    group_rows = _MAILBOX_GROUP_VALUES // token_tile
    return (
        (feature % group_rows) * token_tile
        + (feature // group_rows) * _MAILBOX_PADDED_GROUP_VALUES
        + token
    )


def _a_fragment_offset(lane, compute_warp, m_iter, phase, k_tile):
    """Element offset of one lane's A ``ldmatrix.x4`` address within a stage."""
    linear = lane % 16 + (lane // 16) * 128 + phase * 256
    row = linear % _MMA_SHAPE[0] + m_iter * _MMA_SHAPE[0]
    col = linear // _MMA_SHAPE[0] + compute_warp * (k_tile // _COMPUTE_WARPS)
    return row * k_tile + (col ^ ((row % 8) * 8))


def _b_pair_offset(lane, compute_warp, n_iter, pair, token_tile, k_tile):
    """Element offset of one lane's B ``ldmatrix.x4`` address within a stage."""
    b_atom = lane % 8 + ((lane % 32) // 8) * (token_tile * 8)
    linear = b_atom + pair * 2 * token_tile * 16 + n_iter * 8
    row = linear % token_tile
    col = linear // token_tile + compute_warp * (k_tile // _COMPUTE_WARPS)
    return row * k_tile + (col ^ ((row % 8) * 8))


@dsl_user_op
def _ldmatrix_x4(smem, offset, *, loc=None, ip=None):
    """``ldmatrix.x4`` of the 16-bit fragment ``offset`` elements into ``smem``."""
    return prims.ldmatrix(
        smem.iterator.raw_ptr() + offset,
        num=4,
        layout=prims.MMALayout.ROW,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _l2_policy_evict_first(*, loc=None, ip=None):
    """64-bit L2 evict_first policy (fraction 1.0).

    Must stay a runtime ``createpolicy``: with the equivalent constant, ptxas
    13.0 emits the invalid ``LDGSTS ... [R+UR], desc[UR1]`` encoding.
    """
    return cutlass.Int64(
        llvm.inline_asm(
            cutlass.Int64.mlir_type,
            [],
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cp_async_16_with_policy(dst, src, policy, *, loc=None, ip=None):
    """16-byte cp.async.cg global->shared carrying an L2 cache-policy hint."""
    llvm.inline_asm(
        None,
        [
            dst.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            src.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            policy.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2;",
        "r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _issue_weight_tile(
    smem,
    weight,
    tidx,
    output_tile_idx,
    k_tile,
    stage,
    policy,
    output_tile,
    k_tile_size,
    with_policy,
    *,
    loc=None,
    ip=None,
):
    """Issue one A-loader thread's cp.async copies for weight K tile ``k_tile``.

    Plain Python control flow only: this helper is not preprocessed by the DSL.
    ``with_policy`` selects the ``.L2::cache_hint`` form; the pre-barrier
    prologue tile must use the plain form because ptxas 13.0 mis-encodes the
    hinted copy in that position (``LDGSTS ... [R+UR], desc[UR<odd>]``).
    """
    copy_count = output_tile * k_tile_size // (8 * _A_LOADER_THREADS)
    for copy_idx in range(copy_count):
        linear = tidx * 8 + copy_idx * _A_LOADER_THREADS * 8
        row = linear // k_tile_size
        col = linear % k_tile_size
        dst_offset = (
            stage * output_tile * k_tile_size
            + row * k_tile_size
            + (col ^ ((row % 8) * 8))
        )
        src_offset = weight.layout(
            (
                output_tile_idx * output_tile + row,
                k_tile * k_tile_size + col,
            )
        )
        if with_policy:
            # CuTe pointers: ``toint()`` yields the 32-bit shared address for
            # SMEM and the 64-bit global address for GMEM, as the PTX expects.
            _cp_async_16_with_policy(
                smem.iterator + dst_offset,
                weight.iterator + src_offset,
                policy,
                loc=loc,
                ip=ip,
            )
        else:
            prims.cp_async_shared_global(
                smem.iterator.raw_ptr() + dst_offset,
                weight.iterator.raw_ptr() + src_offset,
                16,
                "cg",
                loc=loc,
                ip=ip,
            )


class WarpSplitKTactic(msgspec.Struct, frozen=True):
    output_tile: int
    token_tile: int
    k_tile: int
    stages: int
    b_loader_warps: int


def _check_capability(capability: tuple[int, int]) -> None:
    # mbarrier waits and griddepcontrol need SM90; SM11x/SM12x are not validated.
    if capability != (9, 0) and capability[0] != 10:
        raise ValueError(
            f"warp split-K GEMM supports SM90 and SM10x, got sm_{capability[0]}{capability[1]}"
        )


@functools.cache
def _smem_capacity(capability: tuple[int, int]) -> int:
    _check_capability(capability)
    # Every SM10x part keeps the sm_100 budget its tactics were tuned under, so
    # tactic choice does not change on sm_107 (which reports more).
    arch = "sm_100" if capability[0] == 10 else f"sm_{capability[0]}{capability[1]}"
    return cutlass.memory.get_smem_capacity_in_bytes(arch)


def _device_capability(device_index: int) -> tuple[int, int]:
    return _torch.cuda.get_device_capability(device_index)


def _current_smem_capacity() -> int:
    return _smem_capacity(_device_capability(_torch.cuda.current_device()))


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _smem_bytes(tactic: WarpSplitKTactic) -> int:
    cursor = tactic.output_tile * tactic.k_tile * 2 * tactic.stages
    cursor = _align_up(cursor, 1024)
    cursor += tactic.token_tile * tactic.k_tile * 2 * tactic.stages
    cursor = _align_up(cursor, 16)
    partial_stride = (
        tactic.output_tile * tactic.token_tile // _MAILBOX_GROUP_VALUES
    ) * _MAILBOX_PADDED_GROUP_VALUES
    cursor += _COMPUTE_WARPS * partial_stride * 4
    cursor = _align_up(cursor, 8)
    cursor += tactic.stages * 8
    cursor = _align_up(cursor, 8)
    return cursor + tactic.stages * 8


def _k_tile_count_supported(k_tile: int, k_tile_count: int) -> bool:
    # The two-tile 256-wide schedule can generate an invalid hinted LDGSTS.
    return k_tile != 256 or k_tile_count != 2


def _legal_stages(
    output_tile: int,
    token_tile: int,
    k_tile: int,
    k: int,
    b_loader_warps: int,
    smem_capacity: int,
) -> tuple[int, ...]:
    # A single-stage ring only makes sense when K is one tile; with ring reuse it
    # cannot overlap anything, and ptxas mis-encodes its hinted LDGSTS.
    k_tile_count = k // k_tile
    if k_tile_count > _MAX_K_TILES or not _k_tile_count_supported(k_tile, k_tile_count):
        return ()
    return tuple(
        stages
        for stages in range(_min_stages(k, k_tile), min(_MAX_STAGES, k_tile_count) + 1)
        if _smem_bytes(
            WarpSplitKTactic(output_tile, token_tile, k_tile, stages, b_loader_warps)
        )
        <= smem_capacity
    )


def _min_stages(k: int, k_tile: int) -> int:
    return 1 if k // k_tile <= 1 else 2


def _k_supported(k: int) -> bool:
    return k > 0 and any(
        k % k_tile == 0 and k // k_tile <= _MAX_K_TILES for k_tile in _SUPPORTED_K_TILES
    )


def validate_tactic(
    tactic: WarpSplitKTactic,
    m: int,
    n: int,
    k: int,
    *,
    k_widths: Sequence[int] | None = None,
    n_widths: Sequence[int] | None = None,
) -> None:
    """Raise ``ValueError`` unless ``tactic`` runs this shape on the current GPU.

    ``k_widths`` / ``n_widths`` are the K segments and N regions of
    ``run_warp_splitk_dense_multi_io``; each must be a whole number of tiles.
    """
    _validate_tactic(tactic, m, n, k, smem_capacity=_current_smem_capacity())
    _validate_split(tactic, k_widths=k_widths or (k,), n_widths=n_widths or (n,))


def _validate_tactic(
    tactic: WarpSplitKTactic, m: int, n: int, k: int, *, smem_capacity: int
) -> None:
    if tactic.output_tile not in _SUPPORTED_OUTPUT_TILES:
        raise ValueError(f"unsupported output_tile={tactic.output_tile}")
    if tactic.token_tile not in _SUPPORTED_TOKEN_TILES:
        raise ValueError(f"unsupported token_tile={tactic.token_tile}")
    if tactic.k_tile not in _SUPPORTED_K_TILES:
        raise ValueError(f"unsupported k_tile={tactic.k_tile}")
    if tactic.b_loader_warps not in _SUPPORTED_B_LOADER_WARPS:
        raise ValueError(f"unsupported b_loader_warps={tactic.b_loader_warps}")
    if not 1 <= m <= _MAX_M or n <= 0 or n % tactic.output_tile:
        raise ValueError(f"unsupported shape {(m, n, k)}")
    if k <= 0 or k % tactic.k_tile:
        raise ValueError(f"K={k} must be divisible by k_tile={tactic.k_tile}")
    k_tile_count = k // tactic.k_tile
    if k_tile_count > _MAX_K_TILES:
        raise ValueError(
            f"K={k} spans more than {_MAX_K_TILES} tiles of {tactic.k_tile}"
        )
    if not _k_tile_count_supported(tactic.k_tile, k_tile_count):
        raise ValueError(f"unsupported k_tile={tactic.k_tile} for K={k}")
    if not (
        _min_stages(k, tactic.k_tile) <= tactic.stages <= min(_MAX_STAGES, k_tile_count)
    ):
        raise ValueError(f"invalid stages={tactic.stages}")
    required_smem = _smem_bytes(tactic)
    if required_smem > smem_capacity:
        raise ValueError(f"tactic requires {required_smem} bytes of shared memory")


def _validate_split(
    tactic: WarpSplitKTactic, *, k_widths: Sequence[int], n_widths: Sequence[int]
) -> None:
    # A K tile is loaded from one activation and a CTA stores to one output.
    if any(width % tactic.k_tile for width in k_widths):
        raise ValueError(
            f"K segments {tuple(k_widths)} must be multiples of k_tile={tactic.k_tile}"
        )
    if any(width % tactic.output_tile for width in n_widths):
        raise ValueError(
            f"N regions {tuple(n_widths)} must be multiples of "
            f"output_tile={tactic.output_tile}"
        )


def validate_inputs(
    a: _torch.Tensor,
    b: _torch.Tensor,
    out: _torch.Tensor,
    bias: _torch.Tensor | None = None,
) -> tuple[int, int, int]:
    tensors = (a, b, out) + ((bias,) if bias is not None else ())
    if any(not isinstance(tensor, _torch.Tensor) for tensor in tensors):
        raise ValueError("a, b, out, and bias must be torch tensors")
    if any(tensor.ndim != 2 for tensor in (a, b, out)):
        raise ValueError("a, b, and out must be 2D tensors")
    if a.device.type != "cuda" or any(tensor.device != a.device for tensor in tensors):
        raise ValueError("a, b, out, and bias must be CUDA tensors on the same device")
    if (
        a.dtype != _torch.bfloat16
        or b.dtype != a.dtype
        or out.dtype not in (_torch.bfloat16, _torch.float32)
        or (bias is not None and bias.dtype != a.dtype)
    ):
        raise ValueError("a, b, and bias must be BF16; out must be BF16 or FP32")

    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError(
            f"incompatible shapes: a is {tuple(a.shape)}, b is {tuple(b.shape)}"
        )
    n = b.shape[1]
    if out.shape != (m, n):
        raise ValueError(f"out must have shape {(m, n)}, got {tuple(out.shape)}")
    if bias is not None and (bias.shape != (n,) or not bias.is_contiguous()):
        raise ValueError(f"bias must be a contiguous ({n},) vector")
    if not 1 <= m <= _MAX_M:
        raise ValueError(f"M must be in [1, {_MAX_M}], got {m}")
    if n <= 0 or n % 16:
        raise ValueError(f"N={n} must be a positive multiple of 16")
    if not _k_supported(k):
        raise ValueError(
            f"K={k} must be a positive multiple of 128 spanning at most "
            f"{_MAX_K_TILES} tiles of 256 (or 128) elements"
        )

    if a.stride() != (k, 1):
        raise ValueError(f"a must be packed row-major, got stride {a.stride()}")
    if b.stride() != (1, k):
        raise ValueError(f"b must be packed column-major, got stride {b.stride()}")
    if out.stride() != (n, 1):
        raise ValueError(f"out must be packed row-major, got stride {out.stride()}")
    if any(tensor.data_ptr() % 32 for tensor in tensors):
        raise ValueError("a, b, and out must be 32-byte aligned")
    return m, n, k


def _validate_multi_io_inputs(
    activations: Sequence[_torch.Tensor],
    weight: _torch.Tensor,
    outputs: Sequence[_torch.Tensor],
    bias: _torch.Tensor | None,
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    if not 1 <= len(activations) <= _MAX_K_SEGMENTS:
        raise ValueError(f"expected 1 to {_MAX_K_SEGMENTS} activations")
    if not 1 <= len(outputs) <= _MAX_OUTPUT_REGIONS:
        raise ValueError(f"expected 1 to {_MAX_OUTPUT_REGIONS} outputs")
    tensors = (*activations, weight, *outputs) + ((bias,) if bias is not None else ())
    if any(not isinstance(tensor, _torch.Tensor) for tensor in tensors):
        raise ValueError("activations, weight, outputs, and bias must be torch tensors")
    if any(tensor.ndim != 2 for tensor in (*activations, weight, *outputs)):
        raise ValueError("activations, weight, and outputs must be 2D tensors")
    device = weight.device
    if device.type != "cuda" or any(tensor.device != device for tensor in tensors):
        raise ValueError("all tensors must be CUDA tensors on the same device")
    if (
        any(tensor.dtype != _torch.bfloat16 for tensor in (*activations, weight))
        or any(out.dtype not in (_torch.bfloat16, _torch.float32) for out in outputs)
        or (bias is not None and bias.dtype != _torch.bfloat16)
    ):
        raise ValueError(
            "activations, weight, and bias must be BF16; outputs must be BF16 or FP32"
        )

    m = activations[0].shape[0]
    k_widths = tuple(act.shape[1] for act in activations)
    n_widths = tuple(out.shape[1] for out in outputs)
    n, k = weight.shape
    if any(act.shape[0] != m for act in activations) or any(
        out.shape[0] != m for out in outputs
    ):
        raise ValueError("activations and outputs must share M")
    if sum(k_widths) != k or sum(n_widths) != n:
        raise ValueError(
            f"K segments {k_widths} and N regions {n_widths} must sum to the "
            f"weight shape {(n, k)}"
        )
    if not 0 <= m <= _MAX_M:
        raise ValueError(f"M must be in [0, {_MAX_M}], got {m}")
    if any(width <= 0 or width % 16 for width in n_widths):
        raise ValueError(f"N regions {n_widths} must be positive multiples of 16")
    if not _k_supported(k) or any(width <= 0 or width % 128 for width in k_widths):
        raise ValueError(
            f"K segments {k_widths} must be positive multiples of 128 and K={k} "
            f"must span at most {_MAX_K_TILES} tiles of 256 (or 128) elements"
        )
    if bias is not None and (bias.shape != (n,) or not bias.is_contiguous()):
        raise ValueError(f"bias must be a contiguous ({n},) vector")

    if any(
        tensor.stride() != (tensor.shape[1], 1)
        for tensor in (*activations, weight, *outputs)
    ):
        raise ValueError("activations, weight, and outputs must be packed row-major")
    if any(tensor.data_ptr() % 32 for tensor in tensors):
        raise ValueError("all tensors must be 32-byte aligned")
    return m, k_widths, n_widths


def default_tactic(
    m: int,
    n: int,
    k: int,
    *,
    k_widths: Sequence[int] | None = None,
    n_widths: Sequence[int] | None = None,
) -> WarpSplitKTactic:
    """Heuristic tactic; ``k_widths`` / ``n_widths`` restrict the tiles to ones
    that divide every K segment and N region."""
    k_widths = tuple(k_widths or (k,))
    n_widths = tuple(n_widths or (n,))
    k_tile = (
        256 if k > 512 and all(width % 256 == 0 for width in (k, *k_widths)) else 128
    )
    k_tile_count = k // k_tile
    device_index = _torch.cuda.current_device()
    sm_count = _torch.cuda.get_device_properties(device_index).multi_processor_count
    smem_capacity = _smem_capacity(_device_capability(device_index))
    output_tiles = tuple(
        tile
        for tile in _SUPPORTED_OUTPUT_TILES
        if all(width % tile == 0 for width in (n, *n_widths))
    )
    if not output_tiles:
        raise ValueError(f"N={n} must be divisible by a supported output tile")
    # Minimize scheduled wave work; prefer the wider tile on ties.
    output_tile = min(
        output_tiles,
        key=lambda tile: (
            ((n // tile + sm_count - 1) // sm_count) * tile,
            -tile,
        ),
    )
    # Splitting M over two half-size token tiles doubles the CTA count at the
    # cost of a second pass over each weight tile; that only pays while the
    # doubled grid still fits in one wave.
    token_tile = 8 if m <= 8 else 16 if m <= 16 else 32
    if token_tile > 8 and 2 * (n // output_tile) <= sm_count:
        token_tile //= 2
    # Two B-loader warps pay off once the ring is reused (long K); for a handful
    # of K tiles the extra loader threads only add barrier arrivals.
    b_loader_warps = 2 if token_tile >= 16 or k_tile_count >= 8 else 1
    legal_stages = _legal_stages(
        output_tile, token_tile, k_tile, k, b_loader_warps, smem_capacity
    )
    if not legal_stages:
        raise ValueError(f"no legal tactic for shape {(m, n, k)}")
    cta_count = (n // output_tile) * ((m + token_tile - 1) // token_tile)
    target_residency = 2 if cta_count > sm_count else 1
    resident_stages = tuple(
        stages
        for stages in legal_stages
        if target_residency
        * (
            _smem_bytes(
                WarpSplitKTactic(
                    output_tile, token_tile, k_tile, stages, b_loader_warps
                )
            )
            + _CTA_RESERVED_SMEM
        )
        <= smem_capacity + _CTA_RESERVED_SMEM
    )
    stages = max(resident_stages or legal_stages)
    tactic = WarpSplitKTactic(output_tile, token_tile, k_tile, stages, b_loader_warps)
    _validate_tactic(tactic, m, n, k, smem_capacity=smem_capacity)
    _validate_split(tactic, k_widths=k_widths, n_widths=n_widths)
    return tactic


def autotune_tactics(m: int, n: int, k: int) -> list[WarpSplitKTactic]:
    if not 1 <= m <= _MAX_M or n <= 0 or n % 16 or not _k_supported(k):
        return []
    smem_capacity = _current_smem_capacity()
    tactics: list[WarpSplitKTactic] = []
    for output_tile in _SUPPORTED_OUTPUT_TILES:
        if n % output_tile:
            continue
        for token_tile in _SUPPORTED_TOKEN_TILES:
            if token_tile > 16 and m <= 16:
                continue  # a 32-token tile only pays for M > 16
            for k_tile in _SUPPORTED_K_TILES:
                if k % k_tile:
                    continue
                # Prune same-grid token padding for multi-tile K.
                # Wider tiles can win when K fits one tile.
                if k // k_tile > 1 and any(
                    smaller_token_tile < token_tile
                    and (m + smaller_token_tile - 1) // smaller_token_tile
                    == (m + token_tile - 1) // token_tile
                    for smaller_token_tile in _SUPPORTED_TOKEN_TILES
                ):
                    continue
                for b_loader_warps in _SUPPORTED_B_LOADER_WARPS:
                    tactics.extend(
                        WarpSplitKTactic(
                            output_tile,
                            token_tile,
                            k_tile,
                            stages,
                            b_loader_warps,
                        )
                        for stages in _legal_stages(
                            output_tile,
                            token_tile,
                            k_tile,
                            k,
                            b_loader_warps,
                            smem_capacity,
                        )
                    )
    default = default_tactic(m, n, k)
    return [default, *(tactic for tactic in tactics if tactic != default)]


def _make_smem_layout(dtype, rows: int, cols: int, stages: int):
    copy_elems = 128 // dtype.width
    major_size = min(cols, 128 * 8 // dtype.width)
    outer = cute.tile_to_shape(
        cute.make_layout((8, major_size), stride=(major_size, 1)),
        (rows, cols, stages),
        (0, 1, 2),
    )
    return cute.make_composed_layout(
        cute.make_swizzle(
            min(int(math.log2(major_size // copy_elems)), 3),
            4,
            int(math.log2(copy_elems)),
        ),
        0,
        outer,
    )


class CpAsyncWarpSplitKKernel:
    def __init__(
        self,
        *,
        tactic: WarpSplitKTactic,
        use_pdl: bool,
        has_bias: bool,
        segment_k_tiles: tuple[int, ...],
        region_bounds: tuple[int, ...],
    ) -> None:
        self.output_tile = tactic.output_tile
        self.token_tile = tactic.token_tile
        self.k_tile = tactic.k_tile
        self.stages = tactic.stages
        self.b_loader_warps = tactic.b_loader_warps
        self.compute_warp_base = _A_LOADER_WARPS + self.b_loader_warps
        self.threads = (self.compute_warp_base + _COMPUTE_WARPS) * 32
        self.load_threads = self.compute_warp_base * 32
        self.use_pdl = use_pdl
        self.has_bias = has_bias
        self.smem_bytes = _smem_bytes(tactic)
        self.values_per_lane = self.output_tile * self.token_tile // 32
        # Segment s supplies the segment_k_tiles[s] K tiles from segment_tile_starts[s].
        self.segment_k_tiles = segment_k_tiles
        self.segment_tile_starts = tuple(
            sum(segment_k_tiles[:segment]) for segment in range(len(segment_k_tiles))
        )
        # Output region r holds weight rows [region_bounds[r], region_bounds[r + 1]).
        self.region_bounds = region_bounds

    @cute.experimental.jit
    def __call__(
        self,
        weight,
        act0,
        act1,
        act2,
        act3,
        bias,
        out0,
        out1,
        out2,
        out3,
        stream: _cuda.CUstream,
    ):
        sA_layout = _make_smem_layout(
            weight.element_type, self.output_tile, self.k_tile, self.stages
        )
        sB_layout = _make_smem_layout(
            act0.element_type, self.token_tile, self.k_tile, self.stages
        )
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(weight.element_type, cutlass.Float32, _MMA_SHAPE),
            cute.make_layout((1, 1, _COMPUTE_WARPS)),
            permutation_mnk=(self.output_tile, self.token_tile, self.k_tile),
        )
        self.kernel(
            weight,
            act0,
            act1,
            act2,
            act3,
            bias,
            out0,
            out1,
            out2,
            out3,
            sA_layout,
            sB_layout,
            tiled_mma,
        ).launch(
            grid=(
                cute.ceil_div(weight.shape[0], self.output_tile),
                cute.ceil_div(act0.shape[0], self.token_tile),
                1,
            ),
            block=(self.threads, 1, 1),
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        weight: cute.Tensor,
        act0: cute.Tensor,
        act1: cute.Tensor,
        act2: cute.Tensor,
        act3: cute.Tensor,
        bias: cute.Tensor,  # (N,) broadcast bias; unread when has_bias is False
        out0: cute.Tensor,
        out1: cute.Tensor,
        out2: cute.Tensor,
        out3: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
    ):
        # Slots past the segment / region count repeat act0 / out0 and are unread.
        activations = (act0, act1, act2, act3)
        outputs = (out0, out1, out2, out3)
        tidx, _, _ = cute.arch.thread_idx()
        output_tile_idx, token_tile_idx, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = tidx % cute.arch.WARP_SIZE
        reuse_stages = cute.size(weight, mode=[1]) > self.stages * self.k_tile

        sA = cute_ext.allocate(
            weight.element_type,
            cute.AddressSpace.smem,
            sA_layout,
            alignment=1024,
        )
        sB = cute_ext.allocate(
            act0.element_type,
            cute.AddressSpace.smem,
            sB_layout,
            alignment=1024,
        )
        values_per_lane = self.values_per_lane
        partial_stride = (
            self.output_tile * self.token_tile // _MAILBOX_GROUP_VALUES
        ) * _MAILBOX_PADDED_GROUP_VALUES
        partials = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(_COMPUTE_WARPS * partial_stride),
            alignment=16,
        )
        bar_full_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.stages),
            alignment=8,
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.stages),
            alignment=8,
        )
        bar_full = bar_full_arr.iterator
        bar_empty = bar_empty_arr.iterator

        k_tile_count = cute.size(weight, mode=[1]) // self.k_tile

        if warp_idx < _A_LOADER_WARPS:
            # Tile 0 has no mbarrier dependency: issue it before the init handshake
            # so its DRAM latency overlaps init and the CTA barrier.
            _issue_weight_tile(
                sA,
                weight,
                tidx,
                output_tile_idx,
                0,
                0,
                cutlass.Int64(0),
                self.output_tile,
                self.k_tile,
                False,
            )
        bias_value = cutlass.Float32(0.0)
        if warp_idx == self.compute_warp_base:
            # The first compute warp is idle here; it initializes the mbarriers so
            # the loader warps are never held behind the init.
            if lane_idx < self.stages:
                cute.arch.mbarrier_init(bar_full + lane_idx, self.load_threads)
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_init(bar_empty + lane_idx, _COMPUTE_WARPS)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp_idx < _A_LOADER_WARPS:
            # Weights stream through L2 evict_first so the activation rows every
            # CTA re-reads stay resident. Only tile 0's arrive may follow the init.
            weight_policy = _l2_policy_evict_first()
            empty_phase = cutlass.Int32(1)
            prims.cp_async_mbarrier_arrive(bar_full.llvm_ptr, noinc=True)
            for k_tile in cutlass.range(1, k_tile_count, unroll=1):
                stage = k_tile % self.stages
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                _issue_weight_tile(
                    sA,
                    weight,
                    tidx,
                    output_tile_idx,
                    k_tile,
                    stage,
                    weight_policy,
                    self.output_tile,
                    self.k_tile,
                    True,
                )
                prims.cp_async_mbarrier_arrive((bar_full + stage).llvm_ptr, noinc=True)
                if stage == self.stages - 1:
                    empty_phase = empty_phase ^ 1
        elif warp_idx < self.compute_warp_base:
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            b_loader_threads = self.b_loader_warps * 32
            local_tid = tidx - _A_LOADER_WARPS * 32
            copy_count = self.token_tile * self.k_tile // (8 * b_loader_threads)
            empty_phase = cutlass.Int32(1)
            # One loop per K segment, so no copy branches on which activation
            # it reads; the phase bookkeeping runs on the global K tile index.
            for segment in cutlass.range_constexpr(len(self.segment_k_tiles)):
                activation = activations[segment]
                tile_start = self.segment_tile_starts[segment]
                for local_k_tile in cutlass.range(
                    self.segment_k_tiles[segment], unroll=1
                ):
                    k_tile = tile_start + local_k_tile
                    stage = k_tile % self.stages
                    if const_expr(reuse_stages):
                        cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                    self._load_activation_tile(
                        sB,
                        activation,
                        stage,
                        local_k_tile,
                        local_tid,
                        token_tile_idx,
                        b_loader_threads,
                        copy_count,
                    )
                    prims.cp_async_mbarrier_arrive(
                        (bar_full + stage).llvm_ptr, noinc=True
                    )
                    if stage == self.stages - 1:
                        empty_phase = empty_phase ^ 1
            if const_expr(self.use_pdl):
                # This grid-level hint is idempotent across B-loader warps.
                cute.arch.griddepcontrol_launch_dependents()
        else:
            if const_expr(self.has_bias):
                # Each reduce lane owns one output feature (32 % output_tile == 0).
                # Issue the load here, after the CTA barrier: waiting on the first
                # full stage hides its latency, whereas a load before the barrier
                # would hold the loader warps behind a cold DRAM read.
                bias_value = bias[
                    output_tile_idx * self.output_tile + lane_idx % self.output_tile
                ].to(cutlass.Float32)
            mma_tid = tidx - self.compute_warp_base * cute.arch.WARP_SIZE
            compute_warp = warp_idx - self.compute_warp_base
            thr_mma = tiled_mma.get_slice(mma_tid)
            acc = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((self.output_tile, self.token_tile)),
                cutlass.Float32,
            )
            acc.fill(0.0)
            k_phases = self.k_tile // (_COMPUTE_WARPS * _MMA_SHAPE[2])
            m_iters = self.output_tile // _MMA_SHAPE[0]
            n_iters = self.token_tile // _MMA_SHAPE[1]
            # Fragment addresses are loop-invariant per lane apart from the stage
            # base; hoist them so the K loop issues ldmatrix from one add each.
            a_offsets = tuple(
                tuple(
                    _a_fragment_offset(
                        lane_idx, compute_warp, m_iter, phase, self.k_tile
                    )
                    for phase in range(k_phases)
                )
                for m_iter in range(m_iters)
            )
            b_offsets = tuple(
                tuple(
                    _b_pair_offset(
                        lane_idx,
                        compute_warp,
                        n_iter,
                        pair,
                        self.token_tile,
                        self.k_tile,
                    )
                    for pair in range(k_phases // 2)
                )
                for n_iter in range(n_iters)
            )
            a_stage_elems = self.output_tile * self.k_tile
            b_stage_elems = self.token_tile * self.k_tile
            full_phase = cutlass.Int32(0)
            # The K-tile count is static: unrolling fully folds `stage`, barrier
            # addresses and phases to constants, so every ldmatrix is
            # `[R_lane + imm]` (a rolled loop costs 3-16% on K=7168).
            for k_tile in cutlass.range(k_tile_count, unroll_full=True):
                stage = k_tile % self.stages
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)
                a_base = stage * a_stage_elems
                b_base = stage * b_stage_elems
                a_registers = tuple(
                    tuple(
                        _ldmatrix_x4(sA, a_base + a_offsets[m_iter][phase])
                        for phase in range(k_phases)
                    )
                    for m_iter in range(m_iters)
                )
                b_registers = tuple(
                    tuple(
                        _ldmatrix_x4(sB, b_base + b_offsets[n_iter][pair])
                        for pair in range(k_phases // 2)
                    )
                    for n_iter in range(n_iters)
                )
                for phase in cutlass.range_constexpr(k_phases):
                    for n_iter in cutlass.range_constexpr(n_iters):
                        packed_b = b_registers[n_iter][phase // 2]
                        b_offset = (phase % 2) * 2
                        for m_iter in cutlass.range_constexpr(m_iters):
                            acc_base = (m_iter * n_iters + n_iter) * 4
                            result = prims.mma_sync(
                                llvm.StructType.get_literal(
                                    [cutlass.Float32.mlir_type] * 4
                                ),
                                shape=_MMA_SHAPE,
                                layout_a=prims.MMALayout.ROW,
                                layout_b=prims.MMALayout.COL,
                                operand_a=[
                                    a_registers[m_iter][phase][i].ir_value()
                                    for i in range(4)
                                ],
                                operand_b=[
                                    packed_b[b_offset + i].ir_value() for i in range(2)
                                ],
                                operand_c=[
                                    acc[acc_base + i].ir_value() for i in range(4)
                                ],
                                multiplicand_a_ptx_type=prims.MMAType.BF16,
                                multiplicand_b_ptx_type=prims.MMAType.BF16,
                            )
                            for i in cutlass.range_constexpr(4):
                                acc[acc_base + i] = cutlass.Float32(
                                    llvm.extractvalue(
                                        cutlass.Float32.mlir_type,
                                        result,
                                        [i],
                                    )
                                )
                if const_expr(reuse_stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(bar_empty + stage)
                if stage == self.stages - 1:
                    full_phase = full_phase ^ 1
            for value in cutlass.range_constexpr(values_per_lane):
                atom = value % 4
                fragment = value // 4
                m_iter = fragment // n_iters
                n_iter = fragment % n_iters
                feature = lane_idx // 4 + 8 * (atom // 2) + m_iter * _MMA_SHAPE[0]
                token = 8 * n_iter + 2 * (lane_idx % 4) + atom % 2
                partials[
                    compute_warp * partial_stride
                    + _partial_offset(feature, token, self.token_tile)
                ] = acc[value]
            prims.barrier_cta_sync(1, thread_count=_COMPUTE_WARPS * 32)

        if warp_idx == self.compute_warp_base:
            final_acc = cute.make_rmem_tensor(
                cute.make_layout(values_per_lane), cutlass.Float32
            )
            for value in cutlass.range_constexpr(values_per_lane):
                linear = value * cute.arch.WARP_SIZE + lane_idx
                feature = linear % self.output_tile
                token = linear // self.output_tile
                total = cutlass.Float32(0)
                for peer in cutlass.range_constexpr(_COMPUTE_WARPS):
                    total = (
                        total
                        + partials[
                            peer * partial_stride
                            + _partial_offset(feature, token, self.token_tile)
                        ]
                    )
                if const_expr(self.has_bias):
                    total = total + bias_value
                final_acc[value] = total

            output_base = output_tile_idx * self.output_tile
            token_base = token_tile_idx * self.token_tile
            if const_expr(len(self.region_bounds) == 2):
                self._store_tile(out0, final_acc, output_base, token_base, lane_idx)
            else:
                # Region bounds are output_tile multiples, so each CTA takes
                # exactly one of these CTA-uniform branches.
                for region in cutlass.range_constexpr(len(self.region_bounds) - 1):
                    region_start = self.region_bounds[region]
                    if (
                        output_base >= region_start
                        and output_base < self.region_bounds[region + 1]
                    ):
                        self._store_tile(
                            outputs[region],
                            final_acc,
                            output_base - region_start,
                            token_base,
                            lane_idx,
                        )

    @cute.jit
    def _load_activation_tile(
        self,
        sB,
        activation,
        stage,
        local_k_tile,
        local_tid,
        token_tile_idx,
        b_loader_threads,
        copy_count,
    ):
        for copy_idx in cutlass.range_constexpr(copy_count):
            linear = local_tid * 8 + copy_idx * b_loader_threads * 8
            row = linear // self.k_tile
            col = linear % self.k_tile
            global_row = token_tile_idx * self.token_tile + row
            # Out-of-range token rows are discarded by the output predicate.
            if global_row < cute.size(activation, mode=[0]):
                dst = (
                    sB.iterator.raw_ptr()
                    + stage * self.token_tile * self.k_tile
                    + row * self.k_tile
                    + (col ^ ((row % 8) * 8))
                )
                src = activation.iterator.raw_ptr() + activation.layout(
                    (
                        global_row,
                        local_k_tile * self.k_tile + col,
                    )
                )
                prims.cp_async_shared_global(dst, src, 16, "cg")

    @cute.jit
    def _store_tile(self, output, final_acc, col_base, token_base, lane_idx):
        for value in cutlass.range_constexpr(self.values_per_lane):
            linear = value * cute.arch.WARP_SIZE + lane_idx
            feature = linear % self.output_tile
            token = linear // self.output_tile
            if token_base + token < cute.size(output, mode=[0]):
                output[token_base + token, col_base + feature] = final_acc[value].to(
                    output.element_type
                )


def _from_dlpack(tensor: _torch.Tensor):
    return from_dlpack(tensor, assumed_align=32)


# Everything a compiled kernel is specialized on, besides the GPU arch.
class _KernelVariant(msgspec.Struct, frozen=True):
    m: int
    k_widths: tuple[int, ...]
    n_widths: tuple[int, ...]
    out_dtypes: tuple[str, ...]
    tactic: WarpSplitKTactic
    use_pdl: bool
    has_bias: bool


# Compiled variants persist across processes under SGLANG_CUTE_AOT_CACHE_DIR, in
# a directory fingerprinted by this file, the CuTe version and the target arch.
_COMPILED = get_jit_cache(
    "dense_bf16_gemm_warp_splitk",
    source_paths=(__file__,),
    enable_tvm_ffi=False,
)


def _cache_key(device: _torch.device, variant: _KernelVariant) -> tuple:
    # The disk key drops the device index, so one export serves every GPU of
    # the same arch; the capability also guards an in-process mixed-arch host.
    return (
        _device_capability(device.index),
        device,
        variant.m,
        variant.k_widths,
        variant.n_widths,
        variant.out_dtypes,
        msgspec.structs.astuple(variant.tactic),
        variant.use_pdl,
        variant.has_bias,
    )


def _compile(device: _torch.device, variant: _KernelVariant):
    tactic = variant.tactic
    with _torch.cuda.device(device):
        kernel = CpAsyncWarpSplitKKernel(
            tactic=tactic,
            use_pdl=variant.use_pdl,
            has_bias=variant.has_bias,
            segment_k_tiles=tuple(width // tactic.k_tile for width in variant.k_widths),
            region_bounds=tuple(
                sum(variant.n_widths[:region])
                for region in range(len(variant.n_widths) + 1)
            ),
        )
        n, k = sum(variant.n_widths), sum(variant.k_widths)
        out_dtypes = tuple(getattr(_torch, name) for name in variant.out_dtypes)
        weight = _torch.empty((n, k), device=device, dtype=_torch.bfloat16)
        activations = tuple(
            _torch.empty((variant.m, width), device=device, dtype=_torch.bfloat16)
            for width in variant.k_widths
        )
        outputs = tuple(
            _torch.empty((variant.m, width), device=device, dtype=dtype)
            for width, dtype in zip(variant.n_widths, out_dtypes)
        )
        bias = (
            _torch.empty((n,), device=device, dtype=_torch.bfloat16)
            if variant.has_bias
            else None
        )
        stream = _cuda.CUstream(_torch.cuda.current_stream(device).cuda_stream)
        return cute_ext.compile(
            kernel, *_kernel_args(weight, activations, outputs, bias), stream
        )


def _get_compiled(device: _torch.device, variant: _KernelVariant):
    key = _cache_key(device, variant)
    if key not in _COMPILED:
        _COMPILED[key] = _compile(device, variant)
    return _COMPILED[key]


def _kernel_args(
    weight: _torch.Tensor,
    activations: Sequence[_torch.Tensor],
    outputs: Sequence[_torch.Tensor],
    bias: _torch.Tensor | None,
) -> tuple:
    act_args = [_from_dlpack(act) for act in activations]
    out_args = [_from_dlpack(out) for out in outputs]
    act_args += [act_args[0]] * (_MAX_K_SEGMENTS - len(act_args))
    out_args += [out_args[0]] * (_MAX_OUTPUT_REGIONS - len(out_args))
    # Without bias the kernel never reads its bias argument; pass a row of the
    # first output so the launch signature stays fixed.
    bias_arg = _from_dlpack(bias if bias is not None else outputs[0][0])
    return (_from_dlpack(weight), *act_args, bias_arg, *out_args)


def _launch(
    *,
    weight: _torch.Tensor,
    activations: Sequence[_torch.Tensor],
    outputs: Sequence[_torch.Tensor],
    bias: _torch.Tensor | None,
    variant: _KernelVariant,
) -> None:
    device = weight.device
    with _torch.cuda.device(device):
        compiled = _get_compiled(device, variant)
        stream = _cuda.CUstream(_torch.cuda.current_stream(device).cuda_stream)
        compiled(*_kernel_args(weight, activations, outputs, bias), stream)


def _dtype_name(dtype: _torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def run_warp_splitk_dense(a, b, out, pdl: bool, tactic: WarpSplitKTactic, bias=None):
    m, n, k = validate_inputs(a, b, out, bias)
    device_index = a.device.index
    assert device_index is not None
    _validate_tactic(
        tactic, m, n, k, smem_capacity=_smem_capacity(_device_capability(device_index))
    )
    _launch(
        weight=b.T,
        activations=(a,),
        outputs=(out,),
        bias=bias,
        variant=_KernelVariant(
            m=m,
            k_widths=(k,),
            n_widths=(n,),
            out_dtypes=(_dtype_name(out.dtype),),
            tactic=tactic,
            use_pdl=pdl,
            has_bias=bias is not None,
        ),
    )
    return out


def run_warp_splitk_dense_multi_io(
    activations: Sequence[_torch.Tensor],
    weight: _torch.Tensor,
    outputs: Sequence[_torch.Tensor],
    *,
    pdl: bool,
    tactic: WarpSplitKTactic | None = None,
    bias: _torch.Tensor | None = None,
) -> None:
    """``concat(outputs, dim=1) = concat(activations, dim=1) @ weight.T + bias`` in one launch.

    ``activations`` are 1 to 4 BF16 ``[M, K_i]`` row-major tensors that split K
    in order; ``weight`` is BF16 ``[N, K]`` row-major; ``outputs`` are 1 to 4
    ``[M, N_j]`` row-major tensors, each BF16 or FP32, that split N in order;
    ``bias`` is an optional BF16 ``[N]`` vector. Every ``K_i`` must be a multiple
    of the tactic's ``k_tile`` and every ``N_j`` of its ``output_tile``;
    ``tactic=None`` picks the default for these splits. The result is bit
    identical to a single-segment, single-output launch with the same tactic on
    the concatenated activations, cast per output. M is 1 to 32; M = 0 is a no-op.
    Each distinct M compiles a kernel, so call this from CUDA-graph capture or
    pre-build the M values with ``prebuild_warp_splitk_dense_multi_io``.
    """
    m, k_widths, n_widths = _validate_multi_io_inputs(
        activations, weight, outputs, bias
    )
    if m == 0:
        return
    n, k = weight.shape
    if tactic is None:
        with _torch.cuda.device(weight.device):
            tactic = default_tactic(m, n, k, k_widths=k_widths, n_widths=n_widths)
    _validate_tactic(
        tactic,
        m,
        n,
        k,
        smem_capacity=_smem_capacity(_device_capability(weight.device.index)),
    )
    _validate_split(tactic, k_widths=k_widths, n_widths=n_widths)
    _launch(
        weight=weight,
        activations=activations,
        outputs=outputs,
        bias=bias,
        variant=_KernelVariant(
            m=m,
            k_widths=k_widths,
            n_widths=n_widths,
            out_dtypes=tuple(_dtype_name(out.dtype) for out in outputs),
            tactic=tactic,
            use_pdl=pdl,
            has_bias=bias is not None,
        ),
    )


def prebuild_warp_splitk_dense_multi_io(
    *,
    m_values: Iterable[int],
    k_widths: Sequence[int],
    n_widths: Sequence[int],
    out_dtypes: Sequence[_torch.dtype],
    pdl: bool,
    has_bias: bool = False,
    tactic_for_m: Callable[[int], WarpSplitKTactic] | None = None,
    device: _torch.device | None = None,
) -> int:
    """Compile, or load from the AOT cache, the kernel for every M in ``m_values``.

    Call at warmup with the batch sizes the serving path launches (for example
    the CUDA-graph batch sizes) so no launch compiles. M outside [1, 32] is
    skipped. ``tactic_for_m`` defaults to ``default_tactic`` for these splits.
    A single-segment, single-output call also covers ``run_warp_splitk_dense``.
    Returns the number of kernels compiled by this call; loads are not counted.
    """
    if device is None or device.index is None:
        device = _torch.device("cuda", _torch.cuda.current_device())
    k_widths, n_widths = tuple(k_widths), tuple(n_widths)
    if len(out_dtypes) != len(n_widths):
        raise ValueError("out_dtypes must have one dtype per N region")
    n, k = sum(n_widths), sum(k_widths)
    compiled_count = 0
    with _torch.cuda.device(device):
        smem_capacity = _smem_capacity(_device_capability(device.index))
        for m in sorted(set(m_values)):
            if not 1 <= m <= _MAX_M:
                continue
            tactic = (
                tactic_for_m(m)
                if tactic_for_m is not None
                else default_tactic(m, n, k, k_widths=k_widths, n_widths=n_widths)
            )
            _validate_tactic(tactic, m, n, k, smem_capacity=smem_capacity)
            _validate_split(tactic, k_widths=k_widths, n_widths=n_widths)
            variant = _KernelVariant(
                m=m,
                k_widths=k_widths,
                n_widths=n_widths,
                out_dtypes=tuple(_dtype_name(dtype) for dtype in out_dtypes),
                tactic=tactic,
                use_pdl=pdl,
                has_bias=has_bias,
            )
            key = _cache_key(device, variant)
            if key not in _COMPILED:
                _COMPILED[key] = _compile(device, variant)
                compiled_count += 1
    return compiled_count


__all__ = [
    "WarpSplitKTactic",
    "autotune_tactics",
    "default_tactic",
    "prebuild_warp_splitk_dense_multi_io",
    "run_warp_splitk_dense",
    "run_warp_splitk_dense_multi_io",
    "validate_inputs",
    "validate_tactic",
]
