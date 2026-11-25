"""Hy4-Flash MoE block dense projections as two skinny BF16 GEMMs at decode sizes.

Every Hy4-Flash MoE block runs five dense BF16 projections: the routed experts'
latent down and up projections, the shared expert's gate_up and down
projections, and the router (FP32 logits). Three read the same activation and
the other two are summed, so they stack into two weights:

- front: ``front = [latent_down; shared_gate_up; router]`` ``[n_lat + n_gu + n_router,
  hidden]``. One launch writes ``lat``, ``gu`` (BF16) and the router logits (FP32,
  exact BF16 products summed in FP32).
- back: ``back = [latent_up | shared_down]`` ``[hidden, k_lat + k_shared]``. One
  launch reads ``lat_sum`` and ``act`` as two K segments and writes their summed
  projection, rounded to BF16 once.

Coverage. Both entry points compile per token count, so they are meant for
CUDA-graph capture and return None for any M they do not cover; the caller then
runs its own path. M 1-32 runs the warp split-K GEMM
(``dense_bf16_gemm_warp_splitk``, multi-IO, AOT-cached), M 33-128 the Triton
kernel below. Launch configs were tuned on H200 at TP4 (front ``[2816, 3072]``,
back ``[3072, 1536 + 384]``); a token bucket whose config does not fit the
actual shapes (the back GEMM at TP8, K = 1536 + 192) is not covered.

Concurrency: the Triton split-K launches count finished tiles in counters owned
by the ``Hy4MoeDenseWeights``. Do not run two launches of the same GEMM on one
``Hy4MoeDenseWeights`` concurrently (two streams, or two graphs replayed at once).
"""

from __future__ import annotations

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

# Largest token count each implementation serves.
_WARP_SPLITK_MAX_M = 32
_TRITON_MAX_M = 128
_MAX_TILES = 2048

# WarpSplitKTactic (output_tile, token_tile, k_tile, stages, b_loader_warps) per
# token bucket (next power of two). Swept on H200 at the TP4 shapes, CUDA-graph
# replay with L2 flushed; tactics within 0.3 us of each other.
_WARP_SPLITK_FRONT = {
    1: (16, 8, 256, 4, 1),
    2: (16, 8, 256, 4, 1),
    4: (16, 8, 256, 4, 1),
    8: (32, 8, 256, 6, 1),
    16: (32, 16, 256, 6, 1),
    32: (32, 32, 256, 6, 2),
}
_WARP_SPLITK_BACK = {
    1: (16, 8, 128, 6, 1),
    2: (16, 8, 128, 6, 2),
    4: (32, 8, 128, 6, 1),
    8: (32, 8, 128, 6, 1),
    16: (16, 16, 128, 6, 2),
    32: (32, 32, 128, 8, 2),
}

# Triton launch (BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages, ACC, FMA) per
# token bucket, tuned on H200 at the TP4 shapes. Only buckets 64 and 128 serve
# captured graphs; 1-32 cover warp split-K tactics that do not fit the shapes.
_TRITON_FRONT = {
    1: (2, 1024, 1, 1, 1, 1, 1),
    2: (2, 512, 1, 1, 1, 1, 1),
    4: (32, 256, 1, 2, 6, 2, 0),
    8: (32, 256, 1, 2, 6, 2, 0),
    16: (32, 256, 1, 2, 6, 2, 0),
    32: (32, 256, 1, 2, 6, 1, 0),
    64: (64, 128, 3, 4, 4, 1, 0),
    128: (64, 128, 3, 4, 4, 1, 0),
}
_TRITON_BACK = {
    1: (8, 128, 5, 1, 1, 1, 1),
    2: (16, 128, 1, 2, 6, 1, 0),
    4: (16, 128, 1, 2, 6, 1, 0),
    8: (16, 128, 1, 2, 6, 1, 0),
    16: (16, 128, 1, 2, 6, 1, 0),
    32: (32, 128, 1, 4, 6, 1, 0),
    64: (64, 128, 1, 4, 6, 2, 0),
    # Split-K 2 gives 96 CTAs instead of 48 (M = 96: 15.9 -> 15.2 us).
    128: (64, 64, 2, 4, 6, 1, 0),
}


class Hy4MoeDenseWeights(msgspec.Struct, frozen=True):
    front: torch.Tensor
    back: torch.Tensor
    n_lat: int
    n_gu: int
    k_lat: int
    warp_splitk_front_buckets: frozenset[int]
    warp_splitk_back_buckets: frozenset[int]
    triton_front_buckets: frozenset[int]
    triton_back_buckets: frozenset[int]
    # int32 [_MAX_TILES] each, zero between launches (the last tile resets it).
    front_tile_counters: torch.Tensor
    back_tile_counters: torch.Tensor

    def front_views(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(latent_down, shared_gate_up, router)``: contiguous row ranges of ``front``."""
        n_lat, n_gu = self.n_lat, self.n_gu
        return (
            self.front[:n_lat],
            self.front[n_lat : n_lat + n_gu],
            self.front[n_lat + n_gu :],
        )

    def back_views(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(latent_up, shared_down)``: column ranges of ``back`` (row stride ``back.shape[1]``)."""
        return self.back[:, : self.k_lat], self.back[:, self.k_lat :]


def stack_hy4_moe_dense_weights(
    *,
    latent_down: torch.Tensor,
    shared_gate_up: torch.Tensor,
    router: torch.Tensor,
    latent_up: torch.Tensor,
    shared_down: torch.Tensor,
) -> Hy4MoeDenseWeights:
    """Copy the five ``[N, K]`` BF16 weights into the stacked front and back weights.

    The copies hold the same bytes as the inputs; point the module parameters at
    ``front_views()`` / ``back_views()`` so the originals can be freed and later
    weight loads write into the stacked tensors.
    """
    weights = (latent_down, shared_gate_up, router, latent_up, shared_down)
    if any(
        w.dtype != torch.bfloat16 or w.dim() != 2 or w.device != latent_down.device
        for w in weights
    ):
        raise ValueError("weights must be 2D BF16 tensors on one device")
    hidden = latent_down.shape[1]
    if shared_gate_up.shape[1] != hidden or router.shape[1] != hidden:
        raise ValueError("front weights must share K")
    if latent_up.shape[0] != hidden or shared_down.shape[0] != hidden:
        raise ValueError("back weights must have N equal to the front K")
    if any(w.shape[1] % 8 for w in weights):
        raise ValueError("K must be a multiple of 8 (16-byte rows)")
    front = torch.cat([latent_down, shared_gate_up, router], dim=0)
    back = torch.cat([latent_up, shared_down], dim=1)
    n_lat, n_gu, k_lat = (
        latent_down.shape[0],
        shared_gate_up.shape[0],
        latent_up.shape[1],
    )
    n_widths = (n_lat, n_gu, router.shape[0])
    k_widths = (k_lat, shared_down.shape[1])
    with torch.cuda.device(front.device):
        ws_front = _warp_splitk_buckets(
            _WARP_SPLITK_FRONT,
            n=front.shape[0],
            k=hidden,
            k_widths=(hidden,),
            n_widths=n_widths,
        )
        ws_back = _warp_splitk_buckets(
            _WARP_SPLITK_BACK,
            n=hidden,
            k=back.shape[1],
            k_widths=k_widths,
            n_widths=(hidden,),
        )
    return Hy4MoeDenseWeights(
        front=front,
        back=back,
        n_lat=n_lat,
        n_gu=n_gu,
        k_lat=k_lat,
        warp_splitk_front_buckets=ws_front,
        warp_splitk_back_buckets=ws_back,
        triton_front_buckets=_triton_buckets(
            _TRITON_FRONT, n=front.shape[0], k=hidden, k0=hidden, n0=n_lat, n1=n_gu
        ),
        triton_back_buckets=_triton_buckets(
            _TRITON_BACK, n=hidden, k=back.shape[1], k0=k_lat, n0=hidden, n1=0
        ),
        front_tile_counters=torch.zeros(
            _MAX_TILES, dtype=torch.int32, device=front.device
        ),
        back_tile_counters=torch.zeros(
            _MAX_TILES, dtype=torch.int32, device=front.device
        ),
    )


def hy4_moe_dense_front(
    weights: Hy4MoeDenseWeights, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """``(lat, gu, router_logits)`` of ``x [M, hidden]`` BF16, or None when no kernel covers M.

    ``lat`` / ``gu`` are BF16, ``router_logits`` FP32. Compiles per M (call during
    CUDA-graph capture).
    """
    m = x.shape[0]
    _check_activation(x, cols=weights.front.shape[1], m=m)
    bucket = _bucket(m)
    n_lat, n_gu = weights.n_lat, weights.n_gu
    n_router = weights.front.shape[0] - n_lat - n_gu
    use_warp_splitk = _use_warp_splitk(m, bucket, weights.warp_splitk_front_buckets, x)
    if not use_warp_splitk and not (
        0 < m <= _TRITON_MAX_M and bucket in weights.triton_front_buckets
    ):
        return None
    lat = torch.empty(m, n_lat, dtype=torch.bfloat16, device=x.device)
    gu = torch.empty(m, n_gu, dtype=torch.bfloat16, device=x.device)
    logits = torch.empty(m, n_router, dtype=torch.float32, device=x.device)
    if use_warp_splitk:
        _run_warp_splitk(
            (x,), weights.front, (lat, gu, logits), _WARP_SPLITK_FRONT[bucket]
        )
    else:
        _launch_triton(
            weights.front,
            x,
            x,
            (lat, gu, logits),
            weights.front_tile_counters,
            k0=weights.front.shape[1],
            n0=n_lat,
            n1=n_gu,
            cfg=_TRITON_FRONT[bucket],
        )
    return lat, gu, logits


def hy4_moe_dense_back(
    weights: Hy4MoeDenseWeights, lat_sum: torch.Tensor, act: torch.Tensor
) -> torch.Tensor | None:
    """``[lat_sum | act] @ back.T`` BF16 ``[M, hidden]``, or None when no kernel covers M.

    One FP32 accumulator and one BF16 rounding for both projections. Compiles per
    M (call during CUDA-graph capture).
    """
    m = lat_sum.shape[0]
    k_lat = weights.k_lat
    _check_activation(lat_sum, cols=k_lat, m=m)
    _check_activation(act, cols=weights.back.shape[1] - k_lat, m=m)
    bucket = _bucket(m)
    use_warp_splitk = _use_warp_splitk(
        m, bucket, weights.warp_splitk_back_buckets, lat_sum, act
    )
    if not use_warp_splitk and not (
        0 < m <= _TRITON_MAX_M and bucket in weights.triton_back_buckets
    ):
        return None
    out = torch.empty(m, weights.back.shape[0], dtype=torch.bfloat16, device=act.device)
    if use_warp_splitk:
        _run_warp_splitk(
            (lat_sum, act), weights.back, (out,), _WARP_SPLITK_BACK[bucket]
        )
    else:
        _launch_triton(
            weights.back,
            lat_sum,
            act,
            (out, out, out),
            weights.back_tile_counters,
            k0=k_lat,
            n0=weights.back.shape[0],
            n1=0,
            cfg=_TRITON_BACK[bucket],
        )
    return out


def _bucket(m: int) -> int:
    return triton.next_power_of_2(m)


def _check_activation(t: torch.Tensor, *, cols: int, m: int) -> None:
    if t.dtype != torch.bfloat16 or t.dim() != 2 or t.shape != (m, cols):
        raise ValueError(
            f"expected a BF16 [{m}, {cols}] activation, got {t.dtype} {tuple(t.shape)}"
        )
    # Both kernels load 16-byte vectors along K.
    if not t.is_contiguous() or t.data_ptr() % 16:
        raise ValueError("activations must be contiguous and 16-byte aligned")


def _use_warp_splitk(
    m: int, bucket: int, buckets: frozenset[int], *tensors: torch.Tensor
) -> bool:
    # The CuTe kernel assumes 32-byte aligned operands.
    return (
        0 < m <= _WARP_SPLITK_MAX_M
        and bucket in buckets
        and all(t.data_ptr() % 32 == 0 for t in tensors)
    )


# ---------------------------------------------------------------------------
# Warp split-K path (M <= 32)
# ---------------------------------------------------------------------------


def _warp_splitk_buckets(
    tactics: dict, *, n: int, k: int, k_widths: tuple, n_widths: tuple
) -> frozenset[int]:
    from sglang.kernels.ops.gemm import dense_bf16_gemm_warp_splitk as warp_splitk

    buckets = set()
    for bucket, fields in tactics.items():
        try:
            warp_splitk.validate_tactic(
                warp_splitk.WarpSplitKTactic(*fields),
                bucket,
                n,
                k,
                k_widths=k_widths,
                n_widths=n_widths,
            )
        except ValueError:
            continue
        buckets.add(bucket)
    return frozenset(buckets)


def _run_warp_splitk(activations, weight, outputs, tactic_fields) -> None:
    from sglang.kernels.ops.gemm import dense_bf16_gemm_warp_splitk as warp_splitk

    warp_splitk.run_warp_splitk_dense_multi_io(
        activations,
        weight,
        outputs,
        pdl=is_arch_support_pdl(),
        tactic=warp_splitk.WarpSplitKTactic(*tactic_fields),
    )


# ---------------------------------------------------------------------------
# Triton path (M <= 128)
# ---------------------------------------------------------------------------
#
# Each program owns BLOCK_N weight rows and one K split, and streams that slice
# of the weight from HBM once. Few tokens take the FMA path (weights straight
# into registers, a [BLOCK_M, BLOCK_N, BLOCK_K] FP32 accumulator); more tokens
# the swap-AB MMA path (acc[BLOCK_N, BLOCK_M] = W_tile @ X^T, tokens padded to
# BLOCK_M >= 16, ACC independent accumulator chains). With SPLIT_K > 1 the
# partials go to an FP32 workspace and the last program to finish a tile sums
# all of them in split order, so the result does not depend on arrival order.


def _triton_buckets(
    configs: dict, *, n: int, k: int, k0: int, n0: int, n1: int
) -> frozenset[int]:
    return frozenset(
        bucket
        for bucket, cfg in configs.items()
        if _triton_cfg_fits(cfg, n=n, k=k, k0=k0, n0=n0, n1=n1)
    )


def _triton_cfg_fits(cfg, *, n: int, k: int, k0: int, n0: int, n1: int) -> bool:
    # Output regions and the K segment boundary on tile boundaries, whole
    # BLOCK_K steps per split, 16-byte aligned k chunks per accumulator chain.
    block_n, block_k, split_k, _, _, acc, _ = cfg
    return (
        n % block_n == 0
        and n0 % block_n == 0
        and n1 % block_n == 0
        and n // block_n <= _MAX_TILES
        and k % split_k == 0
        and (k // split_k) % block_k == 0
        and k0 % block_k == 0
        and block_k % acc == 0
        and (block_k // acc) % 8 == 0
    )


def _launch_triton(
    w, x0, x1, outs, counters, *, k0: int, n0: int, n1: int, cfg
) -> None:
    block_n, block_k, split_k, num_warps, num_stages, acc, fma = cfg
    m = x0.shape[0]
    n, k = w.shape
    block_m = triton.next_power_of_2(m) if fma else max(16, triton.next_power_of_2(m))
    n_tiles = n // block_n
    workspace = (
        torch.empty(
            n_tiles * split_k * block_n * block_m, dtype=torch.float32, device=w.device
        )
        if split_k > 1
        else counters
    )
    _skinny_gemm_kernel[(split_k, n_tiles)](
        w,
        x0,
        x1,
        outs[0],
        outs[1],
        outs[2],
        workspace,
        counters,
        m,
        # Above 16 tokens the token mask is static: one kernel per captured
        # batch size, ~15 -> ~12.7 us for the front GEMM at M = 17 and 24.
        M_CT=m if m > 16 else 0,
        N=n,
        K=k,
        K0=k0,
        N0=n0,
        N1=n1,
        BLOCK_N=block_n,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        SPLIT_K=split_k,
        ACC=acc,
        FMA=fma,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@triton.jit
def _w_tile_ptrs(w_ptr, row_off, k, BKS: tl.constexpr):
    k = tl.multiple_of(k, 8)
    return w_ptr + row_off[:, None] + (k + tl.arange(0, BKS))[None, :]


@triton.jit
def _mma_k_step(
    w_ptr,
    row_off,
    x_ptr,
    x_ld,
    xk,
    k,
    offs_m,
    m_mask,
    acc0,
    acc1,
    acc2,
    acc3,
    BKS: tl.constexpr,
    ACC: tl.constexpr,
):
    # ACC chains of BKS columns each overlap in the tensor pipe instead of one
    # serial accumulate chain over all of BLOCK_K.
    offs_k = tl.arange(0, BKS)
    xp = x_ptr + offs_m[None, :] * x_ld + (xk + offs_k)[:, None]
    w0 = tl.load(_w_tile_ptrs(w_ptr, row_off, k, BKS), eviction_policy="evict_first")
    x0 = tl.load(xp, mask=m_mask[None, :], other=0.0, eviction_policy="evict_last")
    if ACC >= 2:
        w1 = tl.load(
            _w_tile_ptrs(w_ptr, row_off, k + BKS, BKS), eviction_policy="evict_first"
        )
        x1 = tl.load(
            xp + BKS, mask=m_mask[None, :], other=0.0, eviction_policy="evict_last"
        )
    if ACC >= 4:
        w2 = tl.load(
            _w_tile_ptrs(w_ptr, row_off, k + 2 * BKS, BKS),
            eviction_policy="evict_first",
        )
        x2 = tl.load(
            xp + 2 * BKS, mask=m_mask[None, :], other=0.0, eviction_policy="evict_last"
        )
        w3 = tl.load(
            _w_tile_ptrs(w_ptr, row_off, k + 3 * BKS, BKS),
            eviction_policy="evict_first",
        )
        x3 = tl.load(
            xp + 3 * BKS, mask=m_mask[None, :], other=0.0, eviction_policy="evict_last"
        )
    acc0 = tl.dot(w0, x0, acc0)
    if ACC >= 2:
        acc1 = tl.dot(w1, x1, acc1)
    if ACC >= 4:
        acc2 = tl.dot(w2, x2, acc2)
        acc3 = tl.dot(w3, x3, acc3)
    return acc0, acc1, acc2, acc3


@triton.jit
def _mma_loop(
    w_ptr,
    row_off,
    x0_ptr,
    x1_ptr,
    M,
    k_begin,
    k_mid,
    k_end,
    K: tl.constexpr,
    K0: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC: tl.constexpr,
):
    BKS: tl.constexpr = BLOCK_K // ACC
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    acc0 = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
    for k in range(k_begin, k_mid, BLOCK_K):
        acc0, acc1, acc2, acc3 = _mma_k_step(
            w_ptr, row_off, x0_ptr, K0, k, k, offs_m, m_mask,
            acc0, acc1, acc2, acc3, BKS, ACC,
        )  # fmt: skip
    if K0 < K:
        for k in range(k_mid, k_end, BLOCK_K):
            acc0, acc1, acc2, acc3 = _mma_k_step(
                w_ptr, row_off, x1_ptr, K - K0, k - K0, k, offs_m, m_mask,
                acc0, acc1, acc2, acc3, BKS, ACC,
            )  # fmt: skip
    acc = acc0
    if ACC >= 2:
        acc += acc1
    if ACC >= 4:
        acc += acc2 + acc3
    return acc


@triton.jit
def _fma_k_step(
    w_ptr, row_off, x_ptr, x_ld, xk, k, offs_m, m_mask, acc3, BLOCK_K: tl.constexpr
):
    w = tl.load(
        _w_tile_ptrs(w_ptr, row_off, k, BLOCK_K), eviction_policy="evict_first"
    ).to(tl.float32)
    xp = x_ptr + offs_m[:, None] * x_ld + (xk + tl.arange(0, BLOCK_K))[None, :]
    xv = tl.load(xp, mask=m_mask[:, None], other=0.0, eviction_policy="evict_last").to(
        tl.float32
    )
    return acc3 + xv[:, None, :] * w[None, :, :]


@triton.jit
def _fma_loop(
    w_ptr,
    row_off,
    x0_ptr,
    x1_ptr,
    M,
    k_begin,
    k_mid,
    k_end,
    K: tl.constexpr,
    K0: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    acc3 = tl.zeros([BLOCK_M, BLOCK_N, BLOCK_K], dtype=tl.float32)
    for k in range(k_begin, k_mid, BLOCK_K):
        acc3 = _fma_k_step(
            w_ptr, row_off, x0_ptr, K0, k, k, offs_m, m_mask, acc3, BLOCK_K
        )
    if K0 < K:
        for k in range(k_mid, k_end, BLOCK_K):
            acc3 = _fma_k_step(
                w_ptr, row_off, x1_ptr, K - K0, k - K0, k, offs_m, m_mask, acc3, BLOCK_K
            )
    return tl.trans(tl.sum(acc3, axis=2))


@triton.jit
def _store_out(
    acc,
    offs_n,
    offs_m,
    m_mask,
    o0_ptr,
    o1_ptr,
    o2_ptr,
    N: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    n_start,
):
    # Region boundaries are multiples of BLOCK_N (_triton_cfg_fits), so a tile
    # belongs to exactly one output.
    if n_start < N0:
        p0 = o0_ptr + offs_m[None, :] * N0 + offs_n[:, None]
        tl.store(p0, acc.to(o0_ptr.dtype.element_ty), mask=m_mask[None, :])
    elif n_start < N0 + N1:
        p1 = o1_ptr + offs_m[None, :] * N1 + (offs_n - N0)[:, None]
        tl.store(p1, acc.to(o1_ptr.dtype.element_ty), mask=m_mask[None, :])
    else:
        p2 = o2_ptr + offs_m[None, :] * (N - N0 - N1) + (offs_n - N0 - N1)[:, None]
        tl.store(p2, acc.to(o2_ptr.dtype.element_ty), mask=m_mask[None, :])


@triton.jit(do_not_specialize=["M_RT"])
def _skinny_gemm_kernel(
    w_ptr,  # [N, K] BF16, row-major
    x0_ptr,  # [M, K0] BF16: columns [0, K0) of the activation
    x1_ptr,  # [M, K - K0] BF16: columns [K0, K); x0_ptr when K0 == K
    o0_ptr,  # [M, N0]: weight rows [0, N0)
    o1_ptr,  # [M, N1]: weight rows [N0, N0 + N1)
    o2_ptr,  # [M, N - N0 - N1]: weight rows [N0 + N1, N)
    ws_ptr,  # FP32 [n_tiles, SPLIT_K, BLOCK_N, BLOCK_M]; unused when SPLIT_K == 1
    cnt_ptr,  # int32 [n_tiles], zero between launches
    M_RT,
    M_CT: tl.constexpr,  # token count when > 0, else M_RT
    N: tl.constexpr,
    K: tl.constexpr,
    K0: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ACC: tl.constexpr,
    FMA: tl.constexpr,
):
    if M_CT > 0:
        M = M_CT
    else:
        M = M_RT
    K_SPLIT: tl.constexpr = K // SPLIT_K
    # K split fastest: neighbouring programs read neighbouring weight blocks.
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    row_off = offs_n * K

    k_begin = pid_k * K_SPLIT
    k_end = k_begin + K_SPLIT
    k_mid = tl.minimum(tl.maximum(k_begin, K0), k_end)
    if FMA:
        acc = _fma_loop(
            w_ptr, row_off, x0_ptr, x1_ptr, M, k_begin, k_mid, k_end,
            K, K0, BLOCK_N, BLOCK_M, BLOCK_K,
        )  # fmt: skip
    else:
        acc = _mma_loop(
            w_ptr, row_off, x0_ptr, x1_ptr, M, k_begin, k_mid, k_end,
            K, K0, BLOCK_N, BLOCK_M, BLOCK_K, ACC,
        )  # fmt: skip

    if SPLIT_K > 1:
        tile_elems: tl.constexpr = BLOCK_N * BLOCK_M
        loc = tl.arange(0, BLOCK_N)[:, None] * BLOCK_M + offs_m[None, :]
        ws_tile = ws_ptr + pid_n * (SPLIT_K * tile_elems)
        tl.store(ws_tile + pid_k * tile_elems + loc, acc, cache_modifier=".cg")
        # Every thread's partial is stored before the release below.
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            # Start from zero and add every partial in split order, the last
            # arriver's own one included, so the sum order is fixed.
            acc = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
            for s in tl.static_range(SPLIT_K):
                acc += tl.load(ws_tile + s * tile_elems + loc, cache_modifier=".cg")
            tl.atomic_xchg(cnt_ptr + pid_n, 0, sem="relaxed", scope="gpu")
            _store_out(
                acc, offs_n, offs_m, m_mask, o0_ptr, o1_ptr, o2_ptr,
                N, N0, N1, pid_n * BLOCK_N,
            )  # fmt: skip
    else:
        _store_out(
            acc, offs_n, offs_m, m_mask, o0_ptr, o1_ptr, o2_ptr,
            N, N0, N1, pid_n * BLOCK_N,
        )  # fmt: skip
