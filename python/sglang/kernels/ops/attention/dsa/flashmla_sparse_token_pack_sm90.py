"""SM90 sparse-MLA prefill for 16 TP-local heads by token packing.

FlashMLA's SM90 sparse prefill only runs 64 query heads, so a 16-head layer is
zero-padded to 64 and 75% of the kernel's work is padding. Here 4 consecutive
tokens share one 64-head FlashMLA row instead: ``q[T, 16, 576]`` is viewed as
``[T / 4, 64, 576]`` without a copy. A packed row carries one index list, so
it holds only the keys all 4 tokens attend to (the "common" keys); each token's
other keys ("extras") and the attention sink are merged in Triton with an exact
log-sum-exp combine.

The split is exact for any index layout; only its speed depends on the layout.
Per tile, each row's column shift against the tile's reference row is searched
among its first 8 valid columns. A reference column is common when every
participating row holds the same slot at the shifted column, and the coverage
map from row columns to common columns is injective, so every valid column of
every row is counted once (duplicates included). Rows whose shift is not found
(another request inside the tile) take all their keys as extras. For sliding
windows, which shift by one column per token, a tile has 509 common keys of 512
and each token at most 3 extras. Rows with more than 32 extras (the leading
``T % 4`` rows, rows that straddle requests, layouts without shift structure)
are merged by a separate grid-stride launch; per row it costs less than the
padded FlashMLA call, so above ``MIN_PROFITABLE_TOKENS`` no layout is slower
than the padded path.

Kernels per call: pack/split (Triton), FlashMLA on ``[T / 4, 64]`` rows with
``topk_length`` = the common count and no sink, short-row merge (one program
per row), long-row merge. Scratch is one ``torch.empty`` per call on the
current stream; nothing persists between calls and the host never syncs, so
the call is safe on concurrent streams and under CUDA-graph capture.

A token's output depends on which tokens share its tile (summation order and
one extra bf16 rounding of the common part), so it is not batch invariant.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

D_QK = tl.constexpr(576)
D_V = tl.constexpr(512)
D_ROPE = tl.constexpr(64)
LOG2E = 1.4426950408889634

# FlashMLA's SM90 sparse prefill runs 64 query heads per row.
_FLASHMLA_HEADS = 64
# Measured on 16 heads only; 32 heads spill registers in both merge kernels.
_SUPPORTED_HEADS = 16
# The pack kernel holds a power-of-two row of the index width per program;
# only the 512-wide sliding-window rows were measured.
_MAX_WIDTH = 512
# The packed call costs ~90 us more host time than the padded one (136 vs
# 48 us on our H200 host), and Hy4-Flash's eager prefill there is host-bound
# up to ~4k tokens (TTFT 144-162 ms from 512 to 4096), so smaller batches only
# pay the host cost. Measured TTFT: 4096 tokens +3% packed, 4608 tokens -6%.
MIN_PROFITABLE_TOKENS = 4608
# How many leading valid columns the column-shift search looks at.
_SHIFT_SEARCH = 8
# Rows with more than this many extras go to the long-row launch.
_LONG_ROW_EXTRAS = 32
# Long-row launch: programs per SM. H200: one 512-key row 22 us, ~5.9k long
# rows 624 us (1 program of 8 warps per SM: 30 us and 1167 us).
_LONG_ROW_PROGRAMS_PER_SM = 2

_num_sms: dict = {}


@triton.jit
def _row_info(idx_ptr, len_ptr, t, live, T, first, W: tl.constexpr, DM: tl.constexpr):
    # (clamped row, valid length n (0 for a missing row), column shift d vs
    # the reference row, whether the row takes part in the common keys).
    tc = tl.minimum(t, T - 1)
    n = tl.where(live, tl.minimum(tl.load(len_ptr + tc), W), 0)
    dar = tl.arange(0, DM)
    head = tl.load(idx_ptr + tc.to(tl.int64) * W + dar, mask=dar < n, other=-2)
    hit = (head == first) & (first >= 0)
    d = tl.min(tl.where(hit, dar, DM), axis=0)
    part = (n > 0) & ((d < DM) | (first < 0))
    d = tl.where(d == DM, 0, d)
    return tc, n, d, part


@triton.jit(
    do_not_specialize=[
        "o_pidx",
        "o_pcom",
        "o_plen",
        "o_eidx",
        "o_ecnt",
        "o_ucom",
        "o_nlong",
        "T",
        "G",
        "off",
    ]
)
def _pack_split_kernel(
    idx_ptr,  # [T, W] int32
    len_ptr,  # [T] int32, clamped to W here
    ws_ptr,  # int32 scratch; regions at the o_* element offsets:
    o_pidx,  # [G, W] packed index rows, common slots compacted to the front
    o_pcom,  # [G, W] common slots at their reference columns, -1 elsewhere
    o_plen,  # [G] packed topk_length (common count)
    o_eidx,  # [T, W] extras slot ids, compacted
    o_ecnt,  # [T] extras count
    o_ucom,  # [T] 1 if the row uses its tile's common part
    o_nlong,  # long-row counter, zeroed here
    T,
    G,
    off,  # rows [0, off) form the partial tile; full tile g starts at off + g * P
    W: tl.constexpr,
    BW: tl.constexpr,
    P: tl.constexpr,
    DM: tl.constexpr,
):
    pidx_ptr = ws_ptr + o_pidx
    pcom_ptr = ws_ptr + o_pcom
    eidx_ptr = ws_ptr + o_eidx
    g = tl.program_id(0)
    if g == 0:
        # Stream-ordered before the short-row merge appends to the counter.
        tl.store(ws_ptr + o_nlong, 0)
    full = g < G
    t0 = tl.where(full, off + g * P, 0)
    nrows = tl.where(full, P, off)
    cols = tl.arange(0, BW)
    inw = cols < W

    # Reference row: the last row of the tile with a non-empty valid range.
    ref = t0
    for r in tl.static_range(P):
        t = t0 + r
        n = tl.load(len_ptr + tl.minimum(t, T - 1))
        ref = tl.where((r < nrows) & (n > 0), t, ref)
    n_ref = tl.minimum(tl.load(len_ptr + ref), W)
    r_ref = tl.load(idx_ptr + ref.to(tl.int64) * W + cols, mask=inw, other=-1)
    v_ref = inw & (cols < n_ref) & (r_ref >= 0)
    first = tl.where(n_ref > 0, tl.load(idx_ptr + ref.to(tl.int64) * W), -1)

    common = v_ref & full
    for r in tl.static_range(P):
        tc, n, d, part = _row_info(idx_ptr, len_ptr, t0 + r, r < nrows, T, first, W, DM)
        j = cols + d
        ok = j < n
        rv = tl.load(idx_ptr + tc.to(tl.int64) * W + j, mask=ok & part, other=-1)
        common = common & ((~part) | (ok & (rv == r_ref)))

    # FlashMLA's time follows topk_length, so the common keys are compacted.
    # When they form a prefix (sliding windows) the compacted row is the
    # positional row and the cover test below reads it; else pcom holds it.
    ci = common.to(tl.int32)
    ncom = tl.sum(ci, axis=0)
    pre = ncom == tl.max(tl.where(common, cols + 1, 0), axis=0)
    prow = g.to(tl.int64) * W
    if full:
        if pre:
            tl.store(pidx_ptr + prow + cols, tl.where(common, r_ref, -1), mask=inw)
        else:
            tl.store(pcom_ptr + prow + cols, tl.where(common, r_ref, -1), mask=inw)
            tl.store(pidx_ptr + prow + tl.cumsum(ci, axis=0) - 1, r_ref, mask=common)
            tl.store(pidx_ptr + prow + cols, -1, mask=inw & (cols >= ncom))
        tl.store(ws_ptr + o_plen + g, ncom)
    cov_ptr = ws_ptr + tl.where(pre, o_pidx, o_pcom) + prow
    # The common row is re-read at shifted columns below.
    tl.debug_barrier()

    # Extras of row r: valid columns j whose image j - d_r is not common.
    for r in tl.static_range(P):
        if r < nrows:
            t = t0 + r
            tc, n_r, d_r, part = _row_info(
                idx_ptr, len_ptr, t, r < nrows, T, first, W, DM
            )
            row = tl.load(idx_ptr + t.to(tl.int64) * W + cols, mask=inw, other=-1)
            v_r = (cols < n_r) & (row >= 0)
            c = cols - d_r
            okc = (c >= 0) & (c < W) & full
            pc = tl.load(cov_ptr + c, mask=okc, other=-1)
            ex = v_r & ~(okc & (pc >= 0) & part)
            exi = ex.to(tl.int32)
            pos = tl.cumsum(exi, axis=0) - 1
            tl.store(eidx_ptr + t.to(tl.int64) * W + pos, row, mask=ex)
            tl.store(ws_ptr + o_ecnt + t, tl.sum(exi, axis=0))
            tl.store(ws_ptr + o_ucom + t, (part & full).to(tl.int32))


@triton.jit
def _extras_merge_row(
    t,
    cnt,
    q_ptr,
    kv_ptr,
    eidx_ptr,
    ucom_ptr,
    oc_ptr,
    lse_ptr,
    ml_ptr,
    sink_ptr,
    out_ptr,
    off,
    qk_scale2,
    W: tl.constexpr,
    H: tl.constexpr,
    BN: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    # Softmax over the row's extras, merged with its tile's common part and the sink.
    t64 = t.to(tl.int64)
    hs = tl.arange(0, H)
    dv = tl.arange(0, D_V)
    dr = tl.arange(0, D_ROPE)
    ns = tl.arange(0, BN)
    if HAS_SINK:
        sink2 = tl.load(sink_ptr + hs) * 1.4426950408889634
    has_c = tl.load(ucom_ptr + t) != 0
    ci = (t - off).to(tl.int64)
    lse2 = tl.load(lse_ptr + ci * H + hs, mask=has_c & (hs >= 0), other=float("inf"))
    mlc = tl.load(ml_ptr + ci * H + hs, mask=has_c & (hs >= 0), other=float("-inf"))
    oc = tl.load(
        oc_ptr + ci * (H * D_V) + hs[:, None] * D_V + dv[None, :],
        mask=(hs[:, None] >= 0) & has_c,
        other=0.0,
    )

    m_e = tl.full([H], float("-inf"), tl.float32)
    l_e = tl.zeros([H], tl.float32)
    acc = tl.zeros([H, D_V], tl.float32)
    qb = q_ptr + t64 * (H * D_QK) + hs[:, None] * D_QK
    for n0 in range(0, cnt, BN):
        q_n = tl.load(qb + dv[None, :])
        q_r = tl.load(qb + D_V + dr[None, :])
        nm = (n0 + ns) < cnt
        kid = tl.load(eidx_ptr + t64 * W + n0 + ns, mask=nm, other=0).to(tl.int64)
        kb = kv_ptr + kid[:, None] * D_QK
        k_n = tl.load(kb + dv[None, :], mask=nm[:, None], other=0.0)
        k_r = tl.load(kb + D_V + dr[None, :], mask=nm[:, None], other=0.0)
        s = tl.dot(q_n, tl.trans(k_n)) + tl.dot(q_r, tl.trans(k_r))
        s = s * qk_scale2
        s = tl.where(nm[None, :], s, float("-inf"))
        m_new = tl.maximum(m_e, tl.max(s, axis=1))
        alpha = tl.exp2(m_e - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_e = l_e * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k_n)
        m_e = m_new

    # FlashMLA reports an empty common part as max logit -inf / lse +inf.
    emp = (mlc == float("-inf")) | (lse2 == float("inf"))
    L = tl.where(emp, float("-inf"), lse2)
    if HAS_SINK:
        M = tl.maximum(tl.maximum(L, m_e), sink2)
        wc = tl.exp2(L - M)
        we = tl.exp2(m_e - M)
        den = wc + we * l_e + tl.exp2(sink2 - M)
    else:
        # A row without any valid key gets 0, as FlashMLA returns.
        M = tl.maximum(L, m_e)
        M = tl.where(M == float("-inf"), 0.0, M)
        wc = tl.exp2(L - M)
        we = tl.exp2(m_e - M)
        den = wc + we * l_e
        den = tl.where(den == 0.0, 1.0, den)
    o = (wc[:, None] * oc.to(tl.float32) + we[:, None] * acc) / den[:, None]
    tl.store(
        out_ptr + t64 * (H * D_V) + hs[:, None] * D_V + dv[None, :],
        o.to(tl.bfloat16),
    )


@triton.jit(
    do_not_specialize=[
        "o_eidx",
        "o_ecnt",
        "o_ucom",
        "o_lrows",
        "o_nlong",
        "T",
        "off",
    ]
)
def _extras_merge_kernel(
    q_ptr,  # [T, H, 576] bf16
    kv_ptr,  # [S, 576] bf16
    ws_ptr,  # int32 scratch: eidx, ecnt, ucom (pack kernel), lrows, nlong
    o_eidx,
    o_ecnt,
    o_ucom,
    o_lrows,  # [T] ids of rows with more than THR extras, appended by LONG=False
    o_nlong,
    oc_ptr,  # [T - off, H, 512] bf16, common-part output of rows off.. (no sink)
    lse_ptr,  # [T - off, H] fp32, common-part log2-sum-exp2
    ml_ptr,  # [T - off, H] fp32, common-part max logit
    sink_ptr,  # [H] fp32, read only when HAS_SINK
    out_ptr,  # [T, H, 512] bf16
    T,
    off,
    qk_scale2,  # sm_scale * log2(e)
    W: tl.constexpr,
    H: tl.constexpr,
    BN: tl.constexpr,
    THR: tl.constexpr,
    LONG: tl.constexpr,  # False: one program per row; True: grid-stride over long rows
    HAS_SINK: tl.constexpr,
):
    eidx_ptr = ws_ptr + o_eidx
    ecnt_ptr = ws_ptr + o_ecnt
    ucom_ptr = ws_ptr + o_ucom
    lrows_ptr = ws_ptr + o_lrows
    nlong_ptr = ws_ptr + o_nlong
    if LONG:
        nl = tl.minimum(tl.load(nlong_ptr), T)
        for k in range(tl.program_id(0), nl, tl.num_programs(0)):
            t = tl.load(lrows_ptr + k)
            if (t >= 0) & (t < T):
                cnt = tl.load(ecnt_ptr + t)
                _extras_merge_row(
                    t,
                    cnt,
                    q_ptr,
                    kv_ptr,
                    eidx_ptr,
                    ucom_ptr,
                    oc_ptr,
                    lse_ptr,
                    ml_ptr,
                    sink_ptr,
                    out_ptr,
                    off,
                    qk_scale2,
                    W,
                    H,
                    BN,
                    HAS_SINK,
                )
    else:
        t = tl.program_id(0)
        cnt = tl.load(ecnt_ptr + t)
        if cnt <= THR:
            _extras_merge_row(
                t,
                cnt,
                q_ptr,
                kv_ptr,
                eidx_ptr,
                ucom_ptr,
                oc_ptr,
                lse_ptr,
                ml_ptr,
                sink_ptr,
                out_ptr,
                off,
                qk_scale2,
                W,
                H,
                BN,
                HAS_SINK,
            )
        else:
            slot = tl.atomic_add(nlong_ptr, 1)
            tl.store(lrows_ptr + slot, t, mask=slot < T)


def _multi_processor_count(device: torch.device) -> int:
    n = _num_sms.get(device)
    if n is None:
        n = torch.cuda.get_device_properties(device).multi_processor_count
        _num_sms[device] = n
    return n


def _scratch(device: torch.device, sizes: list[int]):
    # One int32 allocation per call, regions 128-byte aligned; returns the
    # buffer and each region's element offset.
    offsets, total = [], 0
    for n in sizes:
        offsets.append(total)
        total += (n + 31) // 32 * 32
    return torch.empty((total,), dtype=torch.int32, device=device), offsets


def token_pack_supported(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    d_v: int,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> bool:
    """Whether ``flashmla_sparse_fwd_token_packed`` serves these inputs."""
    return (
        q.dim() == 3
        and q.shape[0] > 0
        and q.shape[1] == _SUPPORTED_HEADS
        and q.shape[2] == D_QK
        and q.dtype == torch.bfloat16
        and q.is_contiguous()
        and kv.dtype == torch.bfloat16
        and kv.shape[-1] == D_QK
        and kv.numel() == kv.shape[0] * D_QK
        and kv.is_contiguous()
        and indices.dim() == 2
        and indices.dtype == torch.int32
        and indices.shape[0] == q.shape[0]
        and indices.shape[1] <= _MAX_WIDTH
        and indices.is_contiguous()
        and d_v == D_V
        and (
            attn_sink is None
            or (
                attn_sink.shape == (_SUPPORTED_HEADS,)
                and attn_sink.dtype == torch.float32
                and attn_sink.is_contiguous()
            )
        )
        and (
            topk_length is None
            or (
                topk_length.shape == (q.shape[0],)
                and topk_length.dtype == torch.int32
                and topk_length.is_contiguous()
            )
        )
    )


def flashmla_sparse_fwd_token_packed(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sparse-MLA prefill for 16 local heads on SM90; same result as
    ``flash_mla_sparse_fwd`` on ``q`` zero-padded to 64 heads.

    q           [T, 16, 576] bf16, contiguous (512 latent + 64 rope), T > 0.
    kv          [S, 576] or [S, 1, 576] bf16, contiguous; V is kv[:, :512].
    indices     [T, W] int32 slot ids, W <= 512, contiguous; -1 is skipped.
    attn_sink   [16] fp32 natural-log logits added to the softmax denominator,
                or None.
    topk_length [T] int32 valid prefix per row (clamped to W), or None for W.
    Returns     [T, 16, 512] bf16.

    Inputs outside ``token_pack_supported`` are rejected by assert. Every
    column is an independent key: a slot named twice in a row counts twice.
    """
    assert token_pack_supported(q, kv, indices, D_V, attn_sink, topk_length), (
        "unsupported inputs"
    )
    T, H, _ = q.shape
    W = indices.shape[1]
    P = _FLASHMLA_HEADS // H
    G = T // P
    # The partial tile goes first, so q[off:] views as [G, 64, 576].
    off = T - G * P
    device = q.device
    if topk_length is None:
        topk_length = torch.full((T,), W, dtype=torch.int32, device=device)
    ws, (o_pidx, o_pcom, o_plen, o_eidx, o_ecnt, o_ucom, o_lrows, o_nlong) = _scratch(
        device, [G * W, G * W, G, T * W, T, T, T, 1]
    )
    _pack_split_kernel[(G + (1 if off else 0),)](
        indices,
        topk_length,
        ws,
        o_pidx,
        o_pcom,
        o_plen,
        o_eidx,
        o_ecnt,
        o_ucom,
        o_nlong,
        T,
        G,
        off,
        W=W,
        BW=triton.next_power_of_2(W),
        P=P,
        DM=_SHIFT_SEARCH,
        num_warps=4,
    )
    if G > 0:
        from sglang.kernels.ops.attention.flash_mla_interface import (
            flash_mla_sparse_fwd,
        )

        oc, ml, lse = flash_mla_sparse_fwd(
            q[off:].view(G, _FLASHMLA_HEADS, D_QK),
            kv.view(-1, 1, D_QK),
            ws.as_strided((G, 1, W), (W, W, 1), o_pidx),
            sm_scale,
            D_V,
            None,
            ws.as_strided((G,), (1,), o_plen),
        )
    else:
        # No full tile: every row is all extras and never reads these.
        oc = torch.empty((1, H, D_V), dtype=torch.bfloat16, device=device)
        ml = lse = torch.empty((1, H), dtype=torch.float32, device=device)
    out = torch.empty((T, H, D_V), dtype=torch.bfloat16, device=device)
    has_sink = attn_sink is not None
    args = (
        q,
        kv,
        ws,
        o_eidx,
        o_ecnt,
        o_ucom,
        o_lrows,
        o_nlong,
        oc,
        lse,
        ml,
        attn_sink if has_sink else lse,
        out,
        T,
        off,
        float(sm_scale) * LOG2E,
    )
    # Tile configs measured on H200 at T = 7726 (short) and on 1 / 5.9k long rows.
    _extras_merge_kernel[(T,)](
        *args,
        W=W,
        H=H,
        BN=16,
        THR=_LONG_ROW_EXTRAS,
        LONG=False,
        HAS_SINK=has_sink,
        num_warps=2,
        num_stages=3,
    )
    _extras_merge_kernel[
        (min(T, _multi_processor_count(device) * _LONG_ROW_PROGRAMS_PER_SM),)
    ](
        *args,
        W=W,
        H=H,
        BN=64,
        THR=_LONG_ROW_EXTRAS,
        LONG=True,
        HAS_SINK=has_sink,
        num_warps=4,
        num_stages=3,
    )
    return out
