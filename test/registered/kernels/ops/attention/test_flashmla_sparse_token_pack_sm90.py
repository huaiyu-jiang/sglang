"""SM90 token-packed sparse-MLA prefill (16 local heads) against an fp32
reference: the sliding-window layouts the Hy4-Flash window layers produce, any
other index layout (the common/extras split must stay exact), CUDA-graph
replay, and one compiled variant per kernel across token counts.
"""

import unittest

import torch

from sglang.srt.utils import is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

D_QK, D_V, H, W, PAGE = 576, 512, 16, 512, 64
SM_SCALE = 256**-0.5
# Bound of FlashMLA's own error on these inputs (padded 64-head call vs fp32),
# measured over 3360 seeded runs on H200.
ATOL, RTOL = 8e-3, 1.6e-2


def _window_case(extends, prefixes=None, seed=0, pool_pages=512):
    """Paged KV pool and the window indices of Hy4-Flash's window layers:
    row p holds the slots of logical tokens max(p - 511, 0) .. p, then -1."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    prefixes = prefixes or [0] * len(extends)
    pages = torch.randperm(pool_pages - 1, generator=g) + 1
    used, rows, lens = 0, [], []
    for extend, prefix in zip(extends, prefixes):
        total = prefix + extend
        n_pages = (total + PAGE - 1) // PAGE
        req_pages = pages[used : used + n_pages]
        used += n_pages
        logical = torch.arange(total)
        slot = req_pages[logical // PAGE] * PAGE + logical % PAGE
        pos = torch.arange(prefix, total)
        tok = (pos - (W - 1)).clamp(min=0)[:, None] + torch.arange(W)[None, :]
        valid = tok <= pos[:, None]
        rows.append(torch.where(valid, slot[tok.clamp(max=total - 1)], -1))
        lens.append((pos + 1).clamp(max=W))
    indices = torch.cat(rows).to(torch.int32).cuda()
    topk_length = torch.cat(lens).to(torch.int32).cuda()
    T = indices.shape[0]
    gd = torch.Generator(device="cuda").manual_seed(seed + 1)
    kv = torch.randn(
        (pool_pages * PAGE, 1, D_QK), generator=gd, device="cuda"
    ).bfloat16()
    # Per-head logit std 1.0 .. 3.5, from diffuse to sharp heads.
    scale = torch.linspace(1.0, 3.5, H, device="cuda") / (SM_SCALE * D_QK**0.5)
    q = (
        torch.randn((T, H, D_QK), generator=gd, device="cuda") * scale[:, None]
    ).bfloat16()
    sink = (torch.randn((H,), generator=gd, device="cuda") * 1.5 + 0.8).clamp(-8, 6)
    sink[0], sink[-1] = -8.0, 6.0
    return q, kv, indices, topk_length, sink.float().contiguous()


def _reference(q, kv, indices, topk_length, sink):
    """fp32; every column below topk_length with id >= 0 is one key."""
    kvf = kv.view(-1, D_QK).float()
    cols = torch.arange(indices.shape[1], device="cuda")
    lens = (
        topk_length
        if topk_length is not None
        else torch.full_like(indices[:, 0], indices.shape[1])
    )
    out = torch.empty((q.shape[0], H, D_V), device="cuda")
    for s in range(0, q.shape[0], 256):
        e = min(q.shape[0], s + 256)
        ids = indices[s:e].long()
        valid = (ids >= 0) & (cols[None, :] < lens[s:e, None])
        k = kvf[ids.clamp(min=0)]
        logit = torch.einsum("nhd,nkd->nhk", q[s:e].float(), k) * SM_SCALE
        logit = logit.masked_fill(~valid[:, None, :], float("-inf"))
        m = logit.amax(-1, keepdim=True)
        if sink is not None:
            m = torch.maximum(m, sink[None, :, None])
        m = torch.nan_to_num(m, neginf=0.0)
        p = torch.exp(logit - m)
        den = p.sum(-1, keepdim=True)
        if sink is not None:
            den = den + torch.exp(sink[None, :, None] - m)
        out[s:e] = torch.einsum("nhk,nkd->nhd", p / den.clamp(min=1e-30), k[..., :D_V])
    return out


@unittest.skipIf(not is_sm90_supported(), "SM90 only")
class TestFlashMLASparseTokenPackSM90(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention.dsa import flashmla_sparse_token_pack_sm90

        cls.mod = flashmla_sparse_token_pack_sm90

    def _run(self, q, kv, indices, topk_length, sink):
        return self.mod.flashmla_sparse_fwd_token_packed(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=SM_SCALE,
            attn_sink=sink,
            topk_length=topk_length,
        )

    def _assert_close(self, out, ref, tag):
        err = (out.float() - ref).abs()
        bound = ATOL + RTOL * ref.abs()
        self.assertFalse(out.isnan().any().item(), tag)
        self.assertLessEqual((err / bound).max().item(), 1.0, tag)

    def test_window_layouts(self):
        # Shapes production hits: an 8k prompt, the 33-token startup warmup,
        # 1 and 3 tokens (no full tile), short prompts batched with tiles that
        # straddle requests, and prefix extends whose windows are full.
        cases = {
            "prefill_7700": dict(extends=[7700], pool_pages=256),
            "warmup_33": dict(extends=[33]),
            "single_1": dict(extends=[1]),
            "tail_3": dict(extends=[3]),
            "batch_short": dict(extends=[88] * 11 + [37, 66, 5]),
            "prefix_extends": dict(
                extends=[48] * 7 + [5, 2], prefixes=[512] * 7 + [2000, 33]
            ),
        }
        for tag, spec in cases.items():
            with self.subTest(tag):
                args = _window_case(**spec)
                self._assert_close(self._run(*args), _reference(*args), tag)

    def test_any_layout_is_exact(self):
        # The common/extras split depends on no index structure: rows without
        # a shift match (permuted), a slot named twice (counted twice), holes,
        # and topk_length shorter than the live entries.
        q, kv, indices, topk_length, sink = _window_case([256] * 3 + [61], seed=5)
        g = torch.Generator(device="cpu").manual_seed(9)
        T = indices.shape[0]
        idx_cpu, len_cpu = indices.cpu(), topk_length.cpu()
        layouts = {}
        permuted = idx_cpu.clone()
        for t in range(T):
            n = int(len_cpu[t])
            permuted[t, :n] = permuted[t, torch.randperm(n, generator=g)]
        layouts["permuted"] = (permuted, len_cpu)
        dup = idx_cpu.clone()
        dup[1::2, 1] = dup[1::2, 0]
        dup[2::4, 7] = dup[2::4, 3]
        layouts["duplicates"] = (dup, len_cpu)
        holes = idx_cpu.clone()
        holes[torch.rand(holes.shape, generator=g) < 0.1] = -1
        layouts["holes"] = (holes, len_cpu)
        cut = (len_cpu.float() * torch.rand(T, generator=g)).int().clamp(min=1)
        layouts["len_cut"] = (idx_cpu, cut)
        for tag, (idx, lens) in layouts.items():
            with self.subTest(tag):
                args = (q, kv, idx.cuda(), lens.cuda().contiguous(), sink)
                self._assert_close(self._run(*args), _reference(*args), tag)

    def test_no_sink_and_padded_rows(self):
        # DP / graph padding rows (all -1, topk_length 0) come out as 0, as
        # from FlashMLA; topk_length=None means every column of the row counts.
        q, kv, indices, topk_length, _ = _window_case([50, 9], seed=3)
        indices[-6:] = -1
        topk_length[-6:] = 0
        args = (q, kv, indices, topk_length, None)
        out = self._run(*args)
        self._assert_close(out, _reference(*args), "no sink")
        self.assertEqual(torch.count_nonzero(out[-6:]).item(), 0)
        args = (q, kv, indices, None, None)
        self._assert_close(self._run(*args), _reference(*args), "no topk_length")

    def test_cuda_graph_replay(self):
        # Scratch and the long-row counter belong to each call: one capture
        # replayed on new q and indices must not reuse an earlier replay's
        # long-row list. Permuting rows [0, 3T/4) and then [T/4, T) lists more
        # than T long rows over two replays, so a counter carried over from
        # the first replay would drop rows of the second.
        q, kv, indices, topk_length, sink = _window_case([400, 301], seed=11)
        g = torch.Generator(device="cpu").manual_seed(4)
        window = indices.clone()
        T = indices.shape[0]
        layouts = []
        for rows in (range(0, 3 * T // 4), range(T // 4, T)):
            idx = window.clone()
            for t in rows:
                n = int(topk_length[t])
                idx[t, :n] = idx[t, torch.randperm(n, generator=g).cuda()]
            layouts.append(idx)
        args = (q, kv, indices, topk_length, sink)
        self._run(*args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self._run(*args)
        for step in range(4):
            q.copy_(torch.randn_like(q, dtype=torch.float32).bfloat16() * 0.3)
            indices.copy_(layouts[step % 2])
            graph.replay()
            self._assert_close(out, _reference(*args), f"replay {step}")

    def test_one_variant_across_token_counts(self):
        # Token counts and scratch offsets are runtime values: a new batch
        # shape must not compile a new variant on the serving path.
        device = torch.cuda.current_device()
        kernels = (self.mod._pack_split_kernel, self.mod._extras_merge_kernel)
        self._run(*_window_case([40], seed=1))
        counts = [len(k.device_caches[device][0]) for k in kernels]
        for extends in ([1], [2], [3], [16], [17], [32, 1], [7, 26], [4097]):
            with self.subTest(extends=extends):
                self._run(*_window_case(extends, seed=2))
                self.assertEqual(
                    [len(k.device_caches[device][0]) for k in kernels], counts
                )


if __name__ == "__main__":
    unittest.main()
