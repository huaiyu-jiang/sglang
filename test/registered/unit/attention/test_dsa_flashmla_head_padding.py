"""Dispatch, head padding and configuration guards for sparse FlashMLA prefill."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFlashMLASparseHeadPadding(CustomTestCase):
    def run_case(
        self,
        heads,
        dim=576,
        enabled=True,
        topk=128,
        allow_small_heads=True,
        kv_dtype=torch.bfloat16,
        token_packing=False,
        allow_token_packing=False,
        tokens=3,
        packed_variants=None,
    ):
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        q = torch.randn(tokens, heads, dim, dtype=torch.bfloat16)
        sink = torch.randn(heads)
        indices = torch.zeros(tokens, topk, dtype=torch.int32)
        lengths = torch.full((tokens,), topk, dtype=torch.int32)
        calls = []

        def small(padded, kv, ids, scale, active, sinks, lens):
            calls.append("small")
            self.assertEqual(active, 8 if heads <= 8 else 32)
            self.assertEqual(padded.shape, (tokens, 64, dim))
            torch.testing.assert_close(padded[:, :heads], q)
            self.assertEqual(torch.count_nonzero(padded[:, heads:]).item(), 0)
            torch.testing.assert_close(sinks[:heads], sink)
            self.assertEqual(torch.count_nonzero(sinks[heads:]).item(), 0)
            self.assertIs(lens, lengths)
            torch.testing.assert_close(ids, indices.unsqueeze(1))
            return torch.ones(tokens, active, 512), None, None

        def native(**kwargs):
            calls.append("native")
            padded = kwargs["q"].shape[1]
            self.assertEqual(padded, 128)
            torch.testing.assert_close(kwargs["q"][:, :heads], q)
            self.assertIs(kwargs["topk_length"], lengths)
            return torch.ones(tokens, padded, 512), None, None

        def packed(**kwargs):
            calls.append("packed")
            self.assertIs(kwargs["q"], q)
            self.assertIs(kwargs["topk_length"], lengths)
            return torch.ones(tokens, heads, 512)

        backend = SimpleNamespace(
            device_sm_major=10,
            sparse_prefill_small_heads=enabled,
            flashmla_sparse_token_packing=token_packing,
            dsa_index_topk=topk,
            _sink_pad_cache={},
            _token_pack_variants=set() if packed_variants is None else packed_variants,
            _call_flash_mla_sparse_fwd_small_heads=small,
            _call_flash_mla_sparse_fwd=native,
        )
        module = "sglang.kernels.ops.attention.dsa.flashmla_sparse_token_pack_sm90"
        with (
            mock.patch(f"{module}.token_pack_supported", return_value=True),
            mock.patch(f"{module}.MIN_PROFITABLE_TOKENS", 8),
            mock.patch(f"{module}.flashmla_sparse_fwd_token_packed", packed),
        ):
            out = DeepseekSparseAttnBackend._forward_flashmla_sparse(
                backend,
                q,
                torch.empty(1, dtype=kv_dtype),
                512,
                indices,
                0.5,
                lengths,
                sink,
                allow_small_heads=allow_small_heads,
                allow_token_packing=allow_token_packing,
            )
        self.assertEqual(out.shape, (tokens, heads, 512))
        return calls

    def test_all_small_head_counts_dispatch_automatically(self):
        for heads in range(1, 33):
            for dim in (512, 576):
                with self.subTest(heads=heads, dim=dim):
                    self.assertEqual(self.run_case(heads, dim), ["small"])

    def test_native_fallback_preserves_padding(self):
        for heads in (8, 16, 32, 64, 128):
            with self.subTest(heads=heads):
                self.assertEqual(self.run_case(heads, enabled=False), ["native"])
        for heads in (64, 128):
            self.assertEqual(self.run_case(heads), ["native"])
        self.assertEqual(self.run_case(8, enabled=False), ["native"])
        for topk in (64, 130):
            self.assertEqual(self.run_case(8, topk=topk), ["native"])

    def test_decode_and_kv_dtype_stay_native(self):
        self.assertEqual(self.run_case(8, allow_small_heads=False), ["native"])
        self.assertEqual(self.run_case(8, kv_dtype=torch.float16), ["native"])

    def test_token_packing_only_for_plain_extend(self):
        # Packing needs both the startup gate and a plain-extend caller;
        # decode, verify and draft-extend keep the padded call.
        for token_packing, allow, expected in (
            (True, True, ["packed"]),
            (True, False, ["small"]),
            (False, True, ["small"]),
        ):
            with self.subTest(token_packing=token_packing, allow=allow):
                calls = self.run_case(
                    16,
                    token_packing=token_packing,
                    allow_token_packing=allow,
                    tokens=8,
                )
                self.assertEqual(calls, expected)

    def test_token_packing_short_batches_compile_once(self):
        # Below the crossover the padded call is faster, but the first call
        # per variant still packs so that the startup warmup compiles it.
        variants = set()
        case = dict(token_packing=True, allow_token_packing=True)
        for tokens, expected in (
            (3, "packed"),
            (3, "small"),
            (7, "small"),
            (8, "packed"),
        ):
            with self.subTest(tokens=tokens):
                calls = self.run_case(
                    16, tokens=tokens, packed_variants=variants, **case
                )
                self.assertEqual(calls, [expected])

    def test_token_packing_startup_gate(self):
        from sglang.srt.layers.attention import dsa_backend

        gate = dsa_backend._sm90_prefill_token_packing_enabled
        with mock.patch.object(dsa_backend, "_is_cuda", True):
            self.assertTrue(gate(device_sm_major=9, deterministic=False))
            # Packed outputs depend on tile neighbours: not batch invariant.
            self.assertFalse(gate(device_sm_major=9, deterministic=True))
            self.assertFalse(gate(device_sm_major=10, deterministic=False))

    def test_startup_configuration(self):
        from sglang.srt.layers.attention.dsa_backend import _small_heads_prefill_enabled

        args = [10, torch.bfloat16, 8, 512, 64, "flashmla_sparse"]
        self.assertTrue(_small_heads_prefill_enabled(*args))
        args[-1] = "flashmla_auto"
        self.assertTrue(_small_heads_prefill_enabled(*args))
        for index, value in [
            (0, 9),
            (1, torch.float16),
            (2, 64),
            (3, 256),
            (4, 128),
            (5, "trtllm"),
        ]:
            invalid = list(args)
            invalid[index] = value
            self.assertFalse(_small_heads_prefill_enabled(*invalid))


if __name__ == "__main__":
    unittest.main()
