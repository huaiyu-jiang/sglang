"""Tests for the warp split-K dense BF16 GEMM: K segments, N regions, AOT cache."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _supported_gpu() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    major, minor = torch.cuda.get_device_capability()
    return (major, minor) == (9, 0) or major == 10


if not _supported_gpu():
    pytest.skip("SM90 or SM10x required", allow_module_level=True)

from sglang.kernels.ops.gemm.dense_bf16_gemm_warp_splitk import (  # noqa: E402
    WarpSplitKTactic,
    run_warp_splitk_dense,
    run_warp_splitk_dense_multi_io,
)

_SENTINEL_ROWS = 3


def _inputs(m, k_widths, n, seed):
    torch.manual_seed(seed)
    activations = [
        torch.randn(m, width, dtype=torch.bfloat16, device="cuda") for width in k_widths
    ]
    k = sum(k_widths)
    weight = (torch.randn(n, k, device="cuda") / k**0.5).to(torch.bfloat16)
    return activations, weight


def _guarded_outputs(m, n_widths, out_dtypes):
    """Outputs as [M, N_j] views over buffers whose extra rows hold NaN, so a
    store past row M - 1 is visible."""
    buffers = [
        torch.full(
            (m + _SENTINEL_ROWS, width), float("nan"), dtype=dtype, device="cuda"
        )
        for width, dtype in zip(n_widths, out_dtypes)
    ]
    return buffers, [buffer[:m] for buffer in buffers]


def _single_launch_fp32(activations, weight, tactic, bias=None):
    x = torch.cat(activations, dim=1)
    out = torch.empty(x.shape[0], weight.shape[0], dtype=torch.float32, device="cuda")
    run_warp_splitk_dense(x, weight.t(), out, pdl=False, tactic=tactic, bias=bias)
    return out


def _run_and_check_bit_identity(
    *, m, k_widths, n_widths, out_dtypes, tactic, bias=None, seed=0
):
    """The multi-IO launch must reproduce the single launch on the concatenated
    activations bit for bit, region by region in each region's dtype."""
    activations, weight = _inputs(m, k_widths, sum(n_widths), seed)
    buffers, outputs = _guarded_outputs(m, n_widths, out_dtypes)
    run_warp_splitk_dense_multi_io(
        activations, weight, outputs, pdl=False, tactic=tactic, bias=bias
    )
    expected = _single_launch_fp32(activations, weight, tactic, bias=bias)
    torch.cuda.synchronize()
    start = 0
    for buffer, output, width in zip(buffers, outputs, n_widths):
        torch.testing.assert_close(
            output, expected[:, start : start + width].to(output.dtype), rtol=0, atol=0
        )
        assert buffer[m:].isnan().all(), "store past row M - 1"
        start += width
    return activations, weight, expected


@pytest.mark.parametrize(
    "k_widths,k_tile",
    [
        ((384, 128), 128),
        ((1536, 512), 256),
        ((256, 640, 128), 128),
        ((512, 1024, 256), 256),
    ],
)
def test_k_segments_match_single_launch(k_widths, k_tile):
    tactic = WarpSplitKTactic(
        output_tile=16, token_tile=8, k_tile=k_tile, stages=2, b_loader_warps=1
    )
    _run_and_check_bit_identity(
        m=5,
        k_widths=k_widths,
        n_widths=(256,),
        out_dtypes=(torch.bfloat16,),
        tactic=tactic,
    )


@pytest.mark.parametrize(
    "n_widths,out_dtypes,output_tile,has_bias",
    [
        ((96, 32), (torch.bfloat16, torch.float32), 32, False),
        ((48, 160, 16), (torch.float32, torch.bfloat16, torch.float32), 16, False),
        ((64, 32, 128), (torch.bfloat16, torch.bfloat16, torch.float32), 32, True),
    ],
)
def test_n_regions_match_single_launch(n_widths, out_dtypes, output_tile, has_bias):
    tactic = WarpSplitKTactic(
        output_tile=output_tile, token_tile=8, k_tile=128, stages=3, b_loader_warps=1
    )
    bias = (
        torch.randn(sum(n_widths), dtype=torch.bfloat16, device="cuda")
        if has_bias
        else None
    )
    _run_and_check_bit_identity(
        m=7,
        k_widths=(512,),
        n_widths=n_widths,
        out_dtypes=out_dtypes,
        tactic=tactic,
        bias=bias,
    )


@pytest.mark.parametrize("token_tile", [8, 16, 32])
@pytest.mark.parametrize("m", [1, 3, 9, 12, 15, 17, 24, 31, 32])
def test_m_tails_with_segments_and_regions(m, token_tile):
    """Two K segments and three mixed-dtype regions at every M tail: bit identity
    with the single launch, no store past row M - 1, and FP32 accuracy."""
    tactic = WarpSplitKTactic(
        output_tile=32,
        token_tile=token_tile,
        k_tile=128,
        stages=4,
        b_loader_warps=2 if token_tile >= 16 else 1,
    )
    activations, weight, expected = _run_and_check_bit_identity(
        m=m,
        k_widths=(768, 256),
        n_widths=(128, 64, 96),
        out_dtypes=(torch.bfloat16, torch.bfloat16, torch.float32),
        tactic=tactic,
        seed=m,
    )
    reference = torch.cat(activations, dim=1).double() @ weight.double().t()
    # Outputs are ~N(0, 1); a bf16 round trip would err by up to ~4e-3.
    torch.testing.assert_close(expected.double(), reference, rtol=0, atol=1e-4)


def test_fp32_region_keeps_fp32_precision():
    """An FP32 region must hold the FP32 accumulator, not a BF16-rounded value."""
    m, k_widths, n_widths = 6, (1536, 384), (512, 256)
    activations, weight = _inputs(m, k_widths, sum(n_widths), seed=11)
    outputs = [
        torch.empty(m, n_widths[0], dtype=torch.bfloat16, device="cuda"),
        torch.empty(m, n_widths[1], dtype=torch.float32, device="cuda"),
    ]
    run_warp_splitk_dense_multi_io(activations, weight, outputs, pdl=True)
    reference = torch.cat(activations, dim=1).double() @ weight.double().t()
    torch.testing.assert_close(
        outputs[1].double(), reference[:, n_widths[0] :], rtol=0, atol=1e-4
    )
    torch.testing.assert_close(
        outputs[0].float(), reference[:, : n_widths[0]].float(), rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_single_segment_single_output_matches_existing_api(out_dtype, has_bias):
    m, n, k = 13, 512, 2048
    activations, weight = _inputs(m, (k,), n, seed=3)
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda") if has_bias else None
    tactic = WarpSplitKTactic(
        output_tile=32, token_tile=16, k_tile=256, stages=6, b_loader_warps=2
    )
    legacy = torch.empty(m, n, dtype=out_dtype, device="cuda")
    run_warp_splitk_dense(
        activations[0], weight.t(), legacy, pdl=True, tactic=tactic, bias=bias
    )
    multi = torch.empty(m, n, dtype=out_dtype, device="cuda")
    run_warp_splitk_dense_multi_io(
        activations, weight, [multi], pdl=True, tactic=tactic, bias=bias
    )
    torch.testing.assert_close(multi, legacy, rtol=0, atol=0)


def test_rejects_splits_that_are_not_whole_tiles():
    """A K tile may come from only one activation and a CTA may store to only one
    output; splits that break that must be rejected, not computed wrongly."""
    tactic = WarpSplitKTactic(
        output_tile=32, token_tile=8, k_tile=256, stages=2, b_loader_warps=1
    )
    activations, weight = _inputs(4, (384, 384), 64, seed=0)
    outputs = [torch.empty(4, 64, dtype=torch.bfloat16, device="cuda")]
    with pytest.raises(ValueError, match="K segments"):
        run_warp_splitk_dense_multi_io(
            activations, weight, outputs, pdl=False, tactic=tactic
        )
    activations, weight = _inputs(4, (1024,), 64, seed=0)
    outputs = [
        torch.empty(4, 48, dtype=torch.bfloat16, device="cuda"),
        torch.empty(4, 16, dtype=torch.float32, device="cuda"),
    ]
    with pytest.raises(ValueError, match="N regions"):
        run_warp_splitk_dense_multi_io(
            activations, weight, outputs, pdl=False, tactic=tactic
        )


_AOT_CHILD = textwrap.dedent(
    """
    import sys

    import torch

    from sglang.kernels.ops.gemm import dense_bf16_gemm_warp_splitk as ws

    phase, result_path = sys.argv[1], sys.argv[2]
    if phase == "reload":
        def _no_compile(*args, **kwargs):
            raise AssertionError("compiled a variant that is in the AOT cache")
        ws.cute_ext.compile = _no_compile

    k_widths, n_widths = (768, 256), (128, 64, 96)
    out_dtypes = (torch.bfloat16, torch.bfloat16, torch.float32)
    compiled = ws.prebuild_warp_splitk_dense_multi_io(
        m_values=[3, 16, 64],
        k_widths=k_widths,
        n_widths=n_widths,
        out_dtypes=out_dtypes,
        pdl=True,
    )
    torch.manual_seed(0)
    activations = [
        torch.randn(3, width, dtype=torch.bfloat16, device="cuda") for width in k_widths
    ]
    weight = torch.randn(sum(n_widths), sum(k_widths), device="cuda").bfloat16()
    outputs = [
        torch.empty(3, width, dtype=dtype, device="cuda")
        for width, dtype in zip(n_widths, out_dtypes)
    ]
    ws.run_warp_splitk_dense_multi_io(activations, weight, outputs, pdl=True)
    torch.save({"compiled": compiled, "outputs": [o.cpu() for o in outputs]}, result_path)
    """
)


def test_aot_cache_round_trip(tmp_path):
    """Kernels built in one process are exported and reloaded in the next, which
    must then neither compile nor change the result."""
    env = {**os.environ, "SGLANG_CUTE_AOT_CACHE_DIR": str(tmp_path / "cache")}
    script = tmp_path / "child.py"
    script.write_text(_AOT_CHILD)
    results = {}
    for phase in ("build", "reload"):
        result_path = tmp_path / f"{phase}.pt"
        subprocess.run(
            [sys.executable, str(script), phase, str(result_path)],
            env=env,
            check=True,
            timeout=600,
        )
        results[phase] = torch.load(result_path)

    # M = 64 is outside the kernel's range and is skipped, not compiled.
    assert results["build"]["compiled"] == 2
    assert results["reload"]["compiled"] == 0
    for built, reloaded in zip(
        results["build"]["outputs"], results["reload"]["outputs"]
    ):
        torch.testing.assert_close(reloaded, built, rtol=0, atol=0)
    assert len(list((tmp_path / "cache").rglob("*.o"))) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
