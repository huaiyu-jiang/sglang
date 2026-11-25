"""Tests for the stacked Hy4-Flash MoE dense projections (front and back skinny GEMMs)."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if (
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (9, 0)
):
    pytest.skip("SM90 required", allow_module_level=True)

from sglang.kernels.ops.gemm import hy4_flash_moe_dense as dense  # noqa: E402

_HIDDEN, _LATENT, _EXPERTS, _SHARED_INTERMEDIATE = 3072, 1536, 512, 1536
_SENTINEL_ROWS = 3


def _weights(
    tp_size: int, seed: int = 0, latent: int = _LATENT, shared: int | None = None
) -> dict[str, torch.Tensor]:
    """One Hy4-Flash MoE block's dense weights on one rank at ``tp_size``."""
    torch.manual_seed(seed)
    shared = shared or _SHARED_INTERMEDIATE // tp_size

    def weight(n, k):
        return (torch.randn(n, k, device="cuda") / k**0.5).to(torch.bfloat16)

    return dict(
        latent_down=weight(latent, _HIDDEN),
        shared_gate_up=weight(2 * shared, _HIDDEN),
        router=weight(_EXPERTS, _HIDDEN),
        latent_up=weight(_HIDDEN, latent),
        shared_down=weight(_HIDDEN, shared),
    )


@pytest.fixture(scope="module")
def tp4():
    weights = _weights(tp_size=4)
    return weights, dense.stack_hy4_moe_dense_weights(**weights)


def _activation(m: int, cols: int, *, misalign32: bool = False) -> torch.Tensor:
    """BF16 ``[m, cols]``; ``misalign32`` gives a 16- but not 32-byte aligned
    view, which the warp split-K kernel does not take."""
    offset = 8 if misalign32 else 0
    flat = torch.randn(m * cols + offset, device="cuda").to(torch.bfloat16)
    return flat[offset:].view(m, cols)


def _check_front(weights, stacked, x):
    out = dense.hy4_moe_dense_front(stacked, x)
    assert out is not None
    lat, gu, logits = out
    assert (lat.dtype, gu.dtype, logits.dtype) == (
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
    )
    xd = x.double()
    for got, name in ((lat, "latent_down"), (gu, "shared_gate_up")):
        ref = xd @ weights[name].double().t()
        torch.testing.assert_close(got.double(), ref, rtol=1e-2, atol=1e-2)
    # The router logits are exact BF16 products summed in FP32, never rounded
    # to BF16 (a BF16 round trip would be off by ~1e-2 here).
    ref = xd @ weights["router"].double().t()
    torch.testing.assert_close(logits.double(), ref, rtol=0, atol=1e-4)


def _check_back(weights, stacked, lat_sum, act):
    out = dense.hy4_moe_dense_back(stacked, lat_sum, act)
    assert out is not None and out.dtype == torch.bfloat16
    ref = lat_sum.double() @ weights["latent_up"].double().t()
    ref += act.double() @ weights["shared_down"].double().t()
    torch.testing.assert_close(out.double(), ref, rtol=1e-2, atol=1e-2)


def test_views_alias_the_stacked_weights(tp4):
    """The model points its parameters at these views; they must hold exactly
    the input weights and live inside the stacked tensors."""
    weights, stacked = tp4
    names = ("latent_down", "shared_gate_up", "router", "latent_up", "shared_down")
    views = stacked.front_views() + stacked.back_views()
    owners = (stacked.front,) * 3 + (stacked.back,) * 2
    for name, view, owner in zip(names, views, owners):
        torch.testing.assert_close(view, weights[name], rtol=0, atol=0)
        assert view.untyped_storage().data_ptr() == owner.untyped_storage().data_ptr()


@pytest.mark.parametrize("m", range(1, 129))
def test_every_captured_m_matches_reference(tp4, m):
    """Every token count up to 128 is served (warp split-K to 32, Triton above),
    including every token-tile tail."""
    weights, stacked = tp4
    torch.manual_seed(m)
    _check_front(weights, stacked, _activation(m, _HIDDEN))
    lat_sum = _activation(m, _LATENT)
    act = _activation(m, stacked.back.shape[1] - _LATENT)
    _check_back(weights, stacked, lat_sum, act)


@pytest.mark.parametrize("m", range(1, 33))
def test_triton_serves_m_up_to_32_when_warp_splitk_cannot(tp4, m):
    """Operands that are 16- but not 32-byte aligned skip warp split-K; the
    Triton FMA / small-M MMA configs must then give the same answers."""
    weights, stacked = tp4
    torch.manual_seed(m)
    _check_front(weights, stacked, _activation(m, _HIDDEN, misalign32=True))
    lat_sum = _activation(m, _LATENT, misalign32=True)
    act = _activation(m, stacked.back.shape[1] - _LATENT)
    _check_back(weights, stacked, lat_sum, act)


@pytest.mark.parametrize("m", [1, 5, 16, 17, 33, 47, 64, 65, 100, 127])
def test_triton_stores_no_row_past_m(tp4, m):
    _, stacked = tp4
    x = _activation(m, _HIDDEN)
    widths = (
        stacked.n_lat,
        stacked.n_gu,
        stacked.front.shape[0] - stacked.n_lat - stacked.n_gu,
    )
    dtypes = (torch.bfloat16, torch.bfloat16, torch.float32)
    buffers = [
        torch.full((m + _SENTINEL_ROWS, w), float("nan"), dtype=dt, device="cuda")
        for w, dt in zip(widths, dtypes)
    ]
    dense._launch_triton(
        stacked.front,
        x,
        x,
        [b[:m] for b in buffers],
        stacked.front_tile_counters,
        k0=_HIDDEN,
        n0=stacked.n_lat,
        n1=stacked.n_gu,
        cfg=dense._TRITON_FRONT[dense._bucket(m)],
    )
    for buffer in buffers:
        assert not buffer[:m].isnan().any()
        assert buffer[m:].isnan().all(), "store past row M - 1"


@pytest.mark.parametrize("m", [40, 96, 128])
def test_split_k_is_deterministic_and_resets_its_counters(tp4, m):
    """Split-K partials are summed in a fixed order by the last tile, which
    also re-zeroes the tile counters; repeated launches (with other launches
    and an L2 flush in between) must agree bit for bit."""
    _, stacked = tp4
    x = _activation(m, _HIDDEN)
    lat_sum = _activation(m, _LATENT)
    act = _activation(m, stacked.back.shape[1] - _LATENT)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    runs = []
    for _ in range(5):
        flush.zero_()
        dense.hy4_moe_dense_front(stacked, _activation(m + 1, _HIDDEN))
        runs.append(
            (
                *dense.hy4_moe_dense_front(stacked, x),
                dense.hy4_moe_dense_back(stacked, lat_sum, act),
            )
        )
    for run in runs[1:]:
        for got, first in zip(run, runs[0]):
            torch.testing.assert_close(got, first, rtol=0, atol=0)
    assert not stacked.front_tile_counters.any()
    assert not stacked.back_tile_counters.any()


@pytest.mark.parametrize("m", [8, 48])
def test_graph_replay_with_new_inputs(tp4, m):
    """Captured once, replayed on new inputs: same results as a direct launch
    (warp split-K at 8 tokens, Triton split-K at 48)."""
    _, stacked = tp4
    x = _activation(m, _HIDDEN)
    lat_sum = _activation(m, _LATENT)
    act = _activation(m, stacked.back.shape[1] - _LATENT)
    dense.hy4_moe_dense_front(stacked, x)
    dense.hy4_moe_dense_back(stacked, lat_sum, act)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        front = dense.hy4_moe_dense_front(stacked, x)
        back = dense.hy4_moe_dense_back(stacked, lat_sum, act)
    for seed in range(3):
        torch.manual_seed(100 + seed)
        for t in (x, lat_sum, act):
            t.copy_(torch.randn_like(t))
        graph.replay()
        expected = (
            *dense.hy4_moe_dense_front(stacked, x),
            dense.hy4_moe_dense_back(stacked, lat_sum, act),
        )
        for got, want in zip((*front, back), expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("tp_size", [2, 8])
@pytest.mark.parametrize("m", [1, 2, 7, 16, 32, 33, 64, 65, 128])
def test_other_tp_shapes_are_correct_or_declined(tp_size, m):
    """Configs were tuned at TP4. At other TP sizes each launch either fits the
    shapes and is correct, or returns None so the caller runs its own path."""
    weights = _weights(tp_size=tp_size, seed=tp_size)
    stacked = dense.stack_hy4_moe_dense_weights(**weights)
    x = _activation(m, _HIDDEN)
    if dense.hy4_moe_dense_front(stacked, x) is not None:
        _check_front(weights, stacked, x)
    lat_sum = _activation(m, _LATENT)
    act = _activation(m, stacked.back.shape[1] - _LATENT)
    if tp_size == 8:
        # K = 1536 + 192: the segment boundary is on no K tile of either kernel.
        assert dense.hy4_moe_dense_back(stacked, lat_sum, act) is None
    else:
        _check_back(weights, stacked, lat_sum, act)


@pytest.mark.parametrize("m", [1, 16, 48, 128])
def test_k_segment_boundary_off_the_tile_grid_is_declined(m):
    """K = 1568 + 480 = 2048 fills whole K tiles, but the boundary between the
    two activations falls on no K tile of any config; no kernel may take it."""
    weights = _weights(tp_size=4, latent=1568, shared=480)
    stacked = dense.stack_hy4_moe_dense_weights(**weights)
    lat_sum = _activation(m, 1568)
    act = _activation(m, 480)
    assert dense.hy4_moe_dense_back(stacked, lat_sum, act) is None


def test_uncovered_token_counts_return_none(tp4):
    _, stacked = tp4
    for m in (0, 129):
        assert dense.hy4_moe_dense_front(stacked, _activation(m, _HIDDEN)) is None
        lat_sum = _activation(m, _LATENT)
        act = _activation(m, stacked.back.shape[1] - _LATENT)
        assert dense.hy4_moe_dense_back(stacked, lat_sum, act) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
