# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts and mHC math continuity, not distributed model validation.

Run without the GPU-oriented root conftest:
    python -m pytest --noconftest tests/models/glm5next/test_pp_transport_cpu.py
"""

import importlib.util
import itertools
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from vllm.models.common.mhc_pp import (
    make_empty_mhc_intermediate_tensors,
    pack_mhc_intermediate_tensors,
    unpack_mhc_intermediate_tensors,
)
from vllm.sequence import IntermediateTensors

HIDDEN, STREAMS = 16, 4


@pytest.fixture(scope="module", autouse=True)
def _single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _allocate(num_tokens: int) -> IntermediateTensors:
    return make_empty_mhc_intermediate_tensors(
        num_tokens,
        torch.bfloat16,
        torch.device("cpu"),
        hidden_size=HIDDEN,
        hc_mult=STREAMS,
    )


def _state(num_tokens: int) -> IntermediateTensors:
    tensors = _allocate(num_tokens)
    generator = torch.Generator().manual_seed(71)
    for tensor in tensors.tensors.values():
        tensor.normal_(generator=generator)
    return tensors


@pytest.mark.parametrize("num_tokens", [0, 1, 3, 17])
@pytest.mark.parametrize("hidden_size,hc_mult", [(16, 4), (7, 2)])
def test_factory_matches_mixed_dtype_wire_contract(num_tokens, hidden_size, hc_mult):
    result = make_empty_mhc_intermediate_tensors(
        num_tokens,
        torch.bfloat16,
        torch.device("cpu"),
        hidden_size=hidden_size,
        hc_mult=hc_mult,
    )
    assert isinstance(result, IntermediateTensors)
    assert list(result.tensors) == ["hidden_states", "residual", "post", "comb"]
    expected = [
        ((num_tokens, hidden_size), torch.bfloat16),
        ((num_tokens, hc_mult, hidden_size), torch.bfloat16),
        ((num_tokens, hc_mult, 1), torch.float32),
        ((num_tokens, hc_mult, hc_mult), torch.float32),
    ]
    for tensor, (shape, dtype) in zip(result.tensors.values(), expected):
        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.device == torch.device("cpu")
        assert torch.count_nonzero(tensor) == 0


def test_factory_rejects_unsupported_activation_dtype():
    with pytest.raises(ValueError, match="bfloat16"):
        make_empty_mhc_intermediate_tensors(
            2, torch.float16, torch.device("cpu"), hidden_size=HIDDEN, hc_mult=STREAMS
        )


def _copy_through_runner_buffer(
    state: IntermediateTensors, buffer: IntermediateTensors
) -> IntermediateTensors:
    num_tokens = state["hidden_states"].shape[0]
    for key, value in state.items():
        buffer[key][:num_tokens].copy_(value)
    return buffer[:num_tokens]


def test_persistent_receive_buffer_preserves_every_state_and_fp32_gate_bit():
    buffer = _allocate(17)
    for num_tokens in (17, 1, 5, 3, 0, 17):
        state = _state(num_tokens)
        received = _copy_through_runner_buffer(state, buffer)
        assert received == state
        x, residual, post, comb = unpack_mhc_intermediate_tensors(
            received, hidden_size=HIDDEN, hc_mult=STREAMS
        )
        assert pack_mhc_intermediate_tensors(x, residual, post, comb) == state
        if num_tokens:
            assert not torch.equal(post, post.bfloat16().float())


@pytest.mark.parametrize("key", ["hidden_states", "residual", "post", "comb"])
def test_missing_boundary_state_is_an_error(key):
    state = _state(3)
    del state.tensors[key]
    with pytest.raises(ValueError, match=key):
        unpack_mhc_intermediate_tensors(state, hidden_size=HIDDEN, hc_mult=STREAMS)


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("post", None),
        ("post", torch.zeros(3, STREAMS, 1, dtype=torch.bfloat16)),
        ("comb", torch.zeros(3, STREAMS, STREAMS, dtype=torch.bfloat16)),
        ("post", torch.zeros(3, STREAMS)),
        ("comb", torch.zeros(2, STREAMS, STREAMS)),
        ("residual", torch.zeros(3, HIDDEN, STREAMS, dtype=torch.bfloat16)),
        ("residual", torch.zeros(3, STREAMS, HIDDEN)),
        ("hidden_states", torch.zeros(3, HIDDEN)),
        ("post", torch.empty(3, STREAMS, 1, device="meta")),
    ],
)
def test_invalid_state_is_rejected_without_fallback(key, bad_value):
    state = _state(3)
    state.tensors[key] = bad_value
    with pytest.raises(ValueError, match=key):
        unpack_mhc_intermediate_tensors(state, hidden_size=HIDDEN, hc_mult=STREAMS)


def test_declared_model_dimensions_must_match_payload():
    with pytest.raises(ValueError, match="hidden_states"):
        unpack_mhc_intermediate_tensors(_state(3), hidden_size=HIDDEN + 1, hc_mult=4)
    with pytest.raises(ValueError, match="residual"):
        unpack_mhc_intermediate_tensors(_state(3), hidden_size=HIDDEN, hc_mult=2)


def test_transport_is_fullgraph_compilable_with_dynamic_tokens():
    def round_trip(x, residual, post, comb):
        packed = pack_mhc_intermediate_tensors(x, residual, post, comb)
        return unpack_mhc_intermediate_tensors(
            packed, hidden_size=HIDDEN, hc_mult=STREAMS
        )

    compiled = torch.compile(round_trip, backend="eager", fullgraph=True, dynamic=True)
    for num_tokens in (3, 7):
        state = _state(num_tokens)
        expected = tuple(state.tensors.values())
        actual = compiled(*expected)
        for before, after in zip(expected, actual):
            assert torch.equal(before, after)


def _reference_ops():
    # Import the production pure-torch reference without registering GPU ops.
    path = (
        Path(__file__).resolve().parents[3] / "vllm/model_executor/kernels/mhc/torch.py"
    )
    spec = importlib.util.spec_from_file_location("mhc_torch_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("num_tokens", [1, 3, 5])
def test_eight_layer_math_chain_is_exact_across_all_four_stage_partitions(num_tokens):
    """All 35 partitions preserve deferred state; this is not a KDA/MLA test."""
    ops = _reference_ops()
    generator = torch.Generator().manual_seed(19)
    x0 = torch.randn(num_tokens, HIDDEN, generator=generator).bfloat16()
    parameters = [
        (
            torch.randn((2 + STREAMS) * STREAMS, STREAMS * HIDDEN, generator=generator)
            * 0.03,
            torch.randn(3, generator=generator) * 0.1,
            torch.randn((2 + STREAMS) * STREAMS, generator=generator) * 0.1,
            torch.randn(HIDDEN, generator=generator).bfloat16(),
        )
        for _ in range(16)
    ]

    def run(boundaries):
        x, residual, post, comb = x0.clone(), None, None, None
        checkpoints = []
        for layer in range(8):
            for sublayer in range(2):
                if post is None:
                    residual = x.unsqueeze(1).expand(-1, STREAMS, -1).contiguous()
                else:
                    residual = ops.mhc_post_torch(x, residual, post, comb)
                fn, scale, base, norm = parameters[layer * 2 + sublayer]
                post, comb, x = ops.mhc_pre_torch(
                    residual, fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20
                )
                x = F.rms_norm(x, (HIDDEN,), norm, 1e-5)
                x = (torch.tanh(x.float()) * 0.7).bfloat16()
            if layer < 7:
                state = pack_mhc_intermediate_tensors(x, residual, post, comb)
                if layer + 1 in boundaries:
                    state = _copy_through_runner_buffer(state, _allocate(num_tokens))
                    x, residual, post, comb = unpack_mhc_intermediate_tensors(
                        state, hidden_size=HIDDEN, hc_mult=STREAMS
                    )
                checkpoints.append(state)
        output = ops.mhc_post_torch(x, residual, post, comb).mean(dim=1)
        return checkpoints, F.rms_norm(output, (HIDDEN,), eps=1e-5)

    reference_states, reference_output = run(())
    for boundaries in itertools.combinations(range(1, 8), 3):
        states, output = run(boundaries)
        assert states == reference_states, boundaries
        assert torch.equal(output, reference_output), boundaries


def test_post_and_comb_are_observable_inputs_not_discardable_metadata():
    ops = _reference_ops()
    x, residual, post, comb = _state(5).tensors.values()
    expected = ops.mhc_post_torch(x, residual, post, comb)
    assert not torch.equal(expected, ops.mhc_post_torch(x, residual, post * 0, comb))
    assert not torch.equal(
        expected, ops.mhc_post_torch(x, residual, post, comb.transpose(-1, -2))
    )
