# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensor contracts for pipeline boundaries carrying deferred mHC state."""

import torch

from vllm.sequence import IntermediateTensors


def make_empty_mhc_intermediate_tensors(
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    hidden_size: int,
    hc_mult: int,
) -> IntermediateTensors:
    if dtype != torch.bfloat16:
        raise ValueError("mHC pipeline activations require torch.bfloat16")
    return IntermediateTensors(
        {
            "hidden_states": torch.zeros(
                (batch_size, hidden_size), dtype=dtype, device=device
            ),
            "residual": torch.zeros(
                (batch_size, hc_mult, hidden_size), dtype=dtype, device=device
            ),
            "post": torch.zeros(
                (batch_size, hc_mult, 1), dtype=torch.float32, device=device
            ),
            "comb": torch.zeros(
                (batch_size, hc_mult, hc_mult), dtype=torch.float32, device=device
            ),
        }
    )


def unpack_mhc_intermediate_tensors(
    tensors: IntermediateTensors,
    *,
    hidden_size: int,
    hc_mult: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = ("hidden_states", "residual", "post", "comb")
    for key in keys:
        if key not in tensors.tensors or not isinstance(tensors[key], torch.Tensor):
            raise ValueError(f"Missing mHC pipeline tensor: {key}")
    hidden_states = tensors["hidden_states"]
    num_tokens = hidden_states.shape[0]
    expected = (
        ((num_tokens, hidden_size), torch.bfloat16),
        ((num_tokens, hc_mult, hidden_size), torch.bfloat16),
        ((num_tokens, hc_mult, 1), torch.float32),
        ((num_tokens, hc_mult, hc_mult), torch.float32),
    )
    for key, (shape, dtype) in zip(keys, expected):
        tensor = tensors[key]
        if tensor.shape != shape or tensor.dtype != dtype:
            raise ValueError(
                f"Invalid mHC pipeline {key}: expected {shape} {dtype}, "
                f"got {tensor.shape} {tensor.dtype}"
            )
        if tensor.device != hidden_states.device:
            raise ValueError(f"mHC pipeline {key} is on a different device")
    return hidden_states, tensors["residual"], tensors["post"], tensors["comb"]


def pack_mhc_intermediate_tensors(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> IntermediateTensors:
    tensors = IntermediateTensors(
        {
            "hidden_states": hidden_states,
            "residual": residual,
            "post": post,
            "comb": comb,
        }
    )
    if not isinstance(residual, torch.Tensor) or residual.ndim != 3:
        raise ValueError("Invalid mHC pipeline residual: expected [tokens, streams, H]")
    unpack_mhc_intermediate_tensors(
        tensors, hidden_size=residual.shape[-1], hc_mult=residual.shape[-2]
    )
    return tensors
