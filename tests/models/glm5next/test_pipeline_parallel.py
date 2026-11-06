# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real constructors on the configured vLLM device; no replacement layers."""

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.models.interfaces import supports_pp
from vllm.models.glm5next.common.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

from .pp_fixtures import tiny_glm5next_config


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("mhc", [False, True])
def test_real_model_factory_matches_text_boundary(dist_init, tmp_path, multimodal, mhc):
    config = tiny_glm5next_config()
    config.text_config.mhc = mhc
    if multimodal:
        cls = Glm5NextForConditionalGeneration
    else:
        config = config.text_config
        cls = Glm5NextForCausalLM
    config.architectures = [cls.__name__]
    config.save_pretrained(tmp_path)
    vllm_config = EngineArgs(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        limit_mm_per_prompt={"image": 0, "video": 0} if multimodal else {},
    ).create_engine_config()
    vllm_config.kernel_config.enable_jit_warmup = False
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device(current_platform.device_type),
    ):
        model = cls(vllm_config=vllm_config)
    assert supports_pp(cls) and supports_pp(model)
    state = model.make_empty_intermediate_tensors(
        5, torch.bfloat16, torch.device(current_platform.device_type)
    )
    assert state["hidden_states"].shape == (5, 256)
    if mhc:
        assert state["residual"].shape == (5, 4, 256)
        assert state["post"].shape == (5, 4, 1)
        assert state["comb"].shape == (5, 4, 4)
        assert state["post"].dtype == state["comb"].dtype == torch.float32
    else:
        assert set(state.tensors) == {"hidden_states", "residual"}
        assert state["residual"].shape == (5, 256)
