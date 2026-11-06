# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verify the actual checkpoint fixtures without importing vLLM GPU modules."""

import json

import pytest
import torch
from safetensors.torch import load_file

from .pp_fixtures import save_tiny_glm5next_checkpoint


@pytest.mark.parametrize("multimodal,mtp", [(True, False), (False, True)])
def test_real_hybrid_checkpoint_has_original_names_and_mtp_embedding_fallback(
    tmp_path, multimodal, mtp
):
    save_tiny_glm5next_checkpoint(tmp_path, multimodal=multimodal, mtp=mtp)
    config = json.loads((tmp_path / "config.json").read_text())
    text_config = config["text_config"] if multimodal else config
    assert text_config["num_hidden_layers"] == 17
    assert text_config["mhc"] is True
    assert text_config["num_nextn_predict_layers"] == int(mtp)
    assert text_config["layer_types"][16] == "linear_attention"
    prefix = "model.language_model." if multimodal else "model."
    weights = load_file(tmp_path / "model.safetensors")
    assert weights[prefix + "embed_tokens.weight"].shape == (128, 256)
    assert weights[prefix + "layers.0.hc_attn_fn"].dtype == torch.float32
    assert weights[prefix + "layers.0.self_attn.q_conv1d.weight"].shape == (512, 1, 4)
    assert weights[prefix + "layers.3.self_attn.indexer.wk.weight"].shape == (128, 256)
    assert weights[prefix + "layers.3.mlp.experts.0.gate_proj.weight"].shape == (
        256,
        256,
    )
    assert any(name.startswith("model.visual.") for name in weights) == multimodal
    if mtp:
        assert prefix + "layers.17.embed_tokens.weight" not in weights
        assert weights[prefix + "layers.17.eh_proj.weight"].shape == (256, 512)
        assert prefix + "layers.17.hc_attn_fn" not in weights
