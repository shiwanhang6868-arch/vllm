# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real GLM5Next fixtures and numerical checks for pipeline regressions."""

from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import Glm5NextConfig, Glm5NextTextConfig, Glm5NextVisionConfig


def tiny_glm5next_config() -> Glm5NextConfig:
    text = Glm5NextTextConfig(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_hidden_layers=17,
        num_attention_heads=4,
        num_key_value_heads=4,
        linear_num_heads=4,
        linear_head_dim=128,
        q_lora_rank=64,
        kv_lora_rank=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        index_topk=128,
        index_kpool=4,
        index_n_heads=4,
        index_head_dim=128,
        indexer_rope_interleave=True,
        index_kpool_compress=True,
        index_share_for_mtp_iteration=True,
        scoring_func="sigmoid",
        topk_method="noaux_tc",
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=3,
        mhc=True,
        mla_use_nope=True,
        num_nextn_predict_layers=0,
        max_position_embeddings=512,
        pad_token_id=0,
        eos_token_id=0,
    )
    return Glm5NextConfig(
        text_config=text,
        vision_config=Glm5NextVisionConfig(
            depth=1,
            hidden_size=128,
            intermediate_size=256,
            num_heads=2,
            out_hidden_size=text.hidden_size,
            projection_intermediate_size=256,
        ),
        image_token_id=120,
        video_token_id=121,
        image_start_token_id=122,
        image_end_token_id=123,
        video_start_token_id=124,
        video_end_token_id=125,
    )


def save_tiny_glm5next_checkpoint(
    path: Path, *, multimodal: bool = True, mtp: bool = False
) -> None:
    from safetensors.torch import load_file, save_file
    from transformers import Glm5NextForConditionalGeneration

    # Use Transformers' real initialization and original checkpoint naming,
    # not vLLM's dummy loader or hand-written replacement decoder layers.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        config = tiny_glm5next_config()
        config.text_config.num_nextn_predict_layers = int(mtp)
        model = Glm5NextForConditionalGeneration(config)
    model.save_pretrained(path, save_original_format=True)
    weights_path = path / "model.safetensors"
    weights = load_file(weights_path)
    prefix = "model.language_model."
    if mtp:
        # A real untrained drafter: reuse an MLA/MoE block and initialize its
        # additional projection. It tests loading/verification, not acceptance.
        mtp_prefix = f"{prefix}layers.17."
        for name, value in list(weights.items()):
            if name.startswith(f"{prefix}layers.15."):
                suffix = name.removeprefix(f"{prefix}layers.15.")
                if not suffix.startswith("hc_"):
                    weights[mtp_prefix + suffix] = value.clone()
        weights[mtp_prefix + "enorm.weight"] = torch.ones(256)
        weights[mtp_prefix + "hnorm.weight"] = torch.ones(256)
        weights[mtp_prefix + "shared_head.norm.weight"] = torch.ones(256)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            projection = torch.nn.Linear(512, 256, bias=False)
        weights[mtp_prefix + "eh_proj.weight"] = projection.weight.detach()
    if not multimodal:
        weights = {
            name.replace(prefix, "model.", 1): value
            for name, value in weights.items()
            if name.startswith(prefix) or name == "lm_head.weight"
        }
        config = config.text_config
        config.architectures = ["Glm5NextForCausalLM"]
        config.save_pretrained(path)
    save_file(weights, weights_path, metadata={"format": "pt"})


def assert_pp_logprobs_close(
    reference: Sequence[dict[int, float] | None],
    actual: Sequence[dict[int, float] | None],
    *,
    reference_tokens: Sequence[int] | None = None,
    actual_tokens: Sequence[int] | None = None,
) -> None:
    """Compare every fixed-history distribution, including a near-tie step.

    Prompt histories are identical by construction. Sample histories are
    identical only until their first differing generated token.
    """
    assert len(reference) == len(actual)
    if reference_tokens is not None:
        assert actual_tokens is not None
        assert len(reference_tokens) == len(actual_tokens) == len(reference)
    for index, (before, after) in enumerate(zip(reference, actual)):
        if before is None:
            assert after is None
            continue
        assert after is not None and before.keys() == after.keys()
        keys = sorted(before)
        torch.testing.assert_close(
            torch.tensor([before[k] for k in keys], dtype=torch.float32),
            torch.tensor([after[k] for k in keys], dtype=torch.float32),
            atol=1e-3,
            rtol=1e-3,
        )
        if reference_tokens is not None:
            assert actual_tokens is not None
            if reference_tokens[index] != actual_tokens[index]:
                break
