# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real LLM/checkpoint tests. Requires 4 GPUs, or 16 for the TP4 acceptance test.

CPU math continuity is covered separately; no test here replaces KDA/MLA.
"""

from pathlib import Path

import pytest
import torch

from tests.models.glm5next.pp_fixtures import save_tiny_glm5next_checkpoint
from vllm import SamplingParams

PARTITIONS = ["4,4,4,5", "5,4,4,4", "7,4,3,3"]
PROMPTS = [{"prompt_token_ids": list(range(1, n + 1))} for n in (3, 5, 33)]


@pytest.fixture(scope="module", params=[False, True], ids=["text", "multimodal"])
def checkpoint(tmp_path_factory, request):
    path = tmp_path_factory.mktemp("glm5next-pp")
    save_tiny_glm5next_checkpoint(path, multimodal=request.param)
    return str(path)


def _inspect_stage(model):
    from vllm.distributed import get_pp_group
    from vllm.model_executor.models.utils import PPMissingLayer

    text = model.language_model if hasattr(model, "language_model") else model
    inner = text.model
    pp = get_pp_group()
    buffers = model.make_empty_intermediate_tensors(
        5, torch.bfloat16, next(inner.parameters()).device
    )
    assert buffers["residual"].shape == (5, 4, 256)
    assert buffers["post"].dtype == buffers["comb"].dtype == torch.float32
    assert isinstance(inner.embed_tokens, PPMissingLayer) == (not pp.is_first_rank)
    assert isinstance(inner.norm, PPMissingLayer) == (not pp.is_last_rank)
    assert isinstance(text.lm_head, PPMissingLayer) == (not pp.is_last_rank)
    if pp.world_size > 1:
        remote = 7 if inner.start_layer <= 3 < inner.end_layer else 3
        prefix = f"layers.{remote}.self_attn."
        fp8 = torch.ones(128, 128).to(torch.float8_e4m3fn)
        # Both special FP8 helpers must skip a nonlocal layer before buffering
        # or indexing params_dict, independently of weight/scale ordering.
        for projection in ("indexer.wk", "q_a_proj"):
            weights = [
                (prefix + projection + ".weight", fp8),
                (prefix + projection + ".weight_scale_inv", torch.ones(1, 1)),
            ]
            assert inner.load_weights(iter(weights)) == set()
            assert inner.load_weights(iter(reversed(weights))) == set()
        with pytest.raises(KeyError):
            inner.load_weights(
                [(f"layers.{inner.start_layer}.invalid_local.weight", torch.ones(1))]
            )
        model.set_aux_hidden_state_layers((0,))
        try:
            with pytest.raises(ValueError, match="auxiliary hidden states"):
                inner(None, torch.zeros(1, dtype=torch.long), None)
        finally:
            model.set_aux_hidden_state_layers(())
    return inner.start_layer, inner.end_layer


def _run(vllm_runner, checkpoint, tp, pp, partition, monkeypatch):
    if partition is None:
        monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    else:
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", partition)
    with vllm_runner(
        checkpoint,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        data_parallel_size=1,
        distributed_executor_backend="mp",
        skip_tokenizer_init=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_logprobs=128,
        limit_mm_per_prompt={"image": 0, "video": 0},
    ) as runner:
        spans = runner.apply_model(_inspect_stage)
        outputs = runner.llm.generate(
            PROMPTS,
            SamplingParams(
                temperature=0,
                max_tokens=8,
                ignore_eos=True,
                detokenize=False,
                logprobs=128,
                prompt_logprobs=128,
            ),
        )
        return outputs, spans


def _compare_full_logprobs(reference, actual):
    assert len(reference) == len(actual)
    for expected, received in zip(reference, actual):
        assert expected.outputs[0].token_ids == received.outputs[0].token_ids
        groups = (
            (expected.prompt_logprobs, received.prompt_logprobs),
            (expected.outputs[0].logprobs, received.outputs[0].logprobs),
        )
        for expected_steps, received_steps in groups:
            assert len(expected_steps) == len(received_steps)
            for before, after in zip(expected_steps, received_steps):
                if before is None:
                    assert after is None
                    continue
                assert before.keys() == after.keys()
                keys = sorted(before)
                torch.testing.assert_close(
                    torch.tensor([before[k].logprob for k in keys]),
                    torch.tensor([after[k].logprob for k in keys]),
                    atol=1e-3,
                    rtol=1e-3,
                )


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.parametrize("partition", PARTITIONS)
def test_glm5next_pp4_real_logits(vllm_runner, checkpoint, partition, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    reference, _ = _run(vllm_runner, checkpoint, 1, 1, None, monkeypatch)
    actual, spans = _run(vllm_runner, checkpoint, 1, 4, partition, monkeypatch)
    offsets = [0]
    for count in map(int, partition.split(",")):
        offsets.append(offsets[-1] + count)
    assert sorted(spans) == list(zip(offsets[:-1], offsets[1:]))
    _compare_full_logprobs(reference, actual)


@pytest.mark.distributed(num_gpus=16)
def test_glm5next_tp4_pp4_real_logits(vllm_runner, checkpoint, monkeypatch):
    """Same TP on both sides: this must not be replaced by a TP1 baseline."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    reference, _ = _run(vllm_runner, checkpoint, 4, 1, None, monkeypatch)
    actual, _ = _run(vllm_runner, checkpoint, 4, 4, PARTITIONS[0], monkeypatch)
    _compare_full_logprobs(reference, actual)


def _check_mtp_embedding_loads(worker, model_path):
    from safetensors.torch import load_file

    from vllm.distributed import get_pp_group

    draft = worker.get_draft_model()
    if draft is None:
        return None
    weights = load_file(str(Path(model_path) / "model.safetensors"))
    embedding = draft.model.embed_tokens
    target_name = "model.embed_tokens.weight"
    expected = weights[target_name].bfloat16()
    assert torch.equal(embedding.weight.cpu(), expected)
    marker = ("model.layers.17.enorm.weight", torch.ones(256))
    if get_pp_group().world_size == 1:
        assert not draft.has_own_embed_tokens
        assert embedding is worker.get_model().model.embed_tokens
        draft.load_weights([marker])
        return True
    assert draft.has_own_embed_tokens
    explicit_name = "model.layers.17.embed_tokens.weight"
    explicit = expected + 0.5
    for prefix in ("model.", "model.language_model."):
        target = (target_name.replace("model.", prefix, 1), expected)
        own = (explicit_name.replace("model.", prefix, 1), explicit)
        for ordered in ((target, own), (own, target)):
            draft.load_weights([*ordered, marker])
            assert torch.equal(embedding.weight.cpu(), explicit)
    with pytest.raises(ValueError, match="MTP input embedding"):
        draft.load_weights([marker])
    return True


@pytest.mark.distributed(num_gpus=4)
@pytest.mark.parametrize("pp_size", [1, 4])
def test_glm5next_mtp_pp_embedding(vllm_runner, tmp_path, monkeypatch, pp_size):
    path = tmp_path / "mtp"
    save_tiny_glm5next_checkpoint(path, multimodal=False, mtp=True)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    if pp_size > 1:
        monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", PARTITIONS[0])
    else:
        monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    with vllm_runner(
        str(path),
        dtype="bfloat16",
        pipeline_parallel_size=pp_size,
        distributed_executor_backend="mp",
        skip_tokenizer_init=True,
        enforce_eager=True,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=4,
        enable_prefix_caching=False,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    ) as runner:
        results = runner.collective_rpc(_check_mtp_embedding_loads, args=(str(path),))
        assert results.count(True) == 1
