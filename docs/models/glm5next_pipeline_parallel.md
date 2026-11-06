# GLM5Next pipeline parallelism

This branch adds experimental pipeline parallelism (PP) for GLM-5.3-Flash
on vLLM v0.33.0. CPU tensor-contract and mathematical continuity tests are
available. Real multi-GPU model parity, image/video inference, and ROCm 10.0
compatibility have **not** been validated in the development environment.

## Model boundary state

For mHC, stages pass all four deferred states through the existing PP
tensor-dictionary transport:

| Key | Shape | Dtype |
| --- | --- | --- |
| `hidden_states` | `(tokens, hidden_size)` | BF16 |
| `residual` | `(tokens, hc_mult, hidden_size)` | BF16 |
| `post` | `(tokens, hc_mult, 1)` | FP32 |
| `comb` | `(tokens, hc_mult, hc_mult)` | FP32 |

The receiving stage continues the existing fused post/pre operation.
Only global layer 0 expands the streams; only the final global decoder layer
finishes and contracts them. With sequence-parallel MoE, the model gathers
all four states before PP transport and shards all four at stage entry.
TP, DP, EP grouping and generic PP communication remain unchanged.

The multimodal wrapper delegates its intermediate tensor factory to the text
model. Its existing forward passes stage activations to that model. This is
not a claim of image/video numerical parity; the supplied distributed tests
exercise text inputs through both text and multimodal wrappers.

## Constraints

- mHC PP requires BF16 activations and nonempty layer partitions.
- PP with auxiliary hidden-state extraction (including EAGLE3/DFlash paths
  needing cross-stage auxiliary states) is rejected. PP1 behavior is retained.
- MTP under PP loads an independent input embedding from the draft checkpoint.
  Either `model.embed_tokens.weight` or the first MTP layer's
  `model.layers.N.embed_tokens.weight` must be present. Multimodal prefixes
  are normalized. An explicit MTP embedding takes precedence regardless of
  iteration order; missing embeddings fail during loading. PP1 still shares
  the target embedding.
- KDA recurrent/conv caches and sparse MLA KV/indexer caches remain local to
  their owning layers. They are not PP activation tensors.
- The BF16 state format does not imply that FP8/NVFP4 weights or all attention
  backends have been validated with PP. Hardware/kernel support is still
  required.

## CPU checks

From the repository root, create a uv environment using Python 3.12. Install
the CPU PyTorch build appropriate to the host; this branch was checked with
PyTorch 2.14.0. The checkpoint tests also use Transformers in the repository's
required range and safetensors.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python torch==2.14.0 pytest \
  'transformers>=5.16.1,<5.21.0' safetensors
.venv/bin/python -m pytest --noconftest \
  tests/models/glm5next/test_pp_transport_cpu.py \
  tests/models/glm5next/test_pp_checkpoint_cpu.py \
  tests/models/glm5next/test_pp_logprobs_cpu.py -q
```

On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.
`--noconftest` isolates these self-contained tests from the GPU-oriented root
fixtures. It is not used for the real model/distributed tests.

The math test compares an eight-layer mHC reference chain across every one
of its 35 four-stage partitions. It includes explicit RMSNorm and production
tensor packing/copying/unpacking. It does **not** execute real KDA/MLA kernels
or validate multi-GPU PP.

## Real model checks

The distributed suite builds deterministic, small real GLM5Next checkpoints
using Transformers initialization and original checkpoint serialization.
It uses actual `LLM` execution with KDA, sparse MLA, mHC and MoE; the dummy
weight loader and replacement decoder layers are not used.

```bash
# In a compatible vLLM GPU build, with four GPUs:
.venv/bin/python -m pytest tests/distributed/test_glm5next_pipeline.py \
  -k 'not tp4_pp4' -v

# With sixteen GPUs, keep TP4 on both sides of the PP1/PP4 comparison:
.venv/bin/python -m pytest tests/distributed/test_glm5next_pipeline.py \
  -k tp4_pp4 -v
```

The four-GPU suite checks three partitions, full-vocabulary sample/prompt
logprobs, stage ownership, remote FP8 loading filters, and
MTP embedding loading/priority. Both text and multimodal wrappers are covered
with text inputs. The random drafter fixture does not measure trained MTP
acceptance or performance. The sixteen-GPU test uses the small checkpoint;
full-model acceptance is a separate step.

Prompt distributions are compared for every fixed input position. Sample
distributions are compared while the generated histories agree, including
the first near-tie divergence; distributions after different histories are
not compared. The fixed `atol=rtol=1e-3` numerical bound is not widened to
permit a test failure. Greedy token identity is not assumed without batch
invariance. Callable model inspection uses the repository's `enable_pickle`
fixture only in the test scope.

## AMD deployment validation

For a future TP4 + PP4 + DP1 deployment, first pin the complete software and
checkpoint revisions and verify ROCm 10.0's actual compatibility with this
vLLM/PyTorch build. Measure PCIe/P2P/RCCL connectivity and map each four-GPU TP
group to a PCIe switch before choosing rank order.

Start with eager execution, EP/speculation/prefix caching disabled, and
`VLLM_PP_LAYER_PARTITION=11,11,12,11` for the official 45-layer model. Compare
against TP4 + PP1 with identical tokens, backends and cache settings. If that
baseline cannot fit, report full-model parity as unverified and retain the
small-checkpoint comparison rather than substituting a TP1 baseline.

Then validate chunked prefill, decode, mixed batches, request slot reuse,
cancellation/preemption, long-context cache continuity, graphs/compile, MTP
verification/rollback, and image/video inputs separately. Inspect intermediate
states, final hidden states and logits, not just successful service startup.
Only after numerical checks pass should TTFT, TPOT, throughput, memory peaks,
pipeline bubbles and switch traffic be used to tune the layer partition.
