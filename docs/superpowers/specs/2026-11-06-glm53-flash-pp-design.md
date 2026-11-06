# GLM-5.3-Flash / GLM5Next Pipeline Parallelism：源码研究与待审批设计

日期：2026-11-06（Asia/Shanghai）。状态：**第一、二阶段交付；未批准实施；没有修改产品代码或测试代码。**

## 1. 基线、范围与交付状态

- 本机目录：`D:\work\vllm-glm53-pp`，不是 Linux 的 `~/work`。
- 上游：`https://github.com/vllm-project/vllm.git`。
- 版本：annotated tag `v0.33.0`，tag 对象 `3a4d82f569a9f57054ba81633e9b6c73b34b51fe`，对应 commit **`4ba787ec9959e566d3b587a796dff07673324da5`**。
- 克隆方式：`git clone --branch v0.33.0 --depth 1 ... vllm-glm53-pp`。这是完整源码工作树、浅历史，不含模型权重。
- 独立分支：`research/glm53-flash-pp-v033`。实施继续使用独立分支，并把代码、测试、文档分为可审查的提交。
- 目标：未来 AMD 16 GPU，TP4 + PP4 + DP1，按四个 PCIe Switch 的实测拓扑安排 rank；本轮不声称部署或数值正确性已经验证。
- 第一阶段只读分析；第二阶段提出设计；第三阶段必须等待用户确认。因此本次没有创建实现补丁、添加单元测试或安装 vLLM/PyTorch 依赖。本文列出批准后要实现并运行的 CPU 测试。
- 不改通用 PP 通信协议，不调整原有 TP/DP/EP 分组和 MoE 算法。CPU 合同/数学测试、GPU 模型一致性和真实部署验收分别报告。

## 2. 源码定位及已有实现

用户指定的 `git grep` 已执行，v0.33.0 的关键位置：

| 文件与行号 | 对象/作用 |
|---|---|
| `vllm/models/glm5next/common/model.py:356` | `Glm5NextDecoderLayer` |
| 同文件 `529` | decoder `forward`，四元状态及 mHC 融合 |
| 同文件 `701` | `Glm5NextModel` |
| 同文件 `793` | model `forward`，现有不完整的 PP 收发 |
| 同文件 `836` | `PP is gated off` 注释，非末 rank 未发送 post/comb |
| 同文件 `859` | 主模型 `load_weights` |
| 同文件 `1047` | `Glm5NextForCausalLM`，已经继承 `SupportsPP` |
| 同文件 `1145` | `Glm5NextForConditionalGeneration` |
| 同文件 `1235` | 多模态包装没有 alias 空中间张量工厂的说明 |
| 同文件 `1356`、`1423` | `_try_load_fp8_indexer_wk` / `_try_load_fp8_attn_proj` |
| `vllm/models/glm5next/common/mtp.py:290` 起 | MTP 权重加载 |
| `vllm/model_executor/layers/mhc.py` | mHC CustomOp，含 AITER/TileLang/native/CPU 分发 |
| `vllm/model_executor/kernels/mhc/torch.py:48`、`135` | mHC pre/post 数学参考 |

行号指基线，实施后会变化。不能仅搜索 `vllm/model_executor/models`，此版本 GLM5Next 实现已在 `vllm/models/glm5next`。

联网检查发现已有相关工作，并非“从未有人实现”：

1. [vLLM PR #57171](https://github.com/vllm-project/vllm/pull/57171)：已关闭、未合并；方案是在边界完成 hc_post，仅发送 residual streams，并处理 MTP embedding。参考 patch 已下载到 `D:\work\glm53-pp-research\references\upstream-pr-57171.patch`，**没有应用**。本机 `git apply --check` 对当前基线失败，model.py 与 mtp.py 均有上下文冲突。该 PR 不能直接作为 v0.33.0 的可用补丁。
2. [vLLM-Metal PR #1152](https://github.com/vllm-project/vllm-metal/pull/1152)：页面显示 open；属于 Metal/MLX 执行栈，其作者报告 PP2 验证。可参考设计，不能把验证结论移植为 ROCm 正确性证据。
3. [PixelML 的 PP 部署记录](https://github.com/PixelML/club-170hx/blob/main/docs/TOPOLOGY-AND-PARALLELISM.md)：存在 SM80 上 GLM-5.3-Flash PP4 的第三方记录；不同硬件、分支、量化方式，未在本机复现。
4. [vLLM issue #63128](https://github.com/vllm-project/vllm/issues/63128)：指出 PP 下 draft-only checkpoint 缺 embedding 的风险。GLM5Next 当前加载器也存在独立的 embedding 加载缺口，见下文。

这些外部资料是先例和风险线索；下文状态形状、调用关系和加载问题来自本地 tag 源码，不以 PR 文字代替源码审查。

## 3. 三个模型类的现状

### 3.1 Glm5NextModel

构造时首 Stage 建 `VocabParallelEmbedding`，其他 Stage 放 `PPMissingLayer`；`make_layers` 保持全局 layer index，只构造本 rank 的 `[start_layer,end_layer)`，其他位置放占位层；末 Stage 构造 RMSNorm。稀疏 top-k workspace 是每个 Stage 自己的 buffer。

首 Stage 从 input ids 或多模态 `inputs_embeds` 获得 `[N,H]` 输入。四元初态是 `(hidden_states,None,None,None)`。非首 Stage 目前只读 `hidden_states/residual`，并把 `post/comb` 清为 None。非末 Stage 只返回前两项。这条路径无法连续执行延迟的 mHC post。

辅助状态 `_aux_hidden_state` 在 post 存在时临时计算 hc_post，再做 hc_contract，得到完整的 `[N,H]` 特征；它不能替代向下一层传递的未完成四元状态。

### 3.2 Glm5NextForCausalLM

已经显式继承 `SupportsPP`，不是缺这个父类。首尾分工已有基本框架；LM Head 只在末 Stage 构造；logits 由末 Stage 输出。尚未绑定 `self.model.make_empty_intermediate_tensors`。

**接口检测细节：** `interfaces.py:902` 的 class 检测包含 `SupportsPP in model.__mro__` 的捷径。因此不能根据源码注释或旧 PR 声称此 tag 上所有 `supports_pp(model_class)` 调用必然返回 False；class 级检查可能通过，实例仍缺工厂，并会在 runner profiling/graph buffer 分配时失败。需要同时测试 class、真实实例和 runner 实际入口。

### 3.3 Glm5NextForConditionalGeneration

从 `Glm4vForConditionalGeneration` 间接继承 `SupportsPP`。它绕过父类构造，自建 visual 和 language_model，因此没有执行父类的工厂 alias。继承的 forward 已把 `intermediate_tensors` 直接交给 `language_model.model`；有中间状态时会清掉 `inputs_embeds`，避免后续 Stage 再注入视觉 embedding。

保留 `HasInnerState`、`IsHybrid` 及 mamba dtype/shape/copy classmethod 委托；否则会破坏 hybrid cache 配置。权重前缀 mapper 保持：`model.visual.` → `visual.`，`model.language_model.` → `language_model.model.`，`lm_head.` → `language_model.lm_head.`。

`_mark_tower_model` 在 V2 runner 的非首 Stage 会用 `StageMissingLayer` 跳过 visual；旧 runner 行为不同，不能笼统声称所有 runner 都只在首 Stage 分配 vision weights。首轮实机以 V2 eager 为主，再单独验证 graph 与旧 runner。

## 4. mHC 的状态、生命周期和融合

设 N 为该步完整 token 数；H=`hidden_size`；S=`hc_mult`；开启 SP 时本地 token 数 M=`ceil(N/TP)`，包含 padding。无 SP 时 M=N。

官方[配置](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/config.json)在查询时给出 H=4096、S=4、45 个主层，mhc=true；KDA 与稀疏 MLA 混合，MLA 全局层号为 3,7,...,43，最后一层 44 是 KDA。前三层 dense MLP，之后 MoE。实际部署需固定模型 revision，不能长期依赖浮动 main。

| 状态 | 普通 mHC 层返回时形状 | dtype | 含义与下游用途 |
|---|---|---|---|
| hidden_states / x | `[M,H]` | BF16 | FFN 的原始输出，尚未写回 residual streams |
| residual / R | `[M,S,H]` | BF16 | 本层 FFN 输入时已经完成 attention post 的 streams |
| post / p | `[M,S,1]` | FP32 | 本层 FFN 输出写入各 stream 的系数 |
| comb / C | `[M,S,S]` | FP32 | 本层 FFN residual streams 的组合矩阵 |

`mhc_pre_torch` 的 docstring 对 post 省略了尾部 1，实际 return 是 `[M,S,1]`；分配必须按实现。

hc_post 的数学定义：

```text
R_next[t,j,h] = cast_BF16(
    p[t,j,0] * float(x[t,h])
    + sum_i C[t,i,j] * float(R[t,i,h])
)
```

pre 从展平的 streams 计算 FP32 投影、pre/post gates、Sinkhorn comb；pre 混合结果回 BF16，然后按调用参数做 RMSNorm。hc 参数 fn/base/scale 显式 FP32，不应被工厂的 model dtype 统一转成 BF16。

生命周期：

1. 第 0 层唯一执行 `hc_expand([M,H]) -> [M,S,H]`；standalone attention hc_pre 生成 p/C 和 attention 输入。
2. attention（KDA 或 MLA）只接收 `[tokens,H]`。SP 时 pre 在 shard 上算，attention 前 gather 完整 tokens，attention 后 reduce_scatter。
3. 本层 attention post 与 FFN pre 融合；此时 R 更新，生成 FFN 的 p/C 和归一化输入。
4. 非全局最后一层执行 FFN 后直接返回 `(x,R,p,C)`；本层 FFN hc_post 延迟到下一层 attention hc_pre 的融合内核。
5. 全局最后一层在 decoder 内完成 hc_post，`hc_contract` 对 S 条 streams 求均值，返回 `([M,H],None,None,None)`；model 再 gather（如有 SP）和 final RMSNorm。

**绝不能**在普通非末层边界把 p/C 设 None 后继续消费二维 x。那既不是完整 streams，也遗漏 R_next 的计算。

**None 规则：** 首层进入时 R/p/C 可以都为 None；mHC 普通层结束后四项全部存在；全局最后层完成并收缩后 R/p/C 都为 None。非 mHC 普通层的 post/comb 为 None，x 已经加过 residual；MTP 非 mHC 路径返回未相加的 x/residual 给 shared_head fused norm。此差异必须保留。

**CPU 参考限制：** 本 tag 的 `MHCPreOp.forward_native` 返回的数学 pre 未应用 norm；HIP fallback 在外层 `_apply_mhc_norm` 补上。CPU pre wrapper 接收 norm 参数但不使用；fused CPU 路径也不完整复现传入 norm。不能把直接调用这些 native/CPU 方法所得的链作为 GPU fused oracle。数学测试必须显式加入与模型相同 epsilon/weight 的 RMSNorm。此处是既有风险，首轮 PP 补丁不顺手改通用 mHC 内核。

## 5. PP 接口约束与边界决策

### 5.1 接口

- `SupportsPP`：forward 接受 `intermediate_tensors`；实例必须有可调用的 `make_empty_intermediate_tensors(batch_size,dtype,device)`。协议 docstring 关于“only for last rank”与实际模型相反，实际 contract 是非末 Stage 返回 IntermediateTensors，末 Stage 返回 hidden / `(hidden,aux)`。
- `IntermediateTensors`：`dict[str,Tensor]`；切片对**所有值的第 0 维**进行 token slicing。不能把 residual 放成 `[S,N,H]`；不能把 None 放进 tensor dict。
- runner V1 和 V2 会按 key 把收到的值 `copy_` 到持久 buffer，再沿第 0 维切片供 graph replay；工厂与真实发送必须 key、shape、dtype 一致。
- `make_empty_intermediate_tensors_factory` 只分配统一 `[batch,H]` 和统一 dtype，不能直接满足 mHC 四项混合形状和 FP32 gates。
- `PPMissingLayer` 是 identity 占位，不具有真实层 forward 四元返回的语义；仅迭代 active slice。
- `make_layers` 保留全局编号；不得把每个 Stage 的第一层重编号成 0，不得重复 expand 或提前 contract。
- `GroupCoordinator.isend/irecv_tensor_dict` 已通过 metadata 传 key/shape/dtype；允许任意张量 key 和混合 dtype。其 TP 优化在 numel 可整除时按平坦元素切片，接收再 all-gather。**默认要求模型交付给它的是每个 TP rank 上相同的完整逻辑 tensor**，不能把不同 SP shard 当作复制值交给它。

### 5.2 方案比较（建议，尚未得到用户批准）

| 方案 | 优点 | 代价/结论 |
|---|---|---|
| A：传四元延迟状态，接收端继续 fused post+pre | 不丢状态，保留 PP1 的层间融合及舍入路径；直接对应用户约束 | 多传 `[N,H]` 和小量 gates；**推荐首版** |
| B：发送端先 hc_post，只发 `[N,S,H]` 完整 streams | 通信略少，已有 PR 先例 | 拆开融合，需独立 kernel 数值比较；在完成 post 后才可清 p/C，绝不能直接清；不作为首版 |
| C：更改通用 PP 协议携带 opaque state | 无必要优势 | 现有 tensor dict 足够，不采用 |

不把方案 B 说成必然数学错误；它与 A 的差别是 kernel 融合/精度及验证成本。推荐 A 是工程取舍，未宣称用户已经选择。

### 5.3 首版 mHC wire schema

```text
hidden_states: [N,H]   BF16
residual:      [N,S,H] BF16
post:          [N,S,1] FP32
comb:          [N,S,S] FP32
```

全局所有普通 Stage 边界使用同一 schema，不随 KDA/MLA 改变。R/p/C 非 None；接收后按原值进入本地第一层；它的全局 idx>0，post 存在，走原有 fused 路径。工厂分配零张量用于 profiling，而不是未初始化门控值；batch_size 是 runner token capacity，不是 request 数。H/S 取 config，不写死4096/4。此 tag 的 mHC reference 要求 BF16，首版 mHC PP 明确限制 BF16 activation；不假装 FP16/FP32 工厂等于这些模式已验证。

通信量（逻辑完整 payload，尚未计 TP 优化）：BF16 时每 token 为 `2H(1+S)+4(S+S²)` bytes。H4096/S4 时 41,040 bytes，约40.08 KiB；N8192 时约320.625 MiB。方案 B 为32 KiB/token，约256 MiB/8192 tokens。实际 PCIe 每对 rank 字节数受现有 TP-slice 优化影响，不据此直接估算吞吐；还需计 staging buffer、SP gather 临时内存和 pipeline 并发。

### 5.4 SP 与非 mHC

- 目标 TP4/DP1 下，`use_sequence_parallel_moe` 的现有条件要求 DP>1，所以默认关闭 SP。TP4 不等于 SP 已打开。
- SP 开启的扩展布局：发送端对四项各自 flatten trailing dimensions 为2D，`sp_all_gather`，裁掉 padding 到 N，再恢复原形状，保证交给 PP 层的都是复制的完整 N tokens。接收端对四项**一致地** `sp_shard`，不能只 shard x。p/C 保持 FP32。
- 不更改 SP 的 collective 或 MoE 开关；在 model 边界增加布局规范化，避免 PP 通信的 TP 优化误拼不同 shard。
- 非 mHC 工厂沿用 hidden/residual 二元 `[N,H]`、model dtype；普通 decoder 返回的 residual 不为 None，但下一层非 mHC 会覆盖它，不改变原有计算。MTP 不复用这条主模型 PP schema。

### 5.5 各种 Stage 边界

| 边界 | 处理 |
|---|---|
| KDA→KDA | 四项完整传递；KDA cache 留在各自 layer 所在 rank |
| KDA→MLA | 同一四项；接收 MLA 用自己的 KV/indexer cache |
| MLA→KDA | 同一四项；不传 sparse indexer workspace |
| MLA→MLA（其他合法 layer_types） | 同一四项；双方各自执行本层 indexer |
| dense MLP→MoE | schema 不变，不改变 EP/TP 分派 |
| 首 Stage | embed/inputs_embeds；仅 global layer 0 expand |
| 末 Stage | 仅 global layer 44 完成 post+contract，再 final norm/head |
| PP1 | 保留原路径及融合，不增加通信或边界 materialize |
| 非法/空 Stage | 配置阶段拒绝，不能伪造四项或跨过全局最后层 |

45层默认 PP4 分区按本 tag `get_pp_indices` 是 **11,11,12,11**，不是12,11,11,11。测试另覆盖 `12,12,12,9`、`14,12,12,7` 及边界靠前的非均匀分区；这些是测试布局，不承诺性能最优。任意纯 KDA / 纯 MLA Stage 的缓存分组兼容性尚需实机验证，不能照搬 Metal 的限制或宣称已支持。

## 6. KDA/Mamba、MTP、aux 与 MoE 风险

### 6.1 Hybrid cache

KDA 从 `GatedDeltaNetAttention` 获取本层 cache/metadata，状态属于“层×请求”的持久 cache，不是相邻层激活，不应放进 PP intermediate dict。

`kda_state_shape`：local_heads = linear_heads/TP；head_dim=D；conv state 为 `(3*heads*D/TP, kernel_size-1+num_spec)`，可按全局 conv layout 转置；recurrent state 为 `(heads/TP,D,D)`，外部另有 cache block/slot 维。conv dtype 跟 mamba_cache_dtype/model dtype；recurrent 默认 FP32，也允许显式 BF16。TP4 且64 heads/D128 时 local_heads16、conv_dim6144、基础conv宽3，recurrent `[16,128,128]` 每 slot/layer。

要验证：KV specs 保留全局 layer name，各 Stage cache group 的合并、block alignment、KDA checkpoint、chunked prefill、request slot 复用/抢占、prefix cache、MTP verify/reject 回滚。初始用无 prefix caching、无 speculation、eager 建立基线，然后逐项打开。不能通过重置 recurrent state 假装跨 Stage 正确。

### 6.2 MTP

MTP 是全局主层之后的独立 drafter；这里 `is_mtp_layer=True` 不执行 mHC，不能直接用主模型的四项工厂去切分 drafter。主模型末 Stage 输出保持 `[N,H]`，供 draft proposer 使用。

本 tag `Glm5NextMTP.load_weights` 先去掉多模态前缀，再调用 `get_spec_layer_idx_from_weight_name`；主模型 `model.embed_tokens.weight` 返回 None，被丢弃。`LLMBaseProposer._maybe_share_embeddings` 在 PP>1 不共享首 Stage embedding，而末 Stage 主模型 embedding 又是 `PPMissingLayer`。故不能只观察“服务可启动/文本正常”来验证 MTP。

首版设计将 MTP 权重修复独立成提交：PP>1 时接受已经 prefix-normalize 的主 embedding，加载到 drafter 自己的 `model.embed_tokens`；PP1 保持原共享策略。用实际 `get_pp_group().world_size` 判断所在并行环境，并在Glm5NextMTP上设置 `has_own_embed_tokens = (pp_world_size > 1)`，满足V2 `maybe_share_target_embed` 在target embedding缺席时的独立embedding声明；此flag不替代真实加载记录。MTP 自带 embedding 优先于主 embedding，重复的同名/别名输入需确定性处理。流式加载不缓存巨大 embedding：允许先载主 embedding，后出现 MTP 专用 embedding 则覆盖；反序时跳过主 embedding。若二者都缺，在 PP>1 load 结束时明确报错，不能把 torch.empty 内容当权重。分别记录是否加载、来源和真实 loaded_params。

该修复不承诺完整 PP+MTP 支持：V1/V2 proposer 路径、多模态特殊 token、kv index reuse、acceptance 与 rejection state 仍需 GPU 验证；不能仅凭 has_own_embed_tokens 标志证明加载了权重。draft-only checkpoint 若不含 embedding，首版报错并给出补齐 checkpoint 的方法；不自动联网再抓另一份巨大 target checkpoint。

### 6.3 aux_hidden_state_layers / EAGLE3 / DFlash

当前 mixin 的 `supports_aux_hidden_states_over_pp=False`；model 只收集本地 aux，普通边界会丢掉上游 aux。**首版不把这个 flag 改 True**。普通 PP 生成可实现；PP+需要跨 Stage aux 的方法继续明确拒绝。真实非空 aux layer 请求应在模型入口增加明确错误，覆盖绕过 runner guard 的调用；PP1 的 aux 行为不变。

扩展方案已明确但不混入首版：使用 mixin `pack_local_aux_hidden_states/collect_remote_aux_hidden_states` 与 V2 relay/reserve helpers；按输入到第k层的语义定义 aux[k]，边界k只由上游产出一次；所有 aux为完成 hc_post并contract后的 `[N,H]`，不能拿原FFN x冒充；非末 Stage也要收集end_layer位置，接收端不再重复start_layer。需单独验证factory slots、全局排序、SP gather、V1/V2差异和runner公共入口后才开启能力旗标。本文没有授权改通用runner。

### 6.4 MoE/EP

make_layers 保持每个 Stage 的本地专家层集合；不改 routed experts、shared experts、router precision、EP all-to-all 或 EPLB。多模态 wrapper 的 `set_moe_parameters` 已跳过 PPMissingLayer，但局部 Stage 无 MoE 时元数据完整性、expert mapping、各 Stage的负载均衡需要验证。首次验收 EP关闭、DP1；之后再验证EP扩展，不把不同并行模式的数据混作PP正确性证据。

ROCm fused shared experts 当前已有 gfx950、TP4/8、DP1、无EP等条件；保留已有判定，不能为PP去强开。四个PCIe Switch不自动等于rank拓扑，须实测。

## 7. 权重加载设计

`Glm5NextModel.load_weights` 的 FP8 特殊助手在常规 `is_pp_missing_parameter` 检查之前执行；助手组齐 weight+scale后会直接访问 `params_dict[fused_name/target_w]`。即使后面的 stacked/expert/default 路径会跳过远端层，这里仍可能 KeyError，或先无谓解量化其他 Stage 的权重。

修复点：在主模型流式循环内，拿到真实层前缀后、进入任何 FP8 pending buffer/解量化之前，用 `is_pp_missing_parameter(name,self)` 过滤远端层/embedding/norm。保持本地未知参数报错，**不能**用 `if name not in params_dict: continue` 通吃，避免把拼写错误或真正缺权重静默忽略。保留本地 FP8成对加载、shard顺序、expert映射及NoPE padding。上层 `AutoWeightsLoader` 对有自定义 load_weights 的子模块会委托整个迭代器，不能假设它已按主模型的层切好权重。

首 Stage 加载 embedding；末 Stage加载norm/head；其余相应参数由PPMissing过滤。checkpoint `tie_word_embeddings=false`，不新增head/embedding绑定。多模态前缀先由现有mapper规范化；MTP的prefix单独按现有逻辑处理。

## 8. 计划修改清单（目前均未实施）

| 文件 | 类/方法及修改 |
|---|---|
| `vllm/models/common/mhc_pp.py`（新增） | 轻量纯torch工厂、四元pack/unpack及结构校验；仅模型层数据schema，不包含通信、distributed分组或kernel选择。放在轻量common包，避免CPU测试导入GLM5Next包时触发GPU/MTP/triton依赖。 |
| `vllm/models/glm5next/common/model.py` | Glm5NextModel：绑定工厂；forward四项接收/发送；SP四项gather/shard；返回类型；非空stage和aux限制；load_weights提前PP过滤。Glm5NextForCausalLM和Glm5NextForConditionalGeneration分别alias到内层工厂。保留decoder的真实mHC计算。 |
| `vllm/models/glm5next/common/mtp.py` | PP drafter embedding 加载、来源/优先级跟踪、缺失错误；保留PP1共享。 |
| `tests/models/glm5next/test_pp_transport_cpu.py`（新增） | 真实schema helper、IntermediateTensors切片/复制、混合dtype、数学连续性与负例，CPU运行。 |
| `tests/models/glm5next/test_pipeline_parallel.py`（新增） | 真实模型构造、forward/加载/包装回归；用GPU/ROCm运行，不能用假decoder冒充。 |
| `tests/v1/e2e/spec_decode/test_mtp_parallel_load.py`（扩展） | 真实checkpoint/加载入口，检查MTP embedding来源及缺失错误。 |
| `tests/distributed/test_pipeline_parallel.py`（扩展） | 通过LLM/现有比较工具验证实际PP1对PP4；小型真实checkpoint常规CI，完整模型专用16GPU验收。 |
| `.buildkite/test_areas/models_basic.yaml`、`distributed.yaml`（按测试入口需要调整） | 将新回归选入实际执行job，source_file_dependencies覆盖新增helper和model/mtp。大模型手动验收不得硬塞常规4GPU job。 |
| `docs/models/supported_models.md`（核对真实表项）及GLM部署说明 | 记录PP能力范围和限制；在GPU验证前明确标成待硬件验证，不能直接宣称已认证支持。 |

轻量helper提供：`make_empty_mhc_intermediate_tensors(batch_size,dtype,device,*,hidden_size,hc_mult)`；`pack_mhc_intermediate_tensors(x,residual,post,comb,...)`；`unpack_mhc_intermediate_tensors(...)`。固定key顺序，返回真实IntermediateTensors/四元组。结构验证只读shape/dtype/device，不`.item()`、不CPU同步、无依token数的Python数据读取；必须通过torch.compile/graph测试。普通PP1不走新增pack路径。实现者可在不改变wire contract的前提合并过细helper，不需要照此拆成大量小函数。

## 9. 异常与兼容策略

- PP>1 且分区含0/负数、无层、超出主层范围：在模型分区确认后拒绝；通用partition parser继续保留不改。
- 非首Stage缺intermediate；四项缺一；token数不一致；R/p/C形状错误；p/C非FP32；device不一致：尽早给出包含key/expected/actual的错误，绝不静默fallback到None。
- mHC PP activation dtype不是BF16：首版明确拒绝；非mHC保持原有dtype处理，不扩大其已验证范围。
- profiling工厂batch=0与正常batch：分配合法zero-size/zero-filledtensor；不据此强迫真实KDA kernel执行零token请求，运行时按scheduler已有规则跳过。
- 空Stage先拒绝，因此不需要定义“首Stage无层时传embedding还是四元组”的歧义协议。
- post已完成与未完成不能混用；schema固定对应非全局最后层的未完成状态。最后层不会产生PP发送payload。
- 缺失本地权重与远端Stage权重区别处理；只跳PPMissing，不吞本地错误。
- PP+aux方法明确拒绝；PP1原aux不受影响。MTP允许在权重完整时进入GPU验证，但缺embedding必须fail closed。
- 不启用新attention/ROCm backend，不更改mamba/cache dtype默认值，不把FP8量化权重当FP8激活。

## 10. 测试方案与验收门槛

### 10.1 CPU可运行的合同和数学测试（批准后添加）

新增CPU测试不import GLM5Next package或GPU kernels；正常import轻量common helper与真实IntermediateTensors。必要时按文件路径加载现有纯torch mHC数学reference以避开父包注册GPU内核；这只验证数学，不假称真实CustomOp/模型已经执行。

1. 工厂N=0/1/3/17，多种H/S；断言每个key的shape/dtype/device、零初始化；post/comb必须FP32，N切片无错位。
2. 对工厂持久buffer执行与runner相同的逐key copy_/slice，多次大小交替，所有四项逐元素相等；使用FP32中不能被BF16精确表示的门控数值以检出误降精度。
3. 通过生产pack/unpack传递真实非平凡张量；缺key、None、错shape/dtype/device（CPU可测部分）、不同token数产生明确错误。
4. 固定seed、BF16 streams和FP32系数的8层mHC数学链：执行真实reference pre/post，显式RMSNorm，sublayer使用确定性的torch运算；连续执行与四段之间经过生产pack/copy/unpack的结果严格相等。枚举8层的全部35种四段非空分区，逐项比较中间四状态及最终norm/logits。**这个结果名称必须是CPU数学链一致性，不是GLM模型PP4通过。**
5. 加入可检测故障：丢p、错comb转置、把FP32门控转BF16、错token切片应改变预期结果或被合同拒绝，避免测试恒成立。
6. 非mHC二元路径保持原计算；SP的token维布局用TP=4、N=1/3/4/5/17验证padding、恢复和裁剪；这是布局测试，不是RCCL collective测试。

CPU运行使用uv创建独立venv、安装固定版本CPU torch/pytest后运行。用`pytest --noconftest`执行自包含CPU文件，避开仓库顶层conftest导入整套GPU/多模态测试依赖；这项依赖隔离必须实测，不能只标记“CPU-only”就算可运行。Windows解释器用`.venv\Scripts\python.exe`，Linux用`.venv/bin/python`。不使用系统pip或裸python安装。

### 10.2 真实模型与加载回归（需相应vLLM构建/硬件）

- 用真实VllmConfig、Glm5NextTextConfig和真实构造器，小型且attention head/quant shape合法的hybrid checkpoint；不替换attention、decoder、norm或MTP类，不用SimpleNamespace/MagicMock伪造内部状态。
- 真实分布式PP1、PP2、PP4；PP4各非空布局包含三类真实邻接边界，以及dense→MoE。不同layer_types配置另覆盖MLA→MLA。比较每个阶段输出四项、末层hidden和logits，不只比较可启动。
- 权重覆盖主模型BF16/FP8（本地层和远端层交错、scale在前或后、不同shard顺序），embedding/norm/head stage ownership，多模态mapper；不完整本地权重必须失败。
- MTP：主embedding、自带embedding、两者不同顺序、都缺、MM前缀，PP1共享与PP>1独立；真实parameter checksum与指定checkpoint分片对应。
- aux：PP1继续返回正确特征；PP>1明确拒绝。不得绕过guard后默默少返回aux。
- 上游现有GLM5Next tests、mHC ROCm相关用例和MM processing回归按实际影响运行。
- 用户可见回归至少经过`LLM`/`vllm_runner`或`RemoteOpenAIServer`入口；实际测试必须证明基线失败和补丁成功。若硬件未到，报告not run，不能删测试或放宽阈值换绿。

### 10.3 PP1与PP4数值标准

- 固定checkpoint revision/权重、token ids、tokenizer revision、dtype、attention/MoE backend、TP4、DP1、seed、请求顺序、cache设置；唯一主变量是PP1/PP4与分区。
- 不用TP1/PP1对TP4/PP4替代核心对照；TP改变归约路径。完整模型TP4/PP1若装不下，用更大显存/CPU offload且单列影响，或同结构小checkpoint先做TP4对照；不能把没有完整模型基准写成已验证。
- CPU传输和相同运算顺序要求bitwise exact；GPU先报告逐层max_abs/max_rel/RMS、NaN/Inf、logits与top-k/logprobs。阈值在补丁前对同配置重复运行的基线漂移及现有`check_logprobs_close`约定上固定，不事后放宽。
- greedy token相等作为有用但非充分检查；FP32 gates、完整hidden/logits和cache连续性也必须比较。非batch-invariant运行不能只凭token差异判断PP错误，要记录batch/chunk等条件。
- 覆盖短/长prefill、1/3/5等非TP整除token数、多步decode、chunked prefill、混合prefill/decode、并发请求、结束/取消/slot复用；之后打开prefix cache、graphs、MTP分别对比。

### 10.4 静态检查和补丁

批准后：ruff check、ruff format --check、相关mypy/pre-commit hooks、目标文件compileall、git diff --check。遵循仓库AGENTS，安装lint依赖与pre-commit。无实现前不把现有代码静态扫描当成新功能检查通过。

最终交付含：精确commit列表、实际改动文件/方法、`git format-patch v0.33.0..HEAD`导出补丁、CPU测试逐条结果、未执行GPU测试、AMD验证日志路径。`git format-patch`必须覆盖真实修改；本次保存的外部PR patch不能当作本项目实现补丁。

## 11. 未来AMD / ROCm 10.0 验证步骤

ROCm10.0是用户目标环境，本机没有该硬件/软件栈。以下是届时执行的步骤，不承诺v0.33.0的原始依赖矩阵能直接搭配ROCm10.0；不能提前凭空写镜像tag或性能结论。

1. 固定OS/kernel/driver、ROCm10.0具体build、PyTorch HIP、Triton/AITER/TileLang/RCCL、vLLM基线+补丁commit、模型revision与量化。先验证编译/import及单GPU算子；有兼容性错误单列为平台问题，不能扩大PP补丁顺手改内核。
2. 采集`rocminfo`、`amd-smi list`、`amd-smi topology --help`所列可用拓扑命令及`lspci -tv`；保存GPU BDF、NUMA、四组Switch映射。不要只按GPU序号猜。运行P2P/RCCL send-recv、all-reduce、all-gather带宽/正确性工具，记录跨Switch路径。
3. 确认vLLM默认rank顺序后安排每个Stage的4个TP rank在同一Switch，PP链对应跨Switch连接；使用当前ROCm实际支持的可见设备机制（如ROCR_VISIBLE_DEVICES），记录映射日志。不要同时设置互相冲突的device env。
4. 小型真实GLM5Next checkpoint先验证TP4/PP1与TP4/PP4/DP1，eager、EP关、MTP关、prefix caching关；CPU helper tests再在此环境复跑。
5. 完整模型先用明确分区`11,11,12,11`，再测`12,12,12,9`和`14,12,12,7`，记录每Stage的KDA/MLA/MoE层数、权重/activation/KV memory及耗时。分区选择按实测内存与延迟，不把层数平均当成负载平均。
6. 执行10.3的数值对照；完整模型TP4/PP1装不下时显式报告未验证，并保留小checkpoint真实PP对照。full模型只完成PP4运行仍不满足数值验收。
7. 逐项打开graph/compile、prefix cache、chunked prefill、MTP；MTP验证embedding checksum、acceptance、accepted tokens/step、target logits和reject后的KDA状态恢复。多模态图像/视频单独对比PP1/PP4；确认vision只在预期rank运行。
8. 长上下文、取消/抢占、并发、多请求slot重用压力测试，随后才测TTFT、TPOT、aggregate throughput、p50/p95/p99、跨Switch传输量、PP气泡、每卡memory峰值。16GPU的容量与吞吐收益分别报告。

首轮候选启动参数（不是验证结果；具体backend与显存参数现场确定）：

```bash
export VLLM_PP_LAYER_PARTITION=11,11,12,11
vllm serve /models/GLM-5.3-Flash \
  --tokenizer-mode glm5_next \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 4 \
  --data-parallel-size 1 \
  --dtype bfloat16 \
  --distributed-executor-backend mp \
  --enforce-eager \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

此命令初步只验证文本路径；图像/视频验证要去掉对应limit并固定输入。ROCm稀疏MLA/KDA backend按GPU型号和实际支持选择，不在未检测环境下硬编码AITER开关。

## 12. 本轮报告与批准入口

已执行：下载/核对tag与commit、建立分支、用户指定grep、模型/通信/runner/加载/测试源码分析、联网先例检索、下载参考PR patch并只做适用性检查。源代码与tests相对v0.33.0的diff为零。

未执行：新CPU单元测试、新实现lint/compile检查、vLLM构建、模型权重下载、GPU/NCCL/RCCL/ROCm测试、PP1/PP4数值对照、性能测试。没有可报告的“测试通过数”。

等待批准的具体设计：**A方案四状态透传 + 模型边界SP规范化 + 主权重提前PP过滤 + 单独MTP embedding修复；首版PP+aux显式拒绝；CPU合同/数学测试与真实GPU验证分开。** 批准后先按writing-plans细化实施顺序并建立回归，再写产品代码。本次文档的存在不是实施授权。

独立builder check已完成：审阅者对照本地v0.33.0源码检查四项schema、FP32 gates、SP、加载过滤、MTP及验证分层，未提出阻碍制定实施计划的遗留问题。此结果是设计审阅，不是实现review或数值测试通过。
