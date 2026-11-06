# GLM5Next PP 实现与验证报告

日期：2026-11-06。源码目录：`D:\work\vllm-glm53-pp`。
分支：`research/glm53-flash-pp-v033`。上游基线：v0.33.0，`4ba787ec9959e566d3b587a796dff07673324da5`。

## 当前结论

已实现四状态PP边界、模型及多模态工厂委托、SP边界布局处理、远端FP8权重提前过滤，以及PP下MTP embedding独立加载/优先级/缺失错误。没有修改通用PP通信、TP/DP/EP分组或mHC计算内核。

**37项CPU测试通过；真实vLLM模型、多GPU PP1/PP4、ROCm/RCCL、图像/视频数值一致性和性能均未验证。此交付是可审查、可回退的实验性补丁，不是已验证的16GPU部署。**

## 改动文件与行为

| 文件 | 变更 |
| --- | --- |
| `vllm/models/common/mhc_pp.py` | 新增轻量工厂、pack/unpack及shape/dtype/device校验；所有张量第0维为token，p/C固定FP32，支持持久buffer切片；验证torch.compile fullgraph动态token。 |
| `vllm/models/glm5next/common/model.py` | Glm5NextModel四项收发，不清空或提前materialize p/C；SP对四项统一gather/shard；检查非空分区与BF16；拒绝PP+aux；主模型FP8助手之前过滤PPMissing；两个包装类委托空中间张量工厂。 |
| `vllm/models/glm5next/common/mtp.py` | 使用实际PP group声明独立embedding；加载主embedding兜底，首个MTP层的显式embedding优先，顺序无关；PP下二者皆缺时报错；保留PP1共享。 |
| `tests/models/glm5next/test_pp_transport_cpu.py` | 工厂、缺失/畸形输入、混合dtype、零token、持久buffer、动态fullgraph和35种四段数学链分区。 |
| `tests/models/glm5next/pp_fixtures.py` | 真实Transformers构造、原checkpoint命名保存的小型17层hybrid模型，含可选独立MTP权重；约3255万参数。 |
| `tests/models/glm5next/test_pp_checkpoint_cpu.py` | 实际验证上述checkpoint的原始权重名、shape、mHC FP32、MoE、KDA、MLA和MTP fallback布局。 |
| `tests/models/glm5next/test_pipeline_parallel.py` | 使用真实构造器检查文本/多模态、mHC/非mHC的实例接口和工厂；本机未运行。 |
| `tests/distributed/test_glm5next_pipeline.py` | 公共LLM入口PP1/PP4 full-vocabulary logprobs、三种分区、两种包装类、分区/embedding/norm/head所有权、FP8远端过滤、aux拒绝、PP1/PP4 MTP加载；另有16GPU TP4/PP4对TP4/PP1用例。本机未运行。 |
| `.buildkite/test_areas/models_basic.yaml` | 原GLM5Next单元测试job增加common helper source dependency。 |
| `.buildkite/test_areas/distributed.yaml` | 新增4xB200 nightly job及MI355 mirror，执行非16GPU的真实PP测试；16GPU用例单独手动执行。配置存在不代表CI已运行。 |
| `docs/models/supported_models.md`、`glm5next_pipeline_parallel.md` | 实验性状态、限制、测试命令及AMD验收步骤；未将PP标为已认证支持。 |

模型wire状态为：x `[N,H]` BF16；R `[N,S,H]` BF16；p `[N,S,1]` FP32；C `[N,S,S]` FP32。首层expand、末层post/contract/norm/head的原逻辑保留。新增helper放在轻量common包，CPU测试不会导入GLM5Next的GPU/MTP内核。

## 本机已运行的验证

环境：Windows，uv管理的Python 3.12.10，PyTorch `2.14.0+cpu`，Transformers `5.20.2`，pytest `9.2.0`，Ruff `0.14.0`。`torch.cuda.is_available()`为False。

```powershell
.venv\Scripts\python.exe -m pytest --noconftest `
  tests/models/glm5next/test_pp_transport_cpu.py `
  tests/models/glm5next/test_pp_checkpoint_cpu.py -q
```

实际结果：**37 passed**。这37项中包含对全部35种四段非空数学分区的循环断言，不把35种分区另外虚增成35项pytest测试。

| 检查 | 结果与范围 |
| --- | --- |
| CPU合同/数学/checkpoint套件 | 37 passed，无warning；零GPU |
| RED→GREEN：新transport模块 | 先收集失败 `ModuleNotFoundError: vllm.models.common.mhc_pp`，实现后通过 |
| RED→GREEN：标量hidden_states | 先1 failed/29 passed，原因IndexError；新增维数错误检查后整套通过 |
| Ruff check及format | 使用仓库固定0.14.0，对新增/修改Python文件通过；相关pre-commit Ruff hooks也通过 |
| compileall | 修改的产品代码与新增测试语法编译通过 |
| mypy | `--follow-imports=skip` 对3份产品源码通过；这是限制导入范围的检查，不是仓库完整mypy通过 |
| git diff --check | 已运行，无空白错误 |
| CI tethering / Buildkite step keys / SPDX | 新测试路径绑定、job key及Python文件版权头检查通过；Windows tethering脚本使用`-X utf8`运行 |
| Markdownlint | 使用系统Node运行同版本0.21.0，模型说明、实施计划及本报告通过 |
| checkpoint生成器 | 真实HF模型初始化与safetensors保存/读取通过；不代表vLLM加载或推理通过 |

CPU数学参考显式应用RMSNorm；没有直接把现有native/CPU mHC wrapper当作GPU fused oracle。传输复制要求逐元素精确相等，未通过降低容差使测试变绿。
TP4布局测试覆盖N=1/3/4/5/17的padding、四项分片payload复制、顺序恢复及裁剪；它验证布局代数，不是实际SP collective。

## 无法执行或失败的环境检查

- 真实模型构造测试收集时因完整运行依赖缺失（`cbor2`）失败；distributed测试收集因`msgspec`缺失失败。没有将这些导入错误当成实现前的功能RED，也没有标记测试skip/xfail制造通过。
- 尝试按`requirements/common.txt`安装完整运行依赖，下载`pycountry==26.2.16`时TLS连接中断，安装未完成。没有在Windows无GPU环境完成vLLM编译/import验证。
- 安装了仓库pre-commit hooks，但全量hook环境初始化下载Node LTS时连续发生IncompleteRead。相关Ruff hooks单独通过；本地提交临时绕开全量hooks以保存可回退提交，未关闭持久配置，也未推送或合并。
- 完整仓库mypy/pre-commit、GPU模型eval、真实PP对照、SP collectives、ROCm10.0兼容性均不能记为通过。

## 必须在硬件上继续验证

1. 在兼容vLLM的Linux CUDA/ROCm构建上，执行真实构造测试和4GPU distributed套件，确认原始v0.33.0失败、补丁通过。补丁当前没有这些RED/GREEN运行证据。
2. 用16GPU运行同一小checkpoint的TP4/PP1对TP4/PP4；固定TP、DP、dtype、backend、batch/chunk、seed，记录hidden/logits、max_abs/max_rel/RMS及NaN/Inf。现有端到端断言覆盖全词表prompt/sample logprobs；逐层hidden/state dump仍属于硬件验收步骤。
3. 官方完整45层模型先用11,11,12,11，再比较12,12,12,9和14,12,12,7。完整模型TP4/PP1若装不下，明确保留“完整模型数值未验证”，不能用TP1对照替代。
4. 单独验证真实SP、EP扩展、纯KDA/纯MLA分区、graph/compile、prefix cache、抢占/取消/slot复用、长上下文KDA恢复、MTP接受/拒绝回滚、多模态。首版PP+aux仍明确拒绝，未承诺EAGLE3/DFlash跨Stage aux。
5. 在用户目标ROCm10.0环境固定driver/PyTorch/Triton/AITER/TileLang/RCCL与模型revision；采集PCIe/NUMA/P2P拓扑，将同Switch的4卡映射为TP group，先数值再性能。未预设可用镜像tag或硬件吞吐。

详细设计与AMD步骤见[已批准设计](../specs/2026-11-06-glm53-flash-pp-design.md)，可运行命令见[模型说明](../../models/glm5next_pipeline_parallel.md)。

## 实施决定与限制

- 继续在用户指定的新克隆及独立分支工作；没有额外创建worktree，没有再次请求已批准的实施权限。
- 按已批准范围实现普通PP、MTP加载修复；PP+aux继续拒绝。SP边界代码已实现，但本机没有真实collective验证。
- 因运行环境不足，模型/多GPU回归的RED/GREEN证据未获得；保留真实测试与失败日志，交付为实验性补丁。
- 保留37项CPU验证与静态检查的实际范围；全量hooks初始化失败不等于相关Python检查失败，也不等于整套pre-commit通过。

补丁导出以Git提交为准，不使用此前下载的PR #57171参考patch冒充本项目实现。所有远程PR/merge/push均未执行。
