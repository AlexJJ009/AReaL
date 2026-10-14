# yyht/miles-values SAO 实现核查

日期：2026-09-22。实时 GitHub API 确认默认分支 `main-0618`，HEAD
`aa2071f834ff9666a9bee124f7ed62e260b7488c`。本地完整、非 sparse checkout 与该 SHA
一致，未修改其代码，未运行训练。

## 结论

存在真实接入的 SAO 相关组件，尤其是双 lambda、mask-aware GAE、critic 多轮更新和独立 checkpoint
加载。不能据此认定公开版本已经完整实现论文 SAO：默认 actor loss 仍为 PPO clipping，未找到接入的 Direct DIS；公开4B数学示例也是 n=8
PPO。README 的“single-rollout PPO (SAO)”是一种宽泛命名，不能替代算法等价证据。

## 实际代码对应

| 机制             | 已核实路径                                                                                  | 结论                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Policy loss      | `loss_hub/losses.py:99,183` → `math_utils.py:232`                                           | `max(-rA, -clip(r)A)`；ratio参与求导，非双侧越界一律清零                                       |
| Behavior来源     | `use_rollout_logprobs` 可直接选 rollout logs，否则用训练端旧logprob                         | 能选择直接ratio分母，但仍是PPO surrogate                                                       |
| TIS/IcePop       | `corrections.py:7,35`，经 `losses.py:218–234` 调用                                          | 用缓存训练logprob／rollout logprob的ratio另乘PPO loss；不是当前forward直接DIS                  |
| Dual lambda      | `arguments.py:1044–1050`；critic worker `actor.py:397–400`；`advantages.py:77–102`          | critic-lambd默认1，critic override会强制fixed；actor独立计算，确实接入                         |
| Adaptive lambda  | `gae_adaptive.py:46`                                                                        | 公式存在；默认alpha=1、min_length=50，短于max(min_length,2)用lambda=1，与论文alpha=1.5存在差异 |
| Skip-observation | `gae_adaptive.py:68,100,155`                                                                | mask=0不推进next_value/GAE，长序列压缩action子序列做chunked scan；最终奖励放最后有效action     |
| Critic updates   | `actor.py:411–423` 循环num_critic_epochs                                                    | 每轮train可能含多个global batch/optimizer steps；不能直接当作严格两次optimizer更新             |
| Colocated顺序    | `train_async.py:46–58`                                                                      | critic先train，actor后train；分离部署则并行并通过NCCL交换value                                 |
| Value snapshot   | critic `actor.py:380–390`先forward，`425–430`返回原values；actor `495–515`接收并算advantage | 共置路径actor取得更新前value，没有更新后forward刷新                                            |
| Pretrained value | `--critic-load`，worker按critic角色加载                                                     | 可接独立checkpoint；README称独立job预训练，不证明任意checkpoint已合格                          |
| Frozen attention | 在主训练路径未找到论文式attention冻结策略                                                   | 不能声称已实现；freeze-indexer是不同功能                                                       |

上述行号绑定固定SHA，不绑定未来分支。SAO并非必须以sao函数命名；“未找到Direct
DIS”依据实际loss分派、surrogate与correction调用关系，而不是只按关键词搜索。

## 公开4B脚本的实际选择

`scripts/colocate_ppo/run_qwen35_4b_ppo_colocate_critic_async_8gpu.sh`：4 actor
GPU与4独立rollout GPU，Megatron+SGLang；每批16题×8回答=128轨迹；默认actor lambda=.95、critic
lambda=1、critic epochs=1、critic-only steps=10、PPO clipping
.2/.28；advantage_estimator=ppo，没有开启gae_adaptive或TIS/IcePop/直接rollout-logprob选项。脚本默认回答长度4096；README另一组数学实验写16K，不能混同为同一配置。

README 数学实验设置 n=8、critic_update_steps=1，不能充当严格single-rollout SAO结果。README也报告agentic
single-rollout实验和图表，但本次未获得其逐次实验的完整resolved config、checkpoint与原始日志映射，不以图表反推当前代码实现了完整DIS。

## Ratio stop-gradient问题

该仓库PPO路径写 `-r*A`，r不detach，正是常见ratio surrogate。TIS所用缓存训练logprob与rollout logprob不是当前policy
forward的可微ratio，不能把该权重与当前ratio混为一谈。它不提供论文 `-f(r)*A*logpi` 是否detach的官方答案。

本次抽取原始compute_policy_loss运行CPU
autograd：即使改用论文epsilon=.3/5，r=.5且A=+1时梯度=-.5；r=6.5且A=-1时梯度=+6.5。两个ratio均在论文DIS区间外，却保留PPO梯度，因此二者不等价。未执行整个Miles测试集，也未验证分布式/GPU路径。

## 对我们的用途

可参考其已接入双lambda的dispatch、skip-observation scan、独立critic
checkpoint接口和value诊断。不能直接继承loss、默认配置或更新后value刷新假设。保留现有AReaL/FSDP2/SGLang栈，按论文实现Direct
DIS及用户已确认的critic两次更新后刷新value/advantage的时序。

## 固定源码链接

- [Policy loss](https://github.com/yyht/miles-values/blob/aa2071f834ff9666a9bee124f7ed62e260b7488c/miles/backends/training_utils/loss_hub/losses.py#L183)
- [PPO ratio surrogate](https://github.com/yyht/miles-values/blob/aa2071f834ff9666a9bee124f7ed62e260b7488c/miles/backends/training_utils/loss_hub/math_utils.py#L232)
- [TIS / IcePop](https://github.com/yyht/miles-values/blob/aa2071f834ff9666a9bee124f7ed62e260b7488c/miles/backends/training_utils/loss_hub/corrections.py#L35)
- [Dual lambda dispatch](https://github.com/yyht/miles-values/blob/aa2071f834ff9666a9bee124f7ed62e260b7488c/miles/backends/training_utils/loss_hub/advantages.py#L77)
- [Mask-aware GAE](https://github.com/yyht/miles-values/blob/aa2071f834ff9666a9bee124f7ed62e260b7488c/miles/backends/training_utils/loss_hub/gae_adaptive.py#L46)
- [Critic train / values](https://github.com/yyht/miles-values/blob/aa2071f834ff9666a9bee124f7ed62e260b7488c/miles/backends/megatron_utils/actor.py#L380)
- [Async orchestration](https://github.com/yyht/miles-values/blob/aa2071f834ff9666a9bee124f7ed62e260b7488c/train_async.py#L46)
- [4B example](https://github.com/yyht/miles-values/blob/aa2071f834ff9666a9bee124f7ed62e260b7488c/scripts/colocate_ppo/run_qwen35_4b_ppo_colocate_critic_async_8gpu.sh#L46)
