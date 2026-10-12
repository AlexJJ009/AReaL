# README §16 数学审稿：ICML 标准

审阅对象：§16.11（README L2243–2743），并核对 §16.0、§16.2–16.4、§16.9–16.10、§3.4 和 §9.1。以下 `L…`
均指本次读取的
[README.md](/data_storage/yl_test/lgx/data-1/code/_worktrees/AReaL/codex-sao-math/docs/research/2026-10-09-tau2-high-baseline-audit/README.md)
行号；代码路径相对于仓库根目录。审阅日期：2026-10-12。本审阅只读取材料、复算数值；没有修改原稿、代码或配置，也没有运行训练。

## 1. 总体评价

草案最强的部分是把“用户不可靠”和“agent 尚不会应对用户”分开，并承认用户加权会改变目标；在明确的 common-direction model
下，正权重保持理想梯度方向、inverse-variance weighting 提高所定义的 SNR，这两个条件结论可以保留。最弱的部分是从这些条件结论跳到“最后 agent
token 的 residual square 能识别用户噪声”，又把它推广到实际 DIS、length-adaptive GAE 和 Adam
的效率及稳定性保证。最后位置的事后选择、critic 的动作前对齐、同 batch critic fitting，以及用户反应对 agent
行为的依赖，都使这条推理链断开。SNR 表、ESS 和三个 Kantorovich 数值基本算对了，但它们没有证明当前方法的实际收益，尤其不能推出“权重完全估错也保留至少
45% 效率”。按 ICML 标准，目前核心算法可以作为有动机的 heuristic proposal 存活，核心可辨识性与稳健性论证尚不足以支持接收；L2742–2743
承认没有实验是准确的，但前文仍有多处超出这一证据强度的断言。

## 2. 严重问题（会阻碍接收）

### S1. `last agent token` 不是固定的“agent 已做完，只剩用户随机性”边界

**位置：** L2478–2497、L2512–2513、L2583–2588；对照 L1895–1898、L245–246。

这里同时混淆了两个边界。首先，训练 value 是预测当前 action 的 **pre-action prefix value** ：`actor.py:226–227` 把
action mask 左移，GAE 直接使用对应位置的 `values`。它没有读完将要预测的最后一个 action token。若原始最后 action 在位置
$q$，终末 TD baseline 对应 $V_{q-1}$；读取 $V_q$ 才表示看完该 token，但它不是当前 GAE 的终末 baseline，也不保证有对应位置的
value-loss supervision。

其次，“最后一次行动”是对结束后的轨迹回看所得的位置，其选择依赖未来是否终止。README 自己的例子说明：提前 STOP
的分支停在请求确认；成功分支还会执行写操作，随后才结束。两条分支的最后状态不相同，因此 L2487 把它们共同写成 $V=0.5$，不能代表这个实际过程。

一个最小反例足以推翻表格的普遍含义。设 agent 到确认处都正确，用户以概率 $\rho$ 提前结束；继续分支随后确定地完成写操作。确认前缀的正确预测是
$1-\rho$，成功分支最终行动前的预测约为 $1$。按各自最后位置取 residual，有

$$
\mathbb E[e^2]=\rho[0-(1-\rho)]^2+(1-\rho)(1-1)^2
=\rho(1-\rho)^2.
$$

$\rho=0.5$ 时得到 $0.125$，而非 L2487 的 $0.25$；$\rho=0.9$ 时仅为 $0.009$。一个几乎总提前结束的用户反而可能获得很高的
inverse-residual weight。这个反例不要求 critic 在 active-prefix
分布上不准。若确认前缀被事后选为最后位置，已意味着走到了提前失败分支，该 selection-conditioned 成功率为 0，而普通 critic 的
active-prefix 成功率是 $1-\rho$。此时 $\mathbb E[e]=-\rho(1-\rho)\ne0$，因此 $\mathbb E[e^2]$
甚至不等于这个末位置 residual 的 variance。

**具体修改：** 把测量点定义成能在线识别、无需未来终止信息的固定边界，例如某次完整确认请求之后，并区分 pre-action state 与 post-action
observation。若坚持使用现成末 token value，应称其为“事后选点的 residual-risk proxy”，删除“已经看到所有 agent
行为”“之后只剩用户回复”和“等于第 2 步用户噪声”的一般结论。新边界仍不能自动解决下面的因果可辨识性问题。

### S2. residual square 识别预测风险，不能识别“不可靠”与“可学习的困难”

**位置：** L2386–2389、L2455–2457、L2467–2474、L2490–2497、L2547–2549、L2624、L2678–2679。

在一个合法固定边界，令 $H$ 为可观测 prefix，$p(H,u)=\Pr(R=1\mid H,u)$。对冻结的 critic，准确分解是

$$
\mathbb E[(R-V_\phi(H,u))^2\mid H,u]
=p(H,u)[1-p(H,u)]+[p(H,u)-V_\phi(H,u)]^2.
$$

第一项是**相对于当前信息的条件不确定性** ，第二项是 critic approximation error。第一项可能来自用户随机性，也可能来自隐藏任务条件、不可观测 DB
信息、尚未发生的 agent 行动或工具过程。高 Brier risk 不意味着用户应降权；低 risk 也不意味着用户可靠。只有预测值层面的
calibration，$\mathbb E[R\mid V]=V$，还不够推出 $V(H,u)=p(H,u)$：始终预测总体成功率也可以
calibrated，却没有区分状态的 resolution。

对末位置这种事后选点，严格条件均值应另定义为 $p^*(s,u)=\mathbb E[R\mid S_{\rm last}=s,U=u]$。普通 critic 在所有
active positions 上回归的条件均值，不自动等于这个经过终止选择的分布中的条件均值。

没有重复前缀、seed、反事实干预或结构约束，仅凭一次交互中的 $R,V$，不能把“agent 无法影响的用户扰动”与“agent
可以通过另一种表达/追问影响的失败”分开。两个环境可以在已观察的行为上给出同样的 reward 分布，却在未尝试的确认方式上不同。这是 identification
gap，不能靠平方代替均值补上。

还有一个即便在玩具模型内也存在的错误：$\rho$ 大并不推出 $\sigma^2$ 大（L2386–2387）。取 $q=1,\eta=0$，有
$\sigma_R^2=\rho(1-\rho)$；超过 $0.5$ 后，越不可靠反而方差越小。$\rho=0.99$ 的用户几乎总失败，但其方差 $0.0099$
小于普通用户的 $0.09$。不估 $c_u$ 的 inverse-variance rule 在这里会提高坏用户权重，而不是“部分体现”低 signal。

**具体修改：** 把 $\overline{e^2}_u$ 定义为 user-conditioned residual risk；把 A/B
分离改成一组明确环境假设下的预测，而非识别结果。增加失败案例：高于 50% 的提前结束率、隐藏信息、最后动作影响 reward、不同任务构成和 action-dependent
STOP。验证按用户与边界分组的 held-out Brier risk、calibration 和 resolution。正的全局 EV
既不足以保证这些性质，也不是某个局部边界可用的必要条件；负 EV 不能证明误差“主要”来自 critic，更不能证明权重近似独立随机。§3.4 的证据是 46 步中 28 步负
EV、中位数 $-0.068$（L102），并非全部步骤都负 EV。

### S3. baseline 定理、冻结权重目标与实际 SAO 梯度没有接起来

**位置：** L2252–2258、L2268–2280、L2544–2545、L2567–2568、L2601–2621。

$\mathbb E[\nabla\log\pi(A\mid S)b(S)]=0$ 是 on-policy score-function estimator
的条件恒等式；它要求 baseline 对当前 action 独立，且不参与该期望中的 action 抽样。它不意味着“critic
怎么改都只改方差”，更不意味着“advantage 只会更小更稳”。$V^\pi$ 是 reward MSE 的最佳回归值，通常也不是最小 gradient variance
baseline，后者还取决于 score norm。

当前实现还包含四个不能直接忽略的差别：

1. Actor 用 $\lambda<1$ 的 bootstrap GAE，approximate critic 的误差会影响期望梯度。即使讨论精确 critic，当前
   $\lambda(L)$ 由完整 realized trajectory length 决定，是依赖未来动作/终止的随机系数；标准固定-$\lambda$ GAE
   论证不能原样套用。
1. DIS mask 依赖 action 的 probability ratio。被截掉部分 action
   后，$\sum_a\pi(a\mid s)\mathbf1_{\rm kept}(a)\nabla\log\pi(a\mid s)$ 一般不为零，baseline
   cancellation 不再自动成立。
1. 单 token ratio 校正当前状态下 action 概率，没有完整校正 behavior policy 的 state visitation 及
   future-return distribution。它不是任意 staleness 下的精确 trajectory importance sampling。
1. token-count denominator 是 action-dependent、长度相关的随机量；“两种方法都有”不等于它在两种权重方案下抵消。

历史权重本身则应公平评价：**用过去 episode 的 $e^2$ 并不自动违反当前 action independence。** 给定先前历史
$\mathcal F_b$，只要当前 episode 尚未进入权重表，$w_b(u)$ 可以是冻结常数，虽然它依赖过去的 policy 和 critic。这时得到的是条件于
$\mathcal F_b$ 的局部目标

$$
J_b^{\rm frozen}(\theta)=\sum_u p_b(u\mid\mathcal F_b)w_b(u)J_u(\theta).
$$

跨 batch 更新表后，目标随时间变化。若把 $w_u(\theta)$ 当作一个完整可微目标中的函数，则
$\nabla\sum_up_uw_u(\theta)J_u(\theta)$ 还含 $\sum_up_uJ_u\nabla w_u$；算法没有计算这一项。不能同时使用“冻结
surrogate”与“一个固定的 $J_w$”两种解释。

**具体修改：** 将第 1 步限定为 ideal on-policy Monte Carlo PG；单列固定权重命题。第 6 步先给出代码对应的 update
estimator，再把 DIS、GAE、staleness 和 normalization
带来的偏差列为独立误差项。不得把理想目标恒等式当作当前代码的恒等式。ID/history 放进 baseline 是合法候选，但需说明 actor 和 critic 的
information sets，且拟合 baseline 使用当前 batch 时不能直接声称严格 action independence。

### S4. reward variance 被当成 vector gradient variance，SNR 还漏掉 user-mixture variance

**位置：** L2370–2384、L2588–2597、L2629–2655。

inverse-variance 结论可以适用于 vector estimator，但必须先选择明确的 scalar risk。例如令 episode estimator

$$
G_u=c_ud+\epsilon_u,\quad \mathbb E[\epsilon_u\mid u]=0,
\quad v_u=\operatorname{tr}\operatorname{Cov}(G_u\mid u).
$$

固定各组样本数、独立 episode 时，$\sum_up_uw_u^2v_u$ 确实决定 trace variance；不必额外要求 covariance
isotropic。若优化某个投影，应用 $v_u(h)=h^\top\Sigma_uh$；不同投影的最优 scalar weight 可以不同，也不存在一般的单一
scalar weight 使 covariance matrix 在所有方向最优。README 把 $v_u$ 换成 reward 的 $\sigma_R^2$，需要
score vector 二阶矩、长度、DIS retention 等近似一致的额外条件。

更具体地，第 6b 的终末项可以写成 $e_iB_i$，其中

$$
B_i=\sum_j\lambda_i^{L_i-j}m_{ij}r_{ij}z_{ij},
\qquad z_{ij}=\nabla_\theta\log\pi_\theta(A_{ij}\mid H_{ij}).
$$

其 covariance 是

$$
\operatorname{Cov}(eB)=\mathbb E[e^2BB^\top]
-\mathbb E[eB]\mathbb E[eB]^\top,
$$

而非仅由 $\mathbb E[e^2]$ 决定。$e$、$B$ 和 trajectory length 可以相关；末残差还可能携带真实 action signal。前面的
TD 项经过用户/tool observation 后的 value innovation，同样含用户影响，不能标为“agent 自己这段”。两部分之间也有
covariance。L2653 所说“agent 采样噪声两种方法一样大”错误：$w$ 乘了整个 episode gradient，自然也缩放这部分。

若每条 episode 的用户随机抽样而不是固定配额，还需 total covariance 中的 between-user 项：

$$
\operatorname{tr}\operatorname{Cov}(w_UG_U)
=\mathbb E[w_U^2v_U]+\operatorname{Var}(w_Uc_U)\|d\|^2.
$$

L2633 的分母只计算第一项。因此 SNR 表是**固定用户组成下的理想组内噪声指标** ，不是一般的“每条样本 SNR”。其 numerator 也应包括
$\|d\|^2$；表格相当于省略了共同因子。$q=1$ 的 illustrative reward variance 与非零 $\nabla q$
不能被解释为同一真实参数点：若光滑概率 $q$ 在内部参数点达到 1，则 $\nabla q=0$。

**具体修改：** 分别命名 reward variance、residual risk 和 gradient trace variance。给出固定配额的 vector
derivation，再明确算法使用 residual risk 作为 proxy。SNR 表保留为 toy calculation；删除“真实收益上界”“不估 $c_u$
只损失很少”的一般化表述。附加噪声项、相关性和权重误差没有形成严格的收益排序，不能据此证明上界。

### S5. Kantorovich 算对了，但“完全估错仍保留 45%”的保证是错的

**位置：** L2678–2686、L2696–2699。

在文中 ideal SNR 公式下，一般比例为

$$
\frac{\mathcal S(w)}{\mathcal S(1)}
=\frac{(\mathbb E[wc])^2\mathbb E[v]}
{(\mathbb E[c])^2\mathbb E[w^2v]}.
$$

要把它化成 $(\mathbb Ew)^2/\mathbb Ew^2$，需 $c,v$ 为常数，或至少分别满足 signal 和 variance 的
factorization。只说 $w$“与用户噪声无关”不够，因为它仍可能和 $c$ 有关；critic 的 user-specific error
恰可能造成这种有害相关性。

按文中的理想 SNR 定义构造参数，设两组各占一半：

$$
c=(0.9,0.001),\quad v=(0.09,0.25),\quad w=(0.3,2).
$$

代入其 SNR 公式，加权与 SAO 的比例约为 $0.03074$，远低于 $0.45369$。这已经是 common-direction、非负 signal
的反例，不需要梯度方向冲突。若 $c_u\ge0$、$v_u>0$，仅凭 $m\le w\le M$，同一理想公式下可以得到粗下界 $(m/M)^2$；$[0.3,2]$ 对应
$0.0225$。若用户梯度方向不同且有抵消，这个相对实际梯度 SNR 的保证也不成立。

在确实满足 factorization 的模型里，Kantorovich 下界正确；等号还要求 $w$ 只取 $m,M$，且

$$
\Pr(w=m)=\frac{M}{m+M},\qquad
\Pr(w=M)=\frac{m}{m+M}.
$$

“各用户 $c_u,\sigma_u$ 相同”只保证前面的简化，不保证不等式取等号（L2685）。

**具体修改：** 删除“完全估错”“取值范围就是最坏损失上限”的一般保证。将该结果明确标为 independent-weight/homogeneous-signal
model 下的 lemma。保留 ESS、bounds 作为工程监控，不把它们当作学习效率或稳定性的证书。

### S6. common-direction model 的成立条件缺失，且它没有解决原先的 systematic-bias 问题

**位置：** L2315–2356、L2613–2625、L2714–2722；对照 L1831、L1848–1849、L1949–1953。

$J_u=\eta_u+c_uq$ 是有用的假设，但不是从 τ² 的 stop statistics 推出的结果。它要求所有用户/任务共享同一个 $q(\theta)$，且
$\rho_u,\eta_u$ 与 policy/action 无关。参数定义还允许 $\rho_u+\eta_u\ge1$，此时 $c_u\le0$，L2619
的“都是正数乘 $\nabla q$”不成立。应要求总体和加权后的 $c$ 平均均为正；$c_u>0$ 是一个简单充分条件。

L2345 的 $J_B=qr$ 需要说明 $q$ 是“已追问后其余操作成功的条件概率”，或额外假设追问与其余成功独立、且该条件能力与普通用户相同。按 L2315
的“全部做对概率”定义，追问本来也是该做的事，再乘 $r$ 有重复计数风险。一般应写

$$
J_B=r(\theta)q_B(\theta\mid\text{已追问}),
$$

它的 gradient 未必可分成与普通用户完全相同的 $\nabla q$ 加一个独立“追问方向”。B 也可能写成 affine model
的特例；“不能写成这个形式”应改成“一般不满足共享 $q$ 的模型”。

更重要的是，§16.3 的问题 B 讨论用户系统性偏离让 agent 学会绕过确认，§16.0 要求区分能否挽回；§16.11 却假设用户只衰减同一个正确能力方向，并给整条
episode 降权。该算法并没有从 episode 内扣除某轮的用户责任，也没有证明能恢复 clean-user objective。§16.11 的“用户 B：难应付”与
§16.3 的“问题 B：目标被带偏”还用了相同字母指不同问题。

**具体修改：** 明确区分 observed-user return、clean-user return 和部署用户分布。把方法贡献限定为“在指定结构模型下改变
sampling efficiency，并在真实交互中检验 residual-risk weighting”；若目标仍是 causal credit assignment
或纠正 systematic bias，必须提供相应识别条件和新证据。保留“任意正权重不改方向/最优解”的条件定理，删除模型外“多学追问不是坏事”“给 B
权重不低于普通用户”的保证。

### S7. 训练流程省略 critic-first fitting，residual 的 snapshot 没有定义

**位置：** L2512–2524、L2547–2549。

`config_sao.yaml:55` 设置 `critic_updates_before_actor: 2`。`update.py:107–125` 先在固定的当前
batch targets 上更新 critic 两次，再刷新 values、重算 actor GAE，然后做 actor update。若第 5 步读取 actor
实际使用的 refreshed value，记录的是 **in-sample post-fit residual** ，并非尚未看过该 reward 的 prediction
error；一个容易记忆当前 batch 的 critic 会让记录变小。

**具体修改：** 规定表中 residual 使用 pre-update frozen critic，并在同 batch critic fitting 之前保存；actor
仍可用其原有 refreshed critic。对历史记录保存 policy/critic version、测量边界和时间。若用 post-fit
residual，必须论证其估计偏差并做 held-out 对照。过去 batch 维护表避免当前 rollout 直接自降权，但不消除历史 in-sample
fitting、policy drift 或 task-selection confounding。

### S8. normalization、Adam 与 clipping 不提供“步长不变/至少一样稳”的保证

**位置：** L2537–2545、L2629、L2648–2649、L2663–2667、L2694–2699。

batch 内 episode 平均 $\bar w=1$ 不推出 token 平均为 1：$\sum_iL_iw_i/\sum_iL_i$ 会随 length-weight
correlation 改变，也不控制实际 gradient norm。batch normalization 让归一后的 $w_i$ 依赖其他用户的组成；若组成由
rollout completion/过滤决定，还可能依赖 outcome/length。严格“对话开始前就定了”的对象只能是 raw historical weight，除非
batch 用户名单及 normalization constant 事先确定。

Adam 对整个 gradient history 乘一个固定全局常数，在忽略 $\epsilon$ 等项时有尺度抵消；这里的 user weighting 改变每个
coordinate 的 signal、variance 和 moment history，远不是全局常数。改变 $w$ 的时点也会改变 momentum。gradient
clipping 限制输入 optimizer 的 gradient norm，不直接保证 parameter-update norm、KL 或 learning
stability。critic loss 不加权有意义，但 actor 改变会改变后续 critic 的数据分布，因此 L2663 不能排除所有反馈。

此外，第 5 步只写加 $\tau$ 和归一化，第 6g 又要求最终 $w\in[0.3,2]$，没有指定 clipping 与 normalization 的顺序。先
clip 后除均值可能破坏 bounds；先归一后 clip 可能破坏 mean-one。若最终范围是 $[0.3,1]$ 且平均严格为 1，只能所有权重均为 1，L2686
的例子也不能同时承担两个要求。

**具体修改：** 定义一种明确的最终权重映射，例如 $w_i=\operatorname{clip}(a h_i,m,M)$ 并求 $a$ 使 batch mean 为
1；它可满足两者，但仍不是 episode-start frozen objective。理论用固定 population normalization，实践另列 batch
normalization。将“基本不变/噪声变小”改为待测量结论，报告 token-weighted scale、clip frequency、update
norm、KL、moment drift 和按用户的收益。

## 3. 次要问题 / 推理跳步

### 3.1 数值核查：算术多数正确，标签和适用范围需要改

以下结果用独立 Python 标量计算复核；不是重新运行训练或原始评测。

| 主张与位置                  | 复算结果                                                                 | 判断                                                                    |
| --------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| 温度计加权，L2363–2365      | 等权 $1.25$；$0.8/0.2$ 加权 $0.8$                                        | 正确，但先声明两测量误差独立，否则含 covariance                         |
| $q:0.8\to0.9$，L2333–2339   | 普通 $0.72\to0.81$；A $0.40\to0.45$                                      | 正确，依赖固定 flip rates                                               |
| SNR 表 80/20，L2643         | $5.511475,7.282569,7.400000$；相对增益 32.13%                            | 正确，表头应写 ideal $\mathrm{SNR}^2$，非 SNR                           |
| SNR 表 50/50，L2644         | $2.882353,4.764706,5.000000$；相对增益 65.31%                            | 同上；仅含固定组成的组内噪声                                            |
| weights/ESS，L2672–2673     | $1.146789,0.412844$；$\mathrm{ESS}/N=0.920651$                           | 正确，只描述这组权重的集中程度                                          |
| Kantorovich，L2685–2686     | $[0.3,1]:0.710059$；$[0.3,3]:0.330579$                                   | 算术正确，保证条件不完整                                                |
| Kantorovich，L2697–2698     | $[0.3,2]:0.453686$                                                       | 算术正确，“完全估错”的解释错误                                          |
| $\lambda^L$，L2591–2594     | 极限 $0.513417$；$L=1,2,3,10$ 时为 $0.333333,0.444444,0.470508,0.501612$ | 有限 $L$ 从下方趋近，不能据此说所有长度至少一半                         |
| 实际最早 residual 系数      | $\lambda^{L-1}$；$L=1,2,3,10$ 时为 $1,0.666667,0.604938,0.537441$        | 这是实现应使用的指数；其下界为 $e^{-2/3}$                               |
| STOP rates，L1929/L2299     | 90/553 = 16.2749%，20/627 = 3.1898%；rate ratio $5.1022$                 | 16.3%/3.2% 正确；L1949 的 4.5 倍是 count ratio $90/20$，不是 rate ratio |
| Retail drop，L1809/L228–229 | §16.0 写 $0.651$，§9.1 写 $0.649$                                        | 差约 20pp 一致；两种终点应说明不同来源或统一                            |

$\lambda^{L-1}\ge e^{-2/3}$ 可以严格证明：令 $a=2/3$，使用 $\log(1-x)\ge-x/(1-x)$，有

$$
(L-1)\log(1-a/L)\ge-a\frac{L-1}{L-a}\ge-a.
$$

$L=1$ 时系数直接为 1。**这仅是终末 TD 项的 amplitude coefficient 下界** ；不保证整个 advantage
或用户噪声方差下界。固定其他因素时，amplitude $0.5134$ 对应 variance factor 约 $0.2636$，不能把“幅度减半”和“方差减半”混用。

原始 STOP 记录、EV logs、DB reward 和已有 related-work 数字没有在本审阅中重新生成；本次核实的是 README
所述证据、源码及公式算术，不是独立复现所有观测。

### 3.2 shrinkage 是合理 regularization，但尚未被推导成 empirical Bayes estimator

**位置：** L2526–2535、L2549、L2675–2676。

$(n_u\overline{e^2}_u+k\overline{e^2}_{\rm all})/(n_u+k)$ 是明确的 pooled shrinkage
rule。要称其为 empirical Bayes posterior estimator，需给出 residual
likelihood、prior、hyperparameter estimation 和 posterior derivation；$e^2$ 不是 Bernoulli
count，不能只用 pseudo-count 类比完成推导。全局均值含 task mixture 和 user frequency，频繁用户可能主导
prior。窗口较短会加大噪声，并不天然让权重变化更慢；长窗口又更受 policy/critic drift 影响。应先称 regularized running risk
estimate，说明窗口、$k,\tau$、冷启动与 effective sample size，再视是否确有层级模型使用 empirical Bayes 术语。

### 3.3 $e$ 不是一般的 $\Delta_u$ 特例

**位置：** L2497；对照 L1985–1995。

$\Delta_u$ 是 user turn 前后 value difference；$e=R-V$ 是 realized terminal return minus
pre-action baseline。仅当边界正确、接下来只有用户 transition、终态 value 等于 $R$ 且无 bootstrap 时，才能把二者对齐。当前末
token 选点不满足这一通用条件。应写“在特定 terminal-transition model 下有形式联系”，并给出时间轴。

### 3.4 actor weighting 不等于 trajectory credit assignment；“只能改两处”只是设计限制

**位置：** L2250–2264、L2282、L2443–2449；对照 L1859、L1868。

第 1 步应写“本文限定讨论两类干预”。即使 $R$ 固定，sampling distribution、control variates、advantage
estimator、trajectory decomposition 和 optimizer 仍能改变；state-dependent baseline 也不必一定是
learned critic。对整条 rollout 乘同一 $w_u$ 会连同出错前的有用动作一起减弱，没有解决 §16.2 所说“只扣用户那一部分”。改变有限容量
critic 的 loss weights 或补充 user-position training 也可能改变共享参数和 agent-position
predictions，L2445–2446 的“不改变/基本无效”只能在理想 unrestricted population optimum 下成立。

### 3.5 单次 split-half cosine 不能证明 variance reduction

**位置：** L2701–2710、L2727–2731。

一批数据随机拆半后 cosine 上升，可能来自 estimator mean、组别组成或梯度主方向改变；低 cosine 也可能是 near-zero mean
或优化噪声，不等价于 B 被误判。应使用独立于权重估计的数据，固定 policy/critic，控制用户与任务配额，多次 resplit 或多个独立 batch，报告
cosine uncertainty、mean gradient、trace variance 及 estimator norm。若现有 rollout
被反复用来挑权重，结果还会有 selection bias。

“最差组比 baseline 低就说明第一假设被违反”（L2727）同样过强；它可能来自 optimization dynamics、critic error
或评测误差。persona 中设定 0/20/50% 也不证明实际实现了这些外生 STOP rates；自然语言 persona
指令应与已实现的、action-independent 随机终止机制区分。受控 simulator intervention 可以作为独立诊断，但不应改变 16.9
的方法信息预算，更不能作为 API 用户无 seed/no replay 的直接证据。

### 3.6 online baseline 和 P-GRPO 描述存在内部不一致

**位置：** L2164–2167、L2209、L2234–2238、L2271、L2443、L2737–2738。

不能构造标准 same-prompt GRPO group 是对的；“只能用 critic”“critic 必需”却与 L2234 的无 critic REINFORCE++
baseline 冲突。REINFORCE 可以用 zero、constant 或 past-data running baseline；critic 是合理选择，而非
one-shot interaction 的数学必要条件。L2234 的“下界”也没有性能排序定理，宜称 no-critic baseline。

按 README L2209 的公式，P-GRPO 含 $(R-\mu_u)/\sigma_u$：既有平移也有 user-dependent scaling。第 16.11
把它整体归为“只改 $V$”不符合正文自己的描述。应区分“只减 running mean 的消融”与“同时除 running standard deviation
的变体”。本次没有核查 P-GRPO 原论文全文，不据此判断原论文的具体实现。

## 4. 假设核查表

| 假设                                                 | 用在哪一步        | 在 τ² 在线设定下是否成立                                          | 是否容易满足/检验                                 | 依据（README 行号或代码）                  | 建议                                                             |
| ---------------------------------------------------- | ----------------- | ----------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------- |
| 目标 user/task distribution 已固定且可观测           | 1、2、6c          | 未定义清楚；recurring user 跨任务，任务构成可能不同               | 定义容易；消除 task confounding 较难              | L2153–2160、L2608–2609                     | 明确 $P(U,X)$ 与 evaluation distribution，按任务/领域控制比较    |
| 所有用户共享同一 $q(\theta)$                         | 2、6d             | 很可能不成立；4B agent 的追问、工具、policy compliance 是不同能力 | 一般不能证明；可做分组 gradient alignment 诊断    | L2315、L2341–2349、L2716–2719              | 仅作 structural assumption，不能作为经验事实                     |
| flip rates 与 action/policy 无关                     | 2、6d、7          | 不宜默认；API LLM user 对措辞、上下文、操作数作出反应             | 在 one-shot 数据中难做因果检验；可独立受控实验    | L1920–1932、L2720–2722                     | 量化可能的 action dependence，并把相应 bias 保留在分析中         |
| $\rho_u+\eta_u<1$，或两目标总体 signal 均为正        | 2、6d             | 没有证据保证；系统性失败/误反馈可能接近或超过边界                 | 无 privileged correctness 时难估                  | L2316–2319、L2619                          | 显式列出；不把任意参数范围都称“方向不变”                         |
| $J_B=qr$ 的条件分解与共享能力                        | 2                 | 是 toy construction，未由实际数据支持                             | 可定义 conditional $q_B$；共享性难检验            | L2341–2349                                 | 改为 $r q_B$；B 案例应标 hypothetical                            |
| 末 agent 位置是完成任务后的固定边界                  | 4、6b             | 与提前 STOP 分支及 pre-action value 对齐冲突                      | 可检查 transcript 和 action indexes，容易发现反例 | L1897–1898、L2478–2479；`actor.py:226–227` | 重定义边界；否则降级为 residual proxy                            |
| 边界后只剩外生 user randomness                       | 4、6b             | 不保证；成功分支仍可能执行工具和回复                              | 可检查边界后的事件；因果外生性难验证              | L245–246；`gae.py:43–50`                   | 分开 action、tool、user innovations                              |
| $V_\phi(H,u)=\mathbb E[R\mid H,u]$                   | 3、4、6b          | 当前负 EV 是不利信号，且全局指标不能保证局部/分用户预测           | held-out calibration/Brier 分桶可测；稀有用户较难 | L102–103、L2547–2549                       | 不用“正 EV”替代条件准确性；给定测量位置评估                      |
| 可观测 prefix 包含判定 agent 对错所需信息            | 4 的 A/B 分离     | 不保证；actor/critic 未见隐藏目标与完整 DB state                  | 可对比可观测 transcript 与 evaluator 信息         | L1818、L1878、L2158                        | 不把 full-history input 等同 fully observed state                |
| score 二阶矩、长度、DIS retention 各用户近似一致     | 2、6e             | 很可能不成立；困难用户对话长度与 agent behavior 不同              | 可测分用户 norm/length/retention，需样本          | L2384、L2654；`dis.py:62–80`               | 直接测 episode gradient variance，标明 residual proxy 偏差       |
| 固定用户配额、episode 独立                           | 6e                | 当前在线 batch 不保证；同用户、任务、policy version 可相关        | 配额可设计；相关性需诊断                          | L2517、L2633、L2704                        | 写明 conditioned composition，或加入 between-user/correlation 项 |
| 权重在当前 episode 行动前冻结                        | 1、5、6c          | raw past-data weight 可以成立；batch-normalized weight 不自动成立 | snapshot、timestamps 可直接审计                   | L2524、L2541、L2545                        | 明确 filtration、batch composition 和 normalization 时点         |
| 每个 ID 有足够多跨任务 episode                       | 5                 | recurring IDs 允许积累，但没有 counts 支撑；窗口后更稀疏          | 可统计 $n_u$、effective counts、coverage          | L2159、L2532–2535、L2549                   | 给冷启动规则和 uncertainty；区分 LLM backend ID 与真实个体 ID    |
| 用户机制与 critic/policy 在估计窗口中近似 stationary | 5、6f             | API 服务、persona/history 和 policy 持续变化，可能失败            | drift 可监测；只能近似                            | L1939、L2549、L2675–2676                   | 版本化记录，窗口/衰减 sensitivity，避免宣称恒定 $\sigma_u^2$     |
| residual 使用未拟合本 episode 的 critic              | 4、5              | 默认 refreshed actor value 不满足                                 | 可选 pre-update snapshot，容易实现                | `config_sao.yaml:55`；`update.py:107–118`  | 固定 pre-update prediction；另报 post-fit bias                   |
| scalar weight 与 signal、variance 无关               | 6f/6g Kantorovich | 方法专门按用户 risk 加权，通常不成立；错 critic 更不保证独立      | 无法由 bounds/EV 推出                             | L2678–2686、L2697–2698                     | 删除一般 45% 保证，保留条件 lemma                                |
| Adam 重加权等效全局 scale                            | 6e、6f            | 不成立；coordinate moments 和 clipping 参与更新                   | 可测 actual updates，但不能用均值1证明            | L2629、L2666；`fsdp_engine.py:901–920`     | 实测 update norm/KL/clip frequency                               |
| 在线可获得 binary terminal reward                    | 1、3、4           | benchmark 内成立；真实用户部署另需 outcome signal                 | benchmark 易检验；现实需业务定义                  | L2191–2192；`agent.py:752–756`             | 明确这是 benchmark proxy，不等同自然在线满意度                   |

## 5. 代码与文中描述是否一致

下表核对的是当前源码和 YAML inheritance；未运行完整 Hydra resolution 或训练，因此不证明既有 run 实际消费了所有配置。文件均在仓库内。

| §16.11 的代码主张                                                  | 代码证据（file:line）                                                                   | 核查结论                                                                                                                                |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| actor 使用 Direct DIS，非 PPO clip（L2558–2559、L2690）            | `examples/tau2/config_sao.yaml:40–42`；`areal/trainer/ppo/actor.py:535–541`             | 确认；继承的 PPO eps fields 不决定此分支 loss                                                                                           |
| ratio 相对 rollout policy（L2567–2568）                            | `config_sao.yaml:31–32`；`actor.py:256–259`；`dis.py:39–56`                             | 确认使用存储的 behavior logprob；不等于完整 state-distribution IS                                                                       |
| DIS 保留 $0.7<r<6$（L2569–2570）                                   | `areal/trainer/ppo/dis.py:62–66`                                                        | 确认严格开区间，等于边界也拒绝                                                                                                          |
| DIS gradient 为 $m r A\nabla\log\pi$（L2563）                      | `dis.py:71–72`                                                                          | 确认为 loss 的负梯度/上升方向；ratio、advantage 与 mask 不反传                                                                          |
| denominator 是全部 agent token 数（L2566）                         | `dis.py:80`；`actor.py:559–561`                                                         | 确认不是 retained-token count；乘 $w$ 不改 denominator                                                                                  |
| $w$ 不直接改变当前 DIS mask（L2665）                               | `dis.py:55–71`                                                                          | 固定 logits/batch 时正确；改变 actor 后未来 ratios/masks 仍会变                                                                         |
| batch 只做一次 optimizer step（L2560）                             | `config_grpo.yaml:163`；`actor.py:525–535`；`update.py:125`                             | 正常情况为一次 **actor** step；另有两次 critic step，全部 DIS rejection 可跳过 actor（`update.py:64–75`）                               |
| $\lambda=1-1/(1.5L)$（L2571）                                      | `config_sao.yaml:36–38`；`lambda_fn.py:38–60`                                           | 确认，$L$ 是 active action tokens，含生成 EOS，排除 user/tool/padding                                                                   |
| GAE timestep 是 token 而非 turn                                    | `areal/api/cli_args.py:1808–1813`；`actor.py:329–350`                                   | default 为 token；该 SAO YAML 未覆盖为 turn                                                                                             |
| 中间 user/tool token 是否增加 $\gamma/\lambda$ 衰减                | `gae.py:46–50`                                                                          | 不增加；mask=0 保持递推状态。$K$ 必须数 active action transitions，不是原始序列距离或 turn 数                                           |
| 终末 $\delta=R-V$（L2583–2584）                                    | `actor.py:278–288,303–322`；`examples/tau2/agent.py:752–756`                            | τ² producer 显式 zero bootstrap，且 KL=0/无额外 token reward 时确认；一般路径是 $r+\gamma b-V$                                          |
| 前面 token 收到 $\lambda^K e$（L2585–2588）                        | `gae.py:43–50`                                                                          | action-subsequence algebra 确认；最早 $K=L-1$，不能据此把 $e$ 解释为纯用户噪声                                                          |
| critic 对整条输入 forward，仅 active 位置 loss（L2399–2400）       | `critic.py:37–42,168–173`；`actor.py:226–227`                                           | 确认 prefix-state supervision；“最后 token 已看完 action”不确认                                                                         |
| critic $\lambda=1,\gamma=1$，returns 是 terminal $R$（L2402–2405） | `config_sao.yaml:39`；`config_grpo.yaml:162,180`；`actor.py:354–368`；`gae.py:53–54`    | 在 τ² zero bootstrap、KL=0、无额外 token reward 下确认望远镜消去；returns 使用独立 trace decay 重新生成，不由 actor advantages 变换得到 |
| critic 无 value clipping MSE（L2407–2414）                         | `config_sao.yaml:74`；`critic.py:170–174`                                               | 确认无 $1/2$ 系数的 active-token MSE；finite model 不保证得到 $V^*$                                                                     |
| $V(s_{\rm last})$ 是现成末 active value（L2512–2513）              | `actor.py:226–227,302`；`critic.py:37–42`                                               | 可以读取；是 pre-action value，snapshot 必须定义                                                                                        |
| 现有循环按 L2517–2522 的顺序                                       | `config_sao.yaml:55`；`areal/trainer/ppo/update.py:107–125`                             | 不准确；实际先 critic 两次 update，再刷新 values/actor GAE，再 actor update                                                             |
| $n_{\rm samples}=1$、advantage 不归一化                            | `config_sao.yaml:25–28,34–35`；`actor.py:370–377`                                       | 确认；训练样本设置不自动证明 16.9 的 unique $(user,task)$ 协议                                                                          |
| Adam、gradient clipping 1.0（L2629、L2666）                        | `config_grpo.yaml:147–154`；`config_sao.yaml:43–45`；`fsdp_engine.py:901–920,1527–1533` | 配置名为 adam，FSDP 用 AdamW；SAO weight decay=0，此时对应 Adam。规则保留不证明 update 等效                                             |
| 不改 actor-only weighting 之外的 critic targets（L2451、L2553）    | `actor.py:352–368`；`update.py:121–122`                                                 | 可以保持 targets 不变；新增权重表及 $wA$ 尚是方案，无现有实现可确认                                                                     |

## 6. 符号规范建议

### 6.1 建议统一的 notation

| 符号                                      | 定义 / 随机性                                                                        | 粒度与限制                                                      |
| ----------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| $b,\mathcal F_b$                          | update index；当前 batch 之前可用的 records、policy/critic versions                  | 区分条件论证与跨训练历史                                        |
| $U,X\sim P_b$                             | user ID 与 task；$p_b(u)$ 为 user marginal                                           | 必须指定 $P_b(X\mid U)$，不能默认 task distribution 相同        |
| $\mathcal E\sim P_\theta(\cdot\mid U,X)$  | episode，包括 agent actions、user/tool observations、termination                     | 不用 $\tau$ 同时指轨迹和 regularizer                            |
| $L_i,H_{ij},A_{ij}$                       | episode $i$ 的 active action count、第 $j$ 个 action 前的 observable history、action | $j=1,\dots,L_i$；跳过 non-agent tokens；$H$ 不代表隐藏 DB/goal  |
| $R_i\in\{0,1\}$                           | binary terminal outcome                                                              | 理论与实际 reward shaping 分开                                  |
| $\theta_b,\mu_i$                          | current actor parameters；episode behavior policy                                    | 异步 rollout 可有不同 behavior versions                         |
| $V_{\phi^-_b},V_{\phi^+_b}$               | 本 batch critic fitting 前、后的 predictor                                           | risk table 与 actor GAE 使用哪个 snapshot 明写                  |
| $z_{ij}(\theta)$                          | $\nabla_\theta\log\pi_\theta(A_{ij}\mid H_{ij})$                                     | vector-valued score；明确 detach 的对象                         |
| $r_{ij},m_{ij}$                           | importance ratio；DIS retained mask                                                  | 不用 $\rho$ 同时表示 user failure rate 与 importance ratio      |
| $\delta_{ij},\widehat A_{ij}^{\lambda}$   | TD residual；sample GAE estimate                                                     | estimator 用 hat；不是 population advantage 的无条件替代        |
| $f_u^-,f_u^+$                             | agent “正确”时失败率、“错误”时成功率                                                 | 替代 flip model 中 $\rho_u,\eta_u$；与 action 独立是 Assumption |
| $c_u=1-f_u^--f_u^+$                       | signal coefficient                                                                   | 不与任意回归常数 $v$ 或 $k$ 混用                                |
| $q(\theta),d(\theta)=\nabla q(\theta)$    | 共享 latent capability 和 gradient                                                   | 是结构假设，不是观察得到的 agent correctness label              |
| $J_u(\theta),J_b^{\rm frozen}(\theta)$    | 指定 task distribution 下 conditional expected return；冻结权重局部目标              | 每个 expectation 标清 user/task/action/environment              |
| $e_i,D_u$                                 | 指定边界、snapshot 下 residual；$D_u=\mathbb E[e_i^2\mid U=u,\mathcal F_b]$          | $D_u$ 是 residual risk，不称用户固有 noise variance             |
| $\sigma_{R,u}^2$                          | 特定条件下 reward variance                                                           | 不与 $D_u$ 或 gradient variance 等同                            |
| $G_i,\mu_u^G,\Sigma_u,v_u$                | episode vector estimator、conditional mean、covariance、trace covariance             | $v_u=\operatorname{tr}\Sigma_u$；投影风险另定义                 |
| $\widehat D_{b,u},h_{b,u},w_{b,u}$        | shrunk risk estimate、raw weight、最终 weight                                        | 分清 population/batch normalization 和 clipping                 |
| $\widehat g_b^{\rm DIS},\bar g_b$         | 实际随机 update estimator；其指定条件下 expectation                                  | 不把 $g$、$\nabla J$、单 sample、batch mean 混写                |
| $\epsilon_D^-,\epsilon_D^+,\varepsilon_D$ | DIS thresholds；weight denominator regularizer                                       | regularizer 不用易与 trajectory 混淆的 $\tau$                   |

### 6.2 可直接改写进论文的核心命题

以下是建议的**条件理论版本** 。它们没有证明现有 heuristic 满足全部条件。

**Assumption A（ideal episode PG）。** 当前 update 给定 $\mathcal F_b$，$P_b(U,X)$ 固定；episode
on-policy，无 reward shaping，terminal reward 为 $R$。baseline $B_j(H_j,U,\mathcal F_b)$ 在当前
action 前确定，不使用当前 episode 的未来信息或拟合结果。所用 estimator 是未截断的 Monte Carlo score-function
estimator，满足交换微分与期望及有限二阶矩的常规条件。

**Proposition 1（baseline cancellation 与冻结权重目标）。** 定义

$$
G(\mathcal E;\theta)=\sum_{j=1}^{L}z_j(\theta)[R-B_j],
\quad J_u(\theta)=\mathbb E_{X\mid u,\mathcal E\sim P_\theta}[R].
$$

若 $w_b(u)$ 给定 $\mathcal F_b$ 固定且非负，则

$$
\mathbb E[G\mid U=u,\mathcal F_b]=\nabla J_u,
\quad
\mathbb E[w_b(U)G\mid\mathcal F_b]
=\nabla J_b^{\rm frozen},
\quad
J_b^{\rm frozen}=\sum_up_b(u)w_b(u)J_u.
$$

**Proof sketch：** 条件于每个 pre-action
history，$\sum_a\pi(a\mid H)\nabla\log\pi(a\mid H)=\nabla1=0$；baseline 项消去，再使用 trajectory
likelihood-ratio identity。权重在当前条件期望中为常数，不对历史权重机制求导。结论不覆盖 DIS masking、随机 length-dependent
GAE 或 current-batch fitted baseline。

**Assumption B（shared capability / policy-independent flips）。**
对所有组，$J_u(\theta)=f_u^++c_uq(\theta)$，$f_u^\pm,c_u$ 与 $\theta$ 独立，且
$\sum_up_bc_u>0$、$\sum_up_bw_bc_u>0$；令 $c_u>0$ 是简单充分条件。

**Proposition 2（ideal gradient alignment）。** 在 A、B 下，

$$
\nabla J_b=(\mathbb E_{p_b}c_U)\nabla q,
\qquad
\nabla J_b^{\rm frozen}=(\mathbb E_{p_b}w_bc_U)\nabla q.
$$

因此两个冻结目标的 ascent directions 相同，且各自最大化问题与最大化 $q$ 等价。在 $\nabla q=0$ 处不谈 cosine direction。

**Proof sketch：** 对 affine form 求导；两个系数为正。该结论既不适用于一般 B 用户，也不保证非凸问题中有限步 Adam 的相同训练轨迹或终点。

**Assumption C（vector estimator、fixed composition）。** 固定 $N$ 个独立 episode，组数为
$n_u=Np_u$，且 $c_u>0$；$G_{ui}$ 的 conditional mean 为 $c_ud$，covariance 为
$\Sigma_u$，$v_u=\operatorname{tr}\Sigma_u>0$，$d\ne0$。权重非负且在 batch 前固定。

**Proposition 3（所选 vector risk 下的 optimal weighting）。** 令
$\widehat g_w=N^{-1}\sum_{u,i}w_uG_{ui}$，则

$$
\mathrm{SNR}_{\rm tr}^2(w)
=\frac{\|\mathbb E\widehat g_w\|^2}
{\operatorname{tr}\operatorname{Cov}(\widehat g_w)}
=N\|d\|^2\frac{(\sum_up_uw_uc_u)^2}
{\sum_up_uw_u^2v_u}.
$$

忽略共同 factor $N\|d\|^2$ 后，其最大值为 $\sum_up_uc_u^2/v_u$，且最优 $w_u\propto c_u/v_u$（$c_u>0$ 时）。

**Proof sketch：** 对 $\sqrt{p_u}w_u\sqrt{v_u}$ 与 $\sqrt{p_u}c_u/\sqrt{v_u}$ 使用
Cauchy–Schwarz；等号要求二者成比例。若纯 inverse-variance weights 用 population mean 归一为 1，则 batch
mean 的组内 trace variance 为 $1/(N\mathbb E[1/v_U])$，不大于 SAO 的 $\mathbb E[v_U]/N$。这不证明
$1/D_u$ 有同样效果，也不涵盖 random user composition 的 between-user variance。

**Proposition 4（residual-risk decomposition，不是 identification theorem）。** 在固定可测边界 $S$、给定
$U=u$、使用冻结 predictor 时，

$$
D_u=\mathbb E[(R-V_\phi(S,u))^2\mid U=u]
=\mathbb E[p(S,u)(1-p(S,u))\mid u]
+\mathbb E[(p(S,u)-V_\phi(S,u))^2\mid u].
$$

**Proof sketch：** 将 $R-V=(R-p)+(p-V)$ 展开，利用条件均值为零消去交叉项。$D_u$ 只能在另加“边界后不含可影响 outcome 的
agent action、task/hidden-state uncertainty 已解释、user mechanism 外生、$V_\phi=p$（或
approximation error 可忽略）”等假设后解释为相应用户扰动的平均 variance。对事后末位置需重新定义其 selection-conditioned
$p$。将 $D_u$ 当作 $v_u$ 的 proxy 仍需另一条经验或结构假设。

**Proposition 5（实际 SAO 的 algebraic terminal-term decomposition）。** 对当前代码的 active action
subsequence，假定 zero bootstrap、无 KL/token extra reward、$\gamma=1$，令 $K_j=L-j$，则

$$
\delta_L=R-V_L,
\quad
\widehat A_j^\lambda
=\sum_{k=0}^{K_j-1}\lambda^k\delta_{j+k}
+\lambda^{K_j}\delta_L,
\quad
\lambda=1-\frac{2}{3L}.
$$

最早终末系数 $\lambda^{L-1}\ge e^{-2/3}$。这是 `gae.py` 递推的代数性质，不区分前后两部分的 causal responsibility。

对应 actual update estimator 为

$$
\widehat g_b^{\rm DIS}
=\frac{1}{\sum_iL_i}\sum_iw_{b,i}\sum_{j=1}^{L_i}
\mathbf1\{0.7<r_{ij}<6\}\,r_{ij}\widehat A_{ij}^\lambda z_{ij}.
$$

**Proof sketch：** non-agent mask positions 保持递推，按 active index 展开 GAE；DIS 取 detached
coefficients 的 score gradient。这个 estimator 的 conditional mean 是否接近 Proposition 1 的
gradient，是需要单独分析/测量的 approximation error，不由展开式推出。

**Lemma 6（Kantorovich 的窄条件版本）。** 在 Proposition 3 的 normalized ideal SNR 中，若
$\mathbb E[wc]=\mathbb E[w]\mathbb E[c]$、$\mathbb E[w^2v]=\mathbb E[w^2]\mathbb E[v]$，且
$0<m\le w\le M$，则

$$
\frac{\mathcal S(w)}{\mathcal S(1)}
=\frac{(\mathbb Ew)^2}{\mathbb Ew^2}
\ge\frac{4mM}{(m+M)^2}.
$$

**Proof sketch：** $(w-m)(M-w)\ge0$ 给出 $\mathbb Ew^2\le(m+M)\mu-mM$，其中
$\mu=\mathbb Ew$；最小化 $\mu^2/[(m+M)\mu-mM]$，在 $\mu=2mM/(m+M)$ 取到上述
bound。等号需要端点二点分布及前述概率。它不提供任意错误 risk estimate 的保证。

## 7. 书写与表述问题

### 7.1 每步 intent 的清晰度

| 步骤 | 已表达的目的                                 | 仍需补的目标或边界                                                                                         |
| ---- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| 1    | 区分 baseline 与 weighting，清楚             | 先声明只讨论两种设计；区分 ideal PG 和实际 DIS surrogate                                                   |
| 2    | 解释何时降权不改方向，清楚                   | “无害”要限定为 ideal expected direction/argmax；“学得最快”须先指定 variance risk 和 sample allocation      |
| 3    | 说明 critic loss 和 conditional mean，较清楚 | 明确 population MSE optimum、behavior distribution、finite model 与 snapshot；不能把回归最优当作当前已学到 |
| 4    | 想分离 A/B，清楚                             | 先定义合法 measurement boundary 与待估量；现在从想要的解释直接跳到 last token，关键 lemma 缺失             |
| 5    | 实现在线表与 shrinkage，清楚                 | 明确 critic-first 顺序、pre/post-fit residual、user ID semantics、window、clipping/normalization           |
| 6    | 比较目标、方向、效率、稳定性，覆盖全面       | 每小节都需先标 ideal theorem / proxy assumption / empirical test，避免同一个 $\sigma^2$ 跨层复用           |
| 7    | 列出 failure modes，有价值                   | 实际还有 selection、critic fitting、task mixture、nonmonotone variance 等失败；检查结果不能作单一因果诊断  |

### 7.2 专业性与 claim strength

- L2245 的“不追求严格无偏”“留给工程手段”不能代替误差分析。可以选择有偏算法，但必须给 bias 的来源、目标 trade-off 与检查方式。
- L2280 的“这个用户上的表现不重要”、L2339 的“推多大力”、L2367 的“带坏”、L2529 的“冤枉降权”、L2545/L2740
  的“把失败推给用户”、L2718 的“坏事”，适合内部讨论。论文改成 lower contribution to the surrogate、signal
  magnitude、variance inflation、misweighting、outcome-dependent selection bias。
- 温度计（L2360–2368）、天气（L2459–2474）和餐馆评分（L2529）最多保留一个短解释，其余移到
  footnote/appendix。主文用定义、条件和命题，尤其不能让 thermometer analogy 隐藏独立性与共同 estimand。
- “无害”“最优”“正好”“方向完全一样”“不会变错”“噪声变小”“至少一样稳”“上界”都应绑定模型和 risk；第 7 步末尾补 caveat，不能修复第 4/6
  步已经作为事实使用的结论。
- A 的“行为两次完全相同”（L2435）只能指某个共同 prefix/action，成功轨迹后续行为并不完全相同。请避免把相同 prefix 与相同完整 rollout
  混写。
- L2298 把 A/B 都称“实际出现”：A 有具体 STOP 证据，B 没有单独计数（L2300）。B 的 50% success 和末状态 0/1
  是示意构造，应在表名标出。
- $k,\tau,w_{\min}=0.3,w_{\max}=2$, ESS threshold $0.7N$ 都没有统计或任务依据；可以作为 initial
  hyperparameters，不能从现有 bound 宣称它们具有 optimality 或 safety。
- §16.10 L2225–2226 指向“16.11 第 2 步”的 residual-mean 论证，实际详述在第 4c
  步（L2499–2508）。应修正交叉引用；本审阅不直接改稿。

### 7.3 novelty / positioning 风险

inverse-variance weighting、weighted least squares 和共同均值下的最小方差线性估计是经典结果；第 2
步不应作为理论新贡献，L2742 对此承认得正确。scalar vector-risk weighting 的 Cauchy–Schwarz proof
本身也很短，不能仅因搬到 per-user RL 就声称新的最优性理论。

已知且可可靠命名的相关结果包括：

- **Gauss–Markov / generalized least squares：** 共同 estimand、异方差及独立/已知 covariance
  下的线性估计效率；和本文 inverse-variance 推导直接相关。
- **Greensmith, Bartlett & Baxter (2004), “Variance Reduction Techniques for Gradient
  Estimates in Reinforcement Learning”：** baseline/control variate 的 variance 分析，以及普通
  value baseline 与 optimal gradient-variance baseline 的区别。可查
  [JMLR 原文](https://www.jmlr.org/papers/v5/greensmith04a.html)。
- **Schulman et al., “High-Dimensional Continuous Control Using Generalized Advantage
  Estimation”：** GAE 的 bias–variance trade-off 与相应精确性条件；不能用“结论不变”略过。可查
  [原文](https://arxiv.org/abs/1506.02438)。
- **Brier score 与 Murphy decomposition：** prediction risk、reliability/calibration 与
  resolution 不同；residual square 不自动等于不可约环境噪声，更不自动标识 causal responsibility。
- **empirical Bayes / hierarchical shrinkage：** pooled estimates
  和小组样本收缩是标准思想；需要具体模型才有本文这条公式的 posterior interpretation。
- **Dawid–Skene annotator-error model：** 说明 noisy-label reliability
  通常依赖多标注者、重叠样本或额外结构来识别。与当前 one-shot interaction 不同，不能直接套用，但能帮助解释为什么仅靠 per-user
  success/residual 不能识别原因。

heteroscedastic SGD、noise-aware label weighting 和 variance-weighted policy gradient
也构成合理的 prior-art 检索方向；本审阅未核实一篇与该具体方案完全对应的文献，不编造引用，也不据此断言已经完全撞车。§16.10 列出的
P-GRPO、RGPO、FACA 等论文，正文自己承认不少内容未全文核对（L2201）；“检索未找到同时满足全部条件”不是 novelty
proof（L2203）。若有贡献，应该落在**可观测信息受限时 residual proxy 的适用边界、严谨失败分析、以及 recurring-user online
protocol 下可复现的收益** ，而不是经典加权公式或尚未证明的因果可靠度识别。

## 8. 建议的修改优先级（top 5）

1. **先修复 measurement boundary 与 identification claim。** 区分 pre/post-action
   value、事后末位置与固定边界；将 $e^2$ 改称 residual risk，给出 STOP 分支反例和无法识别 causal responsibility
   的条件。没有这一步，后续权重理论对应的量都不明确。
1. **拆开 ideal PG propositions 与 actual SAO estimator。** 明确冻结历史权重、user/task
   distribution、common-direction/positive-signal 条件；列出 adaptive GAE、DIS、staleness 和
   token normalization 的误差，修正 $K=L-j$。
1. **重写 SNR 与稳定性论证。** 定义 vector variance risk 和 sampling design，删除 45%“完全估错”保证、Adam
   等效步长和真实收益上界；保留条件 Cauchy–Schwarz/Kantorovich lemmas 及已核对数值。
1. **补齐可审计的 estimator / training loop。** 规定 pre-update critic snapshot、past-only
   table、ID 定义、窗口与 drift、shrinkage model/heuristic 标签、final clipping/normalization。分组
   held-out calibration 是检查项，不能由“EV>0”代替。
1. **用能够推翻方法的实验决定是否继续。** 在独立数据上检查 fixed-boundary A/B separation、高 STOP rates、action
   dependence、task confounding 和实际 gradient variance；再比较 SAO、仅 running
   baseline、user-conditioned critic、mean/std normalization 与 residual
   weighting，按部署目标、用户组和最差组报告不确定度。实验不得向方法泄露隐藏剧本/seed，受控 simulator 诊断要与 16.9 的在线信息约束分开说明。
