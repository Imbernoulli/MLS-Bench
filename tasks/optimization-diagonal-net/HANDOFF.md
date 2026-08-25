# optimization-diagonal-net：改动交接说明

**分支**：`feat/diagnet-alpha-third-setting`（2 个 commit：代码改动 + leaderboard 重测）
**基线**：`7256efe7` == `origin/fix/concurrency-staleness-batch` 的 tip，该分支已由 PR #81 并入 main
**验证**：`python3 tasks/optimization-diagonal-net/tools/check_consistency.py` → 170/170 通过；
leaderboard 四个设定 × 四个 baseline 全部重测（详见 commit message）

## 本次改动一句话

从代码里完全删除 `sigma`，改用 `alpha_init`（α，初始化尺度）作为第三个设定的区分轴；
四个设定统一重命名为带 α 标注的新方案；顺带修掉 3 个既有 bug（见第二节）。

设定表：

| 新标签 | dim | k | α | blob 前缀 | 备注 |
|---|---|---|---|---|---|
| `d200_k20_a1e3` | 200 | 20 | 0.001 | `a0p001` | |
| `d500_k10_a1e3` | 500 | 10 | 0.001 | `a0p001` | |
| `d500_k10_a5e1` | 500 | 10 | 0.5 | `a0p5` | `--grid-max 2000`，time 06:00:00 |
| `d10000_k50_a1e0` | 10000 | 50 | 1.0 | `a1` | hidden |

`get_hyperparameters` 的签名增加了 `alpha_init: float`。签名增长被吸收进原有的空行 padding
（8 行缩到 3 行），因此 **`def main()` 仍在第 91 行、可编辑区间仍是 23–90**：`config.json`
的 `edit` 区间、20 个 `*.edit.py` 的 `OPS`、instruction 里的行号横幅全部保持字节不变。

**issue #82 由构造消除**：`d500_k10_a1e3` 与 `d500_k10_a5e1` 的 blob 前缀分别是 `a0p001` 与
`a0p5`，不再共享输入文件；此前两列位级相同的 leaderboard 数据也随之消失。

---

## 一、需要 project lead 处理的全局事项

按优先级排列。P2、P3 与 P1 同一个根因（见 P1 末尾）；P4 是独立的一条，且不限于本 task。

### P0 —（已作废）合并顺序依赖已解除

本分支不是从 main 切出来的，而是接在 `origin/fix/concurrency-staleness-batch` 的 tip
（`7256efe7`）上；写这份文档时该分支尚未合入 main，所以原本要求按顺序合并。

**现已解除**：`fix/concurrency-staleness-batch` 已通过 PR #81 并入 main（`cfd57a7e`），
`7256efe7` 现在是 `origin/main` 的祖先。本分支只有 2 个 commit 领先 main、0 个落后，
且与 main 新增的改动无文件重叠，可以直接对 main 开 PR，无需 rebase 或排序。

留档一句以免误读历史：那个分支曾用 `_sigma_tag()` / `sig{tag}` 作为 blob 区分轴来修
issue #82，本次改动把它替换成 `_alpha_tag()` / `a{tag}`。两者解决同一问题、方案互斥，
但因为是先后关系而非并行，不构成冲突。

### P1 — `adapter.py:204` 的 `_ALLOWED_OP_IMPORT_ROOTS`：17/140 个 task 无法重新渲染

```
_ALLOWED_OP_IMPORT_ROOTS = {"custom_template", "importlib", "json", "math", "pathlib", "sys"}
```

`_load_ops_file()` 用这个白名单加载三类文件：baseline 的 `*.edit.py`、`tasks/<id>/edits/mid_edit.py`、
`pkg_configs/<pkg>/pre_edit.py`。问题出在 `mid_edit.py`：

- 全部 48 个 `pre_edit.py` 合规，全部 `*.edit.py` 也合规；
- 但 **17/140** 个 task 的 `edits/` 下有文件 import 了白名单外的模块，于是这些 task 一旦
  重新渲染就抛 `ImportError: import of 'base64' is not allowed while loading edit ops`；
- diagonal-net 需要：`base64`、`dgp`、`io`、`numpy`、`os`；
- 全体缺失模块并集：`base64`、`dgp`、`io`、`numpy`、`os`、`re`、以及相对 import。

设计上像是配错了：白名单的注释说它是为「声明式 op 构造」准备的，而 `mid_edit.py` 是**宿主侧的
数据生成器**，天然需要 numpy / base64 / io / os。建议把 `mid_edit.py` 与 `*.edit.py` 分开走
不同的加载路径，而不是继续扩这一个白名单（连 `re` 都不在里面，可见它有多窄）。

**附带的一个坑**：`--overwrite` 是**先删后渲**。渲染中途失败会留下一个已被删除的 task 目录。
我踩过一次，靠事先做的备份恢复。建议改成渲染到临时目录再原子替换（`_finalize` 那边已经有
`out_dir.rename(final_dir)` 的写法可以复用）。

> **这是 P2、P3 的根因**：因为「重新渲染」实际不可用，`harbor/` 下的产物只能手工维护，
> 漂移必然累积。下面两项就是同一件事的两个症状。

### P2 — `harbor/tasks/dataset.toml`：39/140 个 digest 已陈旧

这是全局单文件（140 条，每条只有 `name` + `digest`，**不含任何设定标签**），我没有动它。

- digest 走的是 fallback 的 `_manual_task_digest()`，不是 harbor 的 `Packager.compute_content_hash()`
  ——证据：101/140 与 `_manual_task_digest()` 精确吻合（不可能是巧合），因为生成时 `harbor` 包不可导入。
  所以这个值是可以离线复算的。
- **39/140 已经陈旧**，且 diagonal-net 在我动手**之前**就已陈旧：`dataset.toml:382` 写的是
  `d7fc0b58…`，而当时 HEAD 版本的 task 目录算出来是 `da9a3832…` ——两者都不是我造成的。
- 本分支（含把第一个 setting 从 k=5 调到 k=20 的改动）之后，diagonal-net 的正确值是：
  `sha256:098dbeb8f1f7c28413e0c24018b85aabead3a47ce418f6b143e51ebd5faa104c`
  （注意 `_manual_task_digest()` 把 `tests/` 整棵树纳进来，所以 `tests/meta/leaderboard.csv`
  一变这个值就变；如果合并前又重测了 leaderboard，请重算而不要照抄这一串。
  前一版 k=5 时的值是 `c44e2694…`，已随本次 k=20 改动作废。）

建议整体重新生成（`python -m mls_bench.main --output-dir <dir>`），因为还有另外 38 个也是陈旧的
——但这被 P1 挡住。如果只想让本 task 自洽，手改 `dataset.toml:382` 那一行即可。

### P3 — 所有 rendered `task.toml` 的 `[verifier].timeout_sec` 都在漂移

对全部 140 个 task 用当前 `adapter._verifier_timeout_sec()` 重算并与 rendered 值比对：

- **140/140 不一致**；
- 其中 139 个的差值恰好是 300 的整数倍（`+300`：98 个，`+600`：25 个，`+900`：13 个，`+1200`：3 个），
  即 `WAVE_GRACE_SEC × wave 数` ——说明所有 rendered `task.toml` 都早于 `WAVE_GRACE_SEC` 的引入，
  此后从未重新渲染过；
- 方向是「rendered 值比 adapter 认为需要的**偏紧** 300–1200s」。按 `_verifier_timeout_sec()`
  自己的注释，这正是它要修的那个 bug：外层 verifier 可能在某个 wave 自己的 deadline 之前就把它杀掉。
  实际风险目前很低——30 分钟的 flat headroom 盖住了 300–1200s 的缺口，注释说要到 ~10 waves 才会暴露，
  而最多的才 4 waves。所以这更多是「产物无法重新生成」的最干净证据，而非当下的事故源。
- **唯一的异常**：`ai4sci-climate-emulation`，rendered 64800 vs 重算 60060，差 −4740，
  方向相反且不是 300 的倍数 → 它的 `config.json` 在渲染之后被改小过，或者 `task.toml` 被手改过。
  这个值得单独看一眼。

顺带说明本 task 的情况：我把 `d500_k10_a5e1` 的 time 从 04:00:00 提到 06:00:00，
**不影响 verifier timeout**——它和 8 小时的 `d10000_k50_a1e0` 同在 group 2，wave 的 deadline
由组内最慢的成员决定，重算前后都是 46080。所以 `task.toml` 不需要因我的改动而变。
（同理，那个 6 小时的声明其实不是硬约束，`a5e1` 只要在 8 小时内跑完就行。）

---

### P4 — leaderboard 的 `n*` 依赖硬件：跨机器数字不可直接比较

这不是 diagonal-net 独有的，凡是「阈值 + 多 seed 投票」型判据的 task 都适用，所以列在这里。

本 task 的判据是「test MSE < 1.0 在 5 个 seed 里至少 4 个成立」，`n*` 是满足它的最小网格点。
同一份代码、同一份数据、同一批 seed，换一批 GPU 就得到不同的 `n*`：

| 格子 | 旧硬件 | 本次硬件（H20） |
|---|---|---|
| `d500_k10` adam | 56 | 59 |
| `d500_k10` adam2 | 53 | 56 |
| `d10000_k50` adam2 | 350 | 362 |

同批 12 个格子里另外 9 个（含 sgd、adagrad 全部）精确复现，说明不是随机噪声，而是**贴着判据边界
的那些 baseline 会被浮点归约顺序翻掉一个 seed**；离边界远的完全稳定。同机器上重跑是确定性的
（改动前/改动后代码各跑一遍，三个值都逐一重现）。

含义与建议：

- 排行榜数字应当**连同测量硬件一起记录**，否则跨提交/跨模型的 ±5% 级差异无法区分是方法差异还是
  机器差异。`leaderboard.csv` 目前没有这一列。
- 判据可以做得更稳（例如把「≥4/5」放宽成带容差的判据，或对 MSE 做多次平均），但那会改变 task
  定义、动到所有历史分数，需要 lead 决策，我没有动。
- 至少在下次全量重测时，让同一批硬件跑完全部 baseline —— 本次是这么做的（16 个格子同一批 H20）。

---

## 二、本 task 范围内已自行修掉的 3 个既有 bug

不需要 lead 动手，列在这里是为了让 review 的人知道这些改动的来由。

1. **隐藏设定永远找不到自己的输入 blob**。`tests/eval/scripts/d10000_k50.sh` 缺了
   `--n-test 10000` 和 `--grid-max 2000`，而宿主侧生成器是带着这两个参数生成的。
   `n_test` 编在 blob 文件名里（`…_nmax{n}_nt{n}_…`），所以脚本按 `nt=2000` 之类去找，
   而文件是按 `nt=10000` 落盘的，必然 miss。
2. **文档里的签名有一个不存在的参数**。`noise_scale` 出现在 `get_hyperparameters` 的
   docstring 里，但代码里从来没有过这个参数。
3. **instruction.md 的 baseline 代码快照陈旧**。还显示 `if __name__ == "__main__":`，
   自 `def main()` 重构之后就没更新过。本次 5 段快照全部按 adapter 的格式重新生成。

---

## 三、GPU 实测结果（已完成）

四个设定 × 四个 baseline 全部重测，每格 5 个 seed（42–46），每格独占 1 张 GPU 并行跑完，
墙钟 4.6 小时。每个设定用的是它自己 `scripts/<name>.sh` 的原始参数（两个 α=1e-3 的设定不传
`--grid-max`，网格上界 1600；另两个到 2000）。`score` 严格等于 `-log2(n*)`。

| n\* | `d200_k20_a1e3` | `d500_k10_a1e3` | `d500_k10_a5e1` | `d10000_k50_a1e0` |
|---|---|---|---|---|
| sgd | 78 | 62 | 62 | 487 |
| adagrad | 200 | 487 | 500 | 2000 |
| adam | 71 | 59 | 65 | 350 |
| adam2 | 68 | 56 | 53 | 362 |

**α=0.5 可用，决策③的底线达到**：不饱和（最大 `n*`=500，网格上界 2000）、不退化（四值互不相同），
并且排序确实变了 —— α=1e-3 是 `adam2 < adam < sgd < adagrad`，α=0.5 是
`adam2 < sgd < adam < adagrad`。只抬初始化尺度就换了 regime，这是原先那个重复设定做不到的。

**旧设定的回归**：12 个旧格子里 9 个精确复现，3 个变了（`d500_k10` 的 adam 56→59、adam2 53→56，
`d10000_k50` 的 adam2 350→362）。这是机器差异而非行为变化，并且是**验证过的**：把改动前的代码
（`129a483d^`，用它自己的 `--sigma` 旧 CLI 和 `sig` 标签 blob）在同一批 GPU 上重跑，同样得到
59 / 56 / 362，而 sgd 对照两版都是 62；改动后的代码复跑也精确重现 59 / 56 / 362，说明这台机器是
确定性的，差异落在机器之间。静态上也只能如此：σ 从不进入 benchmark 数学（被删掉的那个 flag 的
help 文本自己就这么写，它只出现在 blob 文件名里）、数据生成器未改、唯一的功能改动
（`alpha_init` 传进 `get_hyperparameters`）没有任何 baseline 读取。

`leaderboard.csv` 的 elapsed 列也一并换成本次硬件的实测值，两份副本保持字节一致。

## 四、怎么验证这次改动

```bash
python3 tasks/optimization-diagonal-net/tools/check_consistency.py   # 170/170
```

8 组检查：残留的 sigma / 旧标签；5 份 `fixed_benchmark.py` 副本字节一致；16 个设定脚本在
4 个目录间一致；`--inputs-glob` 前缀与 `_input_key()` 的输出一致；**各设定 blob 前缀互不相交
（issue #82 的回归哨兵）**；签名与可编辑区间在 scaffold 和全部 edit op 之间一致；
`pristine_manifest.json` 的哈希；以及 py/sh 语法。
（排除 `mlsbench_src`——vendored；排除 `tools`——自身会命中自己的正则。）
