# optimization-diagonal-net：改动交接说明

**分支**：`feat/diagnet-alpha-third-setting`（工作树，尚未提交）
**基线**：`7256efe7` == `origin/fix/concurrency-staleness-batch` 的 tip
**验证**：`python3 tasks/optimization-diagonal-net/tools/check_consistency.py` → 170/170 通过

## 本次改动一句话

从代码里完全删除 `sigma`，改用 `alpha_init`（α，初始化尺度）作为第三个设定的区分轴；
四个设定统一重命名为带 α 标注的新方案；顺带修掉 3 个既有 bug（见第二节）。

设定表：

| 新标签 | dim | k | α | blob 前缀 | 备注 |
|---|---|---|---|---|---|
| `d200_k5_a1e3` | 200 | 5 | 0.001 | `a0p001` | |
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

按优先级排列。第 2、3、4 项同一个根因，见 P1 末尾。

### P0 — 合并顺序：本工作依赖 `fix/concurrency-staleness-batch`

本分支**不是**从 main 切出来的，而是直接接在 `origin/fix/concurrency-staleness-batch`
的 tip 上（领先 `merge-base(main, HEAD)` = `8a9bd778` 共 33 个 commit）。该分支尚未合入 main。

依赖是硬的：本 task 的 eval 脚本引用 `scripts/fixed_entry.py`，而该文件只存在于那个分支
（main 上不存在，main 版脚本是直接 `python RAIN/opt_diagonal_net/custom_optimizer.py`）。
另外 `mlsbench/agent/input_stager.py`、`tools.py` 的 ephemeral-inputs 基础设施也来自那里。

**需要注意的一点**：那个分支引入了 `_sigma_tag()` / `sig{tag}` 作为 blob 区分轴来修 #82；
本次改动把它替换成 `_alpha_tag()` / `a{tag}`。两者解决同一个问题，方案互斥，但**不构成冲突**
——因为是先后关系而非并行。按 `fix/concurrency-staleness-batch` → 本分支 的顺序合并即可，
中间状态自洽（sigma 版本本身能跑）。请不要试图把本分支单独 rebase 到 main。

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
  `d7fc0b58…`，而 HEAD 版本的 task 目录算出来是 `da9a3832…` ——两者都不是我造成的。
- 本次改动后 diagonal-net 的正确值：
  `sha256:9d8f03bc890d5c9e0898da0f5b0c4ba3ed510f18fb9b6ebd210ab558f7886f67`

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

## 三、仍待 GPU 验证的部分

无 GPU 环境，以下两项按决策③（底线是不能 saturate）留待补齐：

- **α=0.5 尚未经 GPU 验证**，它目前是**基于证据的默认值**而非实测结论。依据：现有 α=1.0 的隐藏
  设定里 adagrad 的 `n*` 已经顶到 2000 —— 正好是它的 `--grid-max`，即已经饱和；所以在 d=500 上
  直接用 α=1.0 很可能同样顶格。α=0.5 配 `--grid-max 2000` 留了余量。
- **`leaderboard.csv` 里 `d500_k10_a5e1` 的 score / n_star / elapsed 三列已留空**，等实测填。
  其余三个设定的数据保留：sgd 50/62/–/487，adagrad 175/487/–/2000，adam 50/56/–/350，adam2 50/53/–/350。

校准工具：`tools/alpha_sweep.py`。它在 (d=500, k=10) 上扫 α × baseline，复用了
`_alpha_tag` / `_input_key` / `resolve_grid` 的同一套逻辑，每个 seed 的数据只生成一次然后以多个
α 文件名写出，并会报出 SATURATED（`n*` 顶到网格上界）、DEGENERATE（所有 baseline 打平）以及
相对 α=1e-3 参考的方法排序变化；结尾会打印「若换用别的 α，需要改哪 6 处」。
已在 CPU 上跑通两轮 smoke（走到过 SATURATED 和 解除-SATURATED/DEGENERATE 两条判定路径），
但那两轮用的是 d=50/k=3 的玩具配置，只验证管路，**不是校准结论**。

按决策③：如果实测发现调不出不饱和的 α，就照实报告，不强求方法排序发生变化。

---

## 四、怎么验证这次改动

```bash
python3 tasks/optimization-diagonal-net/tools/check_consistency.py   # 170/170
```

8 组检查：残留的 sigma / 旧标签；5 份 `fixed_benchmark.py` 副本字节一致；16 个设定脚本在
4 个目录间一致；`--inputs-glob` 前缀与 `_input_key()` 的输出一致；**各设定 blob 前缀互不相交
（issue #82 的回归哨兵）**；签名与可编辑区间在 scaffold 和全部 edit op 之间一致；
`pristine_manifest.json` 的哈希；以及 py/sh 语法。
（排除 `mlsbench_src`——vendored；排除 `tools`——自身会命中自己的正则。）
