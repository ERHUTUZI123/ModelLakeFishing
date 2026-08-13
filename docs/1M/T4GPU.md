# T4 在 watGPU 上怎么跑

上级 runbook：[`100kplan.md`](100kplan.md) §7。结果回填在 [`T4.md`](T4.md)。

你要做的事：提交一个 sbatch 作业，GPU 占用不到 5 分钟；跑完把 §6 的四样东西发我，并回答 §7 的三个问题。

---

## 1. 现在是什么状态

T4 的代码写完了，我在本地 RTX 4060（8 GB）上跑通了一遍，端到端 40 秒，`x_m [100000, 448]` 已落盘，
同机重跑逐字节相同（sha256 `6e8b6924…`）。出闸门五条过了四条，G-B4（CORE/HALO 可分性 AUC）没过，
数字和归因在 `T4.md` §4.5。

这份文档要解决的是另外两件事：

1. T5 建图、T6 训练、T7 导出、T8 评测都在 watGPU 上，特征也得在那边。
2. T4 是整个流程里最轻的 GPU 作业，拿它验证 watGPU 的环境最便宜——跑错只赔 5 分钟，T6 跑错赔几个 GPU 小时。

有一点要先说清楚：MiniLM 在不同设备上末位有浮点差异，本地跑的 `x_m` 和 watGPU 跑的不会逐字节相同
（前 30,183 行会，因为那部分是从 `hgraph_ml_v2.pt` 复制的；后 69,817 行不会）。
所以只能有一份是正式件，另一份是对照。默认走 §7 问题 1 的 A 方案。

---

## 2. 提交前要确认的三件事

| # | 前提 | 怎么确认 | 没有怎么办 |
|---|---|---|---|
| 1 | SSH 能登进去 | `ssh x98liu@watgpu.cs.uwaterloo.ca hostname` | 走 `T1.md` §1：AuthMan 传公钥、核对 host fingerprint |
| 2 | 知道能用哪个 partition、要不要 `--account` | 登录节点跑 `sinfo -o "%P %N %G %m %c %l"` 和 `sacctmgr show assoc where user=$USER format=Account,Partition,MaxWall` | 别猜，问 `watgpu-admin@lists.uwaterloo.ca` |
| 3 | 大文件放哪，quota 够不够 300 MB | `quota -s; df -h $HOME` | 见 `T1.md` §3。不要预设 `/scratch` 或 `$WORK` 存在 |

登录节点只做 git、传文件、`sbatch`、`squeue`。watGPU 明文要求不在登录节点跑 `nvidia-smi` 和训练。

---

## 3. 要传上去什么

T2/T3/T4 那批代码（`scale1m/`、`configs/`、`scripts/watgpu/`、`docs/1M/`）现在还是 untracked，
远端 `git clone` 拿不到。先提交推到私有分支：

```powershell
Set-Location D:\research\model_lake\codes\ModelLakeFishing
git checkout -b r2-100k
git add scale1m configs scripts docs/1M stage2TrainGraphSAGE scale
git commit -m "T0-T4: scale1m pipeline + watgpu scripts"
git push -u origin r2-100k
```

另外两个文件 git 里没有，得单独传。`hgraph_ml_v2.pt` 被 `.gitignore:28` 忽略；梯子在 data 盘。

| 文件 | 大小 | 本地路径 | 远端去处 |
|---|---|---|---|
| `hgraph_ml_v2.pt` | 92 MB | `ModelLakeFishing\stage1BuildTransferGraph\` | `$PROJECT_ROOT/stage1BuildTransferGraph/` |
| `100k_model_ids.csv` | 6.9 MB | `D:\research\model_lake\data\data1m\ladder\` | `$DATA_ROOT/data1m/ladder/` |

`hgraph_ml_v2_sub.pt` 是 T6 跑 12K 锚点要用的，T4 用不上，可以一起传省一次往返。

```powershell
scp D:\research\model_lake\codes\ModelLakeFishing\stage1BuildTransferGraph\hgraph_ml_v2.pt `
    x98liu@watgpu.cs.uwaterloo.ca:<REMOTE_PROJECT_ROOT>/stage1BuildTransferGraph/
scp D:\research\model_lake\data\data1m\ladder\100k_model_ids.csv `
    x98liu@watgpu.cs.uwaterloo.ca:<REMOTE_DATA_ROOT>/data1m/ladder/
```

传完两端各算一次 sha256 对一下。梯子的行序决定了哪个模型拥有哪一行 embedding，传坏了不会报错。

```powershell
Get-FileHash D:\research\model_lake\data\data1m\ladder\100k_model_ids.csv -Algorithm SHA256
```
```bash
sha256sum <REMOTE_DATA_ROOT>/data1m/ladder/100k_model_ids.csv
```

两边都应该是 `ccf288aee2a3caee10b60ab8a7ee3031f44d5766e33074e4090eb35cb128b0ab`，
这个值 T3 已经写进 `LADDER_REPORT.json`。

---

## 4. 环境（登录节点，一次性）

```bash
python -m venv "$HOME/venvs/mlf" && source "$HOME/venvs/mlf/bin/activate"
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121   # 按计算节点实测 CUDA 改
pip install torch_geometric sentence-transformers pandas pyarrow scikit-learn numpy
python -m pip check
pip freeze > "$HOME/mlf_pipfreeze_$(date -u +%Y%m%d).txt"
```

`torch_geometric` 在 T4 里只是被顺带 import：计划 §1 要求 HALO 用 CORE 那个 `model_descriptor`，
所以 `embed_lake.py` 直接 import `scale.modellens_build_graph`，那个模块顶部有 `torch_geometric`。
T5/T6 本来就要装，现在装不亏。

MiniLM 的权重要先在登录节点拉下来。计算节点可能没有外网，真没有的话作业会卡在下载上直到 walltime：

```bash
export HF_HOME="<大文件目录>/hf_cache"
mkdir -p "$HF_HOME"
python - <<'PY'
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2")
print("cached ok:", m.encode(["hello"]).shape)
PY
```

写一个 `~/.mlf_env`，每次作业前 source（不要提交进 git）：

```bash
export PROJECT_ROOT=<REMOTE_PROJECT_ROOT>
export DATA_ROOT=<REMOTE_DATA_ROOT>
export OUTPUT_ROOT=<REMOTE_OUTPUT_ROOT>
export HF_HOME=<大文件目录>/hf_cache
export ENV_ACTIVATE=$HOME/venvs/mlf/bin/activate
```

---

## 5. 提交

脚本是 [`scripts/watgpu/embed_lake.sbatch`](../../scripts/watgpu/embed_lake.sbatch)，
请求 `--gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:30:00`。
真实需求比这小得多（本地 40 秒，显存峰值不到 1 GB），给宽是因为排队 30 分钟比作业被 kill 便宜。

```bash
source ~/.mlf_env
cd "$PROJECT_ROOT"
git pull --ff-only
mkdir -p logs
sbatch --export=ALL,RUN_ID=T4_100k_$(date -u +%Y%m%dT%H%M%SZ) scripts/watgpu/embed_lake.sbatch
```

第 2 步查出来要指定 partition/account 的话：

```bash
sbatch --partition=<PARTITION> --account=<ACCOUNT> \
       --export=ALL,RUN_ID=T4_100k_$(date -u +%Y%m%dT%H%M%SZ) \
       scripts/watgpu/embed_lake.sbatch
```

监控：

```bash
squeue -u "$USER"
tail -f logs/mlf-t4-embed-<JOB_ID>.out
sacct -j <JOB_ID> --format=JobID,State,Elapsed,MaxRSS,AllocTRES,ExitCode
```

---

## 6. 怎么判断跑对了

Slurm 报 `COMPLETED` 只说明进程正常退出。日志尾部应该是这样，逐行对：

```
[ok] x_m [100000, 448] -> <DATA_ROOT>/data1m/feats/100k
  sha256                 <和本地不同，正常>
  CORE verbatim          True
  shape / no NaN         True / True
  row-order (100 rows)    max|d| 0.00e+00 -> True
  family_vocab           341 -> 1896 (+1555)
  descriptor chars       CORE 52.3 | HALO 57.3 (delta +5.0)
  size clause            CORE 0.4791 | HALO 0.5775
  G-B4 separability AUC  {'full': 0.97..., 'e_name': 0.70..., 'e_desc': 0.97...}
  G-B4 clause ablation   {'name_only': 0.84..., 'name_size': 0.93..., 'name_family': 0.91..., 'full': 0.96...}
```

判读：

- 前四行任何一个不是 True，停下来，把日志全文发我，别往 T5 走。这四行分别对应计划 §7.3 的
  「CORE 特征逐字节未变」「shape 与无 NaN」「行序 checksum」三条。
- 中间三行只依赖文本、不依赖 GPU，应该和上面写的完全一致。不一致说明传上去的梯子或 CORE 图
  和本地不是同一份，回 §3 重新对 sha256。
- 最后两行的 AUC 会有 ±0.01 的浮动（MiniLM 浮点差加抽样种子），正常。数值没落进
  计划要求的 [0.5, 0.75] 是已知情况，见 §7 问题 3，不要因为它红就停。

---

## 7. 跑完发我什么

`x_m.npy` 有 179 MB，不用下载，T5–T8 都在 watGPU 上，它待在那儿就行。我只要小文件：

```bash
cd "$DATA_ROOT/data1m/feats/100k"
tar czf ~/t4_report.tgz FEATS_REPORT.json family_vocab.csv halo_descriptors_head.txt
```

| # | 内容 | 用途 |
|---|---|---|
| 1 | `FEATS_REPORT.json` | 全部数字，回填 `T4.md`，和本地这次逐项对账 |
| 2 | `family_vocab.csv`（1,896 行） | 确认 CORE 那 341 行 id 没动；T6 的 checkpoint 要和它绑定 |
| 3 | `logs/mlf-t4-embed-<JOB_ID>.out` 全文 | 过程留痕，成功失败都写进报告 |
| 4 | `runs/<RUN_ID>/metadata/run.txt` 和 `nvidia-smi.txt` | 记进 `WATGPU_ENV.md`，GPU 型号/CUDA/git commit，顺带把 T1 的 G-W2 过了 |

另外口头回我三个数：排队等了多久、GPU 实际跑了多久、分到的是什么卡。
计划里 T6 写的 `--time 04:00:00` 是估的，这三个数用来修正它。

---

## 8. 需要你定的三件事

### 问题 1：哪一份 `x_m` 算正式件

| | A 方案（建议） | B 方案 |
|---|---|---|
| 做法 | watGPU 上这一份是正式件，本地那份当对照 | 本地这份上传，watGPU 不重跑 |
| 好处 | 特征和 T5–T8 在同一台机器上产生；顺带验证了 GPU 环境 | 省一次作业 |
| 代价 | 本地数字在报告里要标成非正式 | 多传 179 MB；T4 从没在 watGPU 上验证过，问题推迟到 T6 才暴露 |

选 A 的话，`T4.md` 里的正式数字我全部换成你跑回来的。

### 问题 2：`family_vocab` 从 341 行涨到 1,896 行，接不接受

新增的 1,555 行里有真家族（`roberta` 1521 个、`xlm-roberta` 962、`distilbert` 857、`qwen3` 378、`modernbert` 318），
也有明显是噪声的（`test`、`demo`、`readme`、`final`、`age`、`amazon`）——它们只是某个名字前缀被至少 3 个仓库用过。

建议接受，不动。`FAMILY_MIN_COUNT=3` 是 CORE 自己用的规则，给 HALO 单独加一层过滤等于两群用不同规则，
那才是真破坏受控实验。噪声家族最坏结果是学出一行没信息的 embedding，污染不到 CORE 的 341 行
（append-only 在代码里是断言，不是日志）。

不同意的话现在说。T6 训完之后 vocab 和 checkpoint 就绑死了，改它等于重训。

### 问题 3：G-B4 没过门（AUC 0.973，门是 ≤0.75），怎么办

计划 §7.3 要求 AUC 偏高时先归因再考虑回滚描述符。我把描述符的子句一个个关掉重测了：

| 描述符内容 | AUC |
|---|---|
| 只有名字（family、size 子句都关掉） | 0.847 |
| 名字 + size 子句 | 0.931 |
| 名字 + family 子句 | 0.914 |
| 完整（现状） | 0.966 |

把描述符退到只剩名字，AUC 也只降到 0.847，够不到 0.75。可分性的地板来自
「CORE 是 ModelLens 那份基准语料、HALO 是 2026 年的 HF 均衡爬取」这个事实本身，不是描述符函数选错了。
计划 §1 第二条要防的是流水线人为制造差异，这一层是干净的。

三个选项：

| | 做法 | 代价 |
|---|---|---|
| A（建议） | 不回滚描述符。G-B4 改成报两个数：绝对 AUC 0.966，以及流水线贡献的增量 0.966 − 0.847 = 0.119。两个都写进报告，T8 的冷启动分析把它列为混杂因素 | 报告里要写「两群本来就可分」，不能再说「扛住了 83× 同分布干扰」 |
| B | 砍掉 family/size 子句把 AUC 压到 0.847 | 主动丢信息，AUC 还是不过门，白亏 |
| C | 改 HALO 选样让它更像 CORE | 推翻 T2 的 G1–G20 均衡总体，代价极大，而且「像 CORE」没有客观定义 |

T5 不依赖这个问题，可以先跑；但 T8 的结论怎么写由它决定。请回一个 A/B/C。

---

## 9. 容易踩的坑

| 症状 | 原因 | 处理 |
|---|---|---|
| 作业挂着不动，日志停在 SentenceTransformer | 计算节点无外网，在等 HF 下载 | §4 的预下载没做；`scancel` 后补上，确认 `HF_HOME` 通过 `--export=ALL` 传进去了 |
| `ModuleNotFoundError: scale1m` | 不在 `PROJECT_ROOT` 下跑，或用的是系统 python | 脚本里已 `cd "$PROJECT_ROOT"`；查 `run.txt` 里的 `PYTHON=` |
| `FileNotFoundError: hgraph_ml_v2.pt` | 被 `.gitignore` 忽略，clone 不会带 | §3 单独 scp |
| `AssertionError: ladder CORE prefix != ...` | 传上去的梯子或 CORE 图版本不对 | 回 §3 对 sha256，别改代码绕过断言 |
| `AssertionError: family row moved` | vocab 文件被手改过，或换了 CORE 图 | 删掉 `feats/100k/family_vocab.csv` 重跑，它每次从 CORE 重新生成 |
| 长时间 `PENDING` | partition/account 没权限，或请求过严 | `scontrol show job -dd <ID>` 看 Reason，别反复 cancel 重投 |
| 日志里出现 `D:\` 或 `C:\` | Windows 路径漏进了 resolved config | 告诉我，这是 G-W3 要拦的情况 |
