### 附录 P4-R：同数据集推荐结果实例（我方 vs ModelLens，同 12K 宇宙）

> **怎么读**：两系统对**同一个 held-out 查询**排同一批 12,000 个模型；`acc` = 该模型在这个数据集上的公开记录归一化准确率，`—` = 该模型没在这个数据集上被评测过（不代表它差）。我方 = held-out（无泄漏）嵌入的纯 MIPS `z_d·z_m`；ModelLens = 其自身打分。**ModelLens 是 blind release 版**（`dataset_desc` 与 `dataset2id` 均未发布，看不见「这是哪个数据集」，P2 §3.1）—— 故它倾向返回 (task,metric)-泛化强但非本数据集专属的模型；此对比是「公开件可复现」口径，非同台方法优劣。

#### bbh_boolean_expressions — Question Answering

该数据集最优 acc = **0.960**，3491 个带标签模型（候选宇宙 12,000）。
我方 top-10 带标签 6/10；ModelLens top-10 带标签 1/10。

**我方 L1L3b（MIPS，held-out）：**

| rank | model | acc | MIPS |
|---|---|---|---|
| 1 | `alea-institute/charboundary-medium-onnx` | — | 0.901 |
| 2 | `alea-institute/charboundary-medium` | — | 0.899 |
| 3 | `lyra4-gutenberg2-12b` | **0.840** | 0.893 |
| 4 | `euphrates-14b` | **0.892** | 0.888 |
| 5 | `alea-institute/charboundary-small-onnx` | — | 0.888 |
| 6 | `thebeagle-v2beta-32b-mgs` | **0.922** | 0.887 |
| 7 | `falcon3-10b-tensopolis-v2` | **0.876** | 0.885 |
| 8 | `alea-institute/charboundary-large-onnx` | — | 0.884 |
| 9 | `falcon3-10b-tensopolis-v1` | **0.863** | 0.884 |
| 10 | `deepseek-lumen-r1-qwen2-5-14b` | **0.752** | 0.883 |

**ModelLens（blind release 版）：**

| rank | model | acc | score |
|---|---|---|---|
| 1 | `samerzaher80/aethermind-srl` | — | 27.29 |
| 2 | `rifre` | — | 24.30 |
| 3 | `dqubit/frida-f16` | — | 23.87 |
| 4 | `169pi/alpie-core` | — | 23.33 |
| 5 | `zai-org/glm-4-7` | — | 22.53 |
| 6 | `gemini-3-0-pro` | — | 22.18 |
| 7 | `li-14b-v0-4-slerp0-1` | **0.892** | 22.07 |
| 8 | `zomba/fsg-net` | — | 21.84 |
| 9 | `gladiator/microsoft-deberta-v3-large-ner-wnut-17` | — | 21.79 |
| 10 | `r@m` | — | 21.73 |

#### GPQA (0-shot) — Text Generation

该数据集最优 acc = **0.384**，1306 个带标签模型（候选宇宙 12,000）。
我方 top-10 带标签 4/10；ModelLens top-10 带标签 1/10。

**我方 L1L3b（MIPS，held-out）：**

| rank | model | acc | MIPS |
|---|---|---|---|
| 1 | `mobilellm-pro-base-int4-cpu` | — | 0.914 |
| 2 | `word2li/mistral-7b-v0-3-middo-wizard` | — | 0.889 |
| 3 | `triangle104/primal-opus-14b-optimus-v2-q5-k-s-gguf` | **0.189** | 0.885 |
| 4 | `fluently/fluentlyqwen2-5-32b` | **0.182** | 0.878 |
| 5 | `word2li/mistral-7b-v0-3-middo-alpaca` | — | 0.876 |
| 6 | `triangle104/primal-opus-14b-optimus-v2-q8-0-gguf` | **0.189** | 0.869 |
| 7 | `devquasar/coma-7b-v0-1` | — | 0.865 |
| 8 | `triangle104/primal-opus-14b-optimus-v2-q4-k-m-gguf` | **0.189** | 0.863 |
| 9 | `baseline-gemma-3-270m` | — | 0.863 |
| 10 | `justinj92/qwen3-hermes8b-v1` | — | 0.862 |

**ModelLens（blind release 版）：**

| rank | model | acc | score |
|---|---|---|---|
| 1 | `samerzaher80/aethermind-srl` | — | 26.73 |
| 2 | `169pi/alpie-core` | — | 25.06 |
| 3 | `gemini-3-0-pro` | — | 24.61 |
| 4 | `rifre` | — | 24.17 |
| 5 | `zomba/fsg-net` | — | 22.95 |
| 6 | `alirzb/s5-m1-fold1-swint-42507053` | — | 22.37 |
| 7 | `cchance27/calmerys-78b-orpo-v0-1-q4-mlx` | **0.200** | 21.95 |
| 8 | `dqubit/frida-f16` | — | 21.93 |
| 9 | `gladiator/microsoft-deberta-v3-large-ner-wnut-17` | — | 21.38 |
| 10 | `goodcasper/vit-4090` | — | 21.11 |

#### MTEB MTOPDomainClassification (en) — Classification

该数据集最优 acc = **0.992**，744 个带标签模型（候选宇宙 12,000）。
我方 top-10 带标签 10/10；ModelLens top-10 带标签 0/10。

**我方 L1L3b（MIPS，held-out）：**

| rank | model | acc | MIPS |
|---|---|---|---|
| 1 | `barnowak/gte-qwen2-7b-instruct-q4-k-m-gguf` | **0.990** | 0.938 |
| 2 | `gme-qwen2-vl-7b-instruct` | **0.976** | 0.920 |
| 3 | `bnightning/gte-qwen2-7b-instruct-q4-k-m-gguf` | **0.990** | 0.916 |
| 4 | `gte-qwen1-5-7b-instruct` | **0.958** | 0.912 |
| 5 | `nizzouuu/gte-qwen2-7b-instruct-q6-k-gguf` | **0.990** | 0.912 |
| 6 | `datouge/gte-qwen2-7b-instruct-q4-0-gguf` | **0.990** | 0.909 |
| 7 | `bn7002/gte-qwen2-7b-instruct-q4-k-m-gguf` | **0.990** | 0.906 |
| 8 | `niancheng/gte-qwen2-7b-instruct-q4-k-m-gguf` | **0.990** | 0.903 |
| 9 | `lxc1999/gte-qwen2-7b-instruct-q4-k-m-gguf` | **0.990** | 0.901 |
| 10 | `brtaydin/gte-qwen2-7b-instruct-q5-k-m-gguf` | **0.990** | 0.898 |

**ModelLens（blind release 版）：**

| rank | model | acc | score |
|---|---|---|---|
| 1 | `rifre` | — | 27.12 |
| 2 | `169pi/alpie-core` | — | 25.61 |
| 3 | `samerzaher80/aethermind-srl` | — | 24.10 |
| 4 | `dqubit/frida-f16` | — | 23.59 |
| 5 | `viktorzver/frida` | — | 21.38 |
| 6 | `zomba/fsg-net` | — | 20.95 |
| 7 | `gemini-3-0-pro` | — | 20.67 |
| 8 | `botbotrobotics/cabrallama3-70b` | — | 19.31 |
| 9 | `palm-540b-self-improvement,-self-consistency` | — | 19.11 |
| 10 | `li-14b-v0-4-slerp0-1` | — | 19.09 |

#### MTEB ArguAna — Retrieval

该数据集最优 acc = **0.831**，696 个带标签模型（候选宇宙 12,000）。
我方 top-10 带标签 10/10；ModelLens top-10 带标签 1/10。

**我方 L1L3b（MIPS，held-out）：**

| rank | model | acc | MIPS |
|---|---|---|---|
| 1 | `barnowak/gte-qwen2-7b-instruct-q4-k-m-gguf` | **0.643** | 0.753 |
| 2 | `sparkhonyuk/gte-qwen2-7b-instruct-gguf` | **0.643** | 0.742 |
| 3 | `michieleeckhout/gte-qwen2-7b-instruct-q4-k-m-gguf` | **0.643** | 0.739 |
| 4 | `cleatherbury/gte-qwen2-7b-instruct-q5-k-m-gguf` | **0.643** | 0.736 |
| 5 | `bnightning/gte-qwen2-7b-instruct-q4-k-m-gguf` | **0.643** | 0.735 |
| 6 | `niancheng/gte-qwen2-7b-instruct-q4-k-m-gguf` | **0.643** | 0.732 |
| 7 | `gme-qwen2-vl-7b-instruct` | **0.646** | 0.731 |
| 8 | `lxc1999/gte-qwen2-7b-instruct-q4-k-m-gguf` | **0.643** | 0.728 |
| 9 | `datouge/gte-qwen2-7b-instruct-q4-0-gguf` | **0.643** | 0.727 |
| 10 | `sunzx0810/gte-qwen2-7b-instruct-q5-k-m-gguf` | **0.643** | 0.724 |

**ModelLens（blind release 版）：**

| rank | model | acc | score |
|---|---|---|---|
| 1 | `rifre` | — | 24.99 |
| 2 | `dqubit/frida-f16` | — | 23.27 |
| 3 | `169pi/alpie-core` | — | 21.70 |
| 4 | `r@m` | — | 20.92 |
| 5 | `samerzaher80/aethermind-srl` | — | 20.84 |
| 6 | `leocristt/hackathon-embedding-model` | — | 19.16 |
| 7 | `zomba/fsg-net` | — | 19.05 |
| 8 | `blevlabs/stella-en-v5` | **0.653** | 19.02 |
| 9 | `viktorzver/frida` | — | 17.96 |
| 10 | `zabir735/outputs` | — | 17.54 |

