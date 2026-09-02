# ModelLakeFishing 1M/X evidence source manifest

Compilation cutoff: **2026-09-01 (America/Toronto)**. Repository root:
`D:\research\model_lake\codes\ModelLakeFishing`.

This manifest is the provenance companion to the Chinese and English evidence-source reports. A hash below identifies the exact local file inspected while compiling those reports; it does not imply that the file is committed to Git.

## Repository state and verification boundary

- Local `HEAD`: `97320b65df3ff9ea436a7525db66fd90af585174` (`F1: update 1M plan with completed crawl artifacts`, 2026-08-18).
- The inspected worktree is dirty and contains the later F/X implementation and evidence files. Therefore `HEAD` alone is **not** a reproducible identifier for the completed pipeline.
- All **92** JSON files under `docs/1M/*_runs/` parsed successfully with PowerShell `ConvertFrom-Json` at compilation time.
- The **28** principal Python modules listed in the code table parsed successfully with Python `ast.parse`.
- Fresh suite-isolated pytest runs used the repository `.venv` (Python 3.13.1, pytest 9.1.1): `scale1m/tests` **262 passed**, `stage2TrainGraphSAGE/tests` **67 passed**, `stage1BuildTransferGraph/tests` **8 passed**, and `stage3HNSW/tests` **7 passed**, for **344 passed** in total. The first two suites reproduce X5's archived **329 passed** exactly. A single pytest invocation from the repository root is not a valid entry point: it raises 18 collection errors because multiple top-level directories expose packages named `tests`. The stage1 suite additionally requires the parent of the repository on `PYTHONPATH`. These are collection/import conditions, not failed test assertions.
- The X4 as-run hashes for `losses.py`, `ablation.py`, `train_rung.py`, and `train_rung_x4.sbatch` match the present local files. F6 is less cleanly bound because its run metadata records an old Git head plus tracked and untracked changes; see `F6.md` and the reports' limitations section.

## Writing specification

| SHA-256 | Source |
|---|---|
| `8185c9f0a29988551ad38b7ed8156f5b82f7c6fd742f314bc8140eeb76301729` | `D:\research\model_lake\weeks\tech_reports_writing_skills.md` |

## Primary requested documents

| SHA-256 | Source |
|---|---|
| `067bd113968ec82f0e96a26094ca02ead9203cde0b8993b85716d7de87e66a6c` | [`1Mplan.md`](1Mplan.md) |
| `7af9f8da75bc787d66b6f573d63f0d19b3e3772b68ac7b8a83941d63f8bf5179` | [`F0.md`](F0.md) |
| `b945366b7752b861fdd4a3c80960301774535f9bf0d034e43c6ff3c5e640ed92` | [`F1.md`](F1.md) |
| `a1bb91deb141a65a3fee6a8028b3ea85935ad17a9c73f6a55a3c9c5d890f4aef` | [`F1.5.md`](F1.5.md) |
| `16d5a2835290c4f6625b99155730222841318f3a0cf372a9d6c53a519595464d` | [`F2.md`](F2.md) |
| `07a5e4185123ccee60a5dfd95ec9e0284229f63e5ffd4b0ff089e72e7a6e374d` | [`F3.md`](F3.md) |
| `6b9341fa7d82b0c9bfa2ed20e8a04fbb53f8c7346c8cdf360ca6091b4d46792b` | [`F4.md`](F4.md) |
| `670819aa3bde7850da00145fe4853b1dc353a9e6886d915fc47d63c69ef6c4b2` | [`F5.md`](F5.md) |
| `89248798a73684803076b420cbd71b075aa535624e2987afae7b0a67179c549f` | [`F6.md`](F6.md) |
| `6ae44a2750b0c0263672d5e3f0721219240f2738206ca7daa755909fa23d5c09` | [`F7.md`](F7.md) |
| `83d86c219b582bf805f16f73cae76779363f430d8140860a4f8e28872b11af79` | [`F8.md`](F8.md) |
| `bd78a21bbad39536a4ae7dc543b9230c903b5bdb8f897502c6dbec0f46df2ae4` | [`F9.md`](F9.md) |
| `a2fd8d4543cf51ea468826c87f3c2951e36ea4b558c0d2e9a741b6997aadbb0a` | [`X1.md`](X1.md) |
| `b6874780685d942413446e780b3abf558f0fb69e569c26a8a6230ec5ccd51728` | [`X2.md`](X2.md) |
| `d0d3d95b33898754330e5c1ef7713217cbc32ca58e12e1c03b0c08a6cce4d5e1` | [`X3.md`](X3.md) |
| `702ab760fea5a1e7d2ad7a2ec7bc0f1a23d8192802ade988b6be11872b48b220` | [`X4.md`](X4.md) |
| `c8ca8d37dc04b1318bf822fb1251647f761e2a8ceca0d7ed7c39d30bf9124158` | [`X5.md`](X5.md) |

## Execution guides used as supporting evidence

| SHA-256 | Source |
|---|---|
| `7ffaa3ef364fae6ad54e86c4693f63a5c7a66537d1c9b748c36297a3eea0c515` | [`F6GPU.md`](F6GPU.md) |
| `586fea6458a6fa3b680dfa532d59c9bca40959d61b11dab0ae9c8fab343964d2` | [`X4GPU.md`](X4GPU.md) |
| `d0d9331410f22fa0a33160d3d71c27c0ef1906306f66dc4e96c3ba5c27c1e8c7` | [`F6.5guide.md`](F6.5guide.md) |
| `b62c30b7d44618a6b878b79ab285e0b3a11b0fa4f76a2129a2f15028db29dda6` | [`F7.5guide.md`](F7.5guide.md) |

## Principal measurement artifacts

| SHA-256 | Artifact |
|---|---|
| `32b0dcdb5a00f88f67b43fb2550939db6ed557b0f9f1f206d73f2c3ea4ad10a1` | [`F0_runs/f0_probe.json`](F0_runs/f0_probe.json) |
| `92be94634052c0994ceb92da18c9826ba897c39e0ba552803c9df9bd3e9bb02d` | [`F0_runs/f0_size_estimate.json`](F0_runs/f0_size_estimate.json) |
| `f1905fe8f3e036dd6acd08060a5c5a5d06f25fa65d542e8455fa6d040559793d` | [`F1_runs/PROVENANCE.json`](F1_runs/PROVENANCE.json) |
| `c22c63ab5d7a3f3bf192623dad4446ddc1fd5bf616f82d8688de4310b4bb34e5` | [`F15_runs/PROVENANCE.json`](F15_runs/PROVENANCE.json) |
| `1edae3c611f7ac81f02ac18906e53f5315124fd214a1418d37ba7b1b686dd708` | [`F15_runs/f15_policy.json`](F15_runs/f15_policy.json) |
| `bbb6ad02c945b1e6fe25d7439322bf2f4b44c955fa082c1e2748c8514756989e` | [`F15_runs/f15b_coverage.json`](F15_runs/f15b_coverage.json) |
| `6cb841550fa81de145f402938021bd6d5b8c6f7927cfb5f99ccaf278e68a181b` | [`F2_runs/F2_REPORT.json`](F2_runs/F2_REPORT.json) |
| `1d27d672ff3470cd71410b09cd077c929d75f01b03bc9abdcc48b9fa85dd0d67` | [`F2_runs/F2_MERGE_REPORT.json`](F2_runs/F2_MERGE_REPORT.json) |
| `be3fb05ecf92ea0112ba4de10b8191d71dfe096b8667a86911c2ebaeadb9cdb1` | [`F2_runs/rf_gold_rules.json`](F2_runs/rf_gold_rules.json), also the frozen rule-file hash |
| `14dcbdc32e8317e5f66deb893d9cc4c9e9ece691c6f4260978044dc861bd4307` | [`F3_runs/LADDER_REPORT.json`](F3_runs/LADDER_REPORT.json) |
| `dbafb576203a7dec589c608c81b66d1cb4df4a27e8658c870d548e69f8dc7601` | [`F4_runs/FEATS_REPORT.json`](F4_runs/FEATS_REPORT.json) |
| `53081ab8fe339c4aedb22c37f4ac6a8b19acc710e2e5915aaab0577bb5074527` | [`F5_runs/GRAPH_REPORT.json`](F5_runs/GRAPH_REPORT.json) |
| `8a0d3ea913723693c647fffdad032708510378a9481ef3a23aa3bc1ec3eb75bb` | [`F5_runs/lineage_stats.json`](F5_runs/lineage_stats.json) |
| `d4344771c384b15c8aee9e23af5c02d24cd36b51dd58df628d66ef5974efb56f` | [`F5_runs/f5_splits.json`](F5_runs/f5_splits.json) |
| `fbdd3f809479e2e11d12721437f510fa328a3175d8042198377b81f36d4b8b7c` | [`F8_runs/F8_REPORT.json`](F8_runs/F8_REPORT.json) |
| `599e53d88013f83547bdb4497994bb64bc25183603f2555723a5d0891b1d6345` | [`F9_runs/F9_SCORECARD.json`](F9_runs/F9_SCORECARD.json) |
| `f69f2cdfcb301f3acffd16fd0953b1189a72d00f96566067b9ad3b41d6eefe58` | [`X1_runs/X1_FAST_LEVERS.json`](X1_runs/X1_FAST_LEVERS.json) |
| `675290e9167fc1581f08d2c76ec7e9e50867a35d94c1bbbde0cb8c65d7799f95` | [`X2_runs/X2_PRIOR_FUSION.json`](X2_runs/X2_PRIOR_FUSION.json) |
| `4ee147206cda041826ebb6bccc80cffb8976cbc61d5b986b19a872b7ec2da613` | [`X3_runs/PREREGISTRATION.md`](X3_runs/PREREGISTRATION.md), hash embedded in the X3 result |
| `1924de3055fb9732c39650ac00c42ad8a921c4bc40c83be86cf1cc3cb2df182b` | [`X3_runs/X3_PROTOCOL_B.json`](X3_runs/X3_PROTOCOL_B.json) |
| `0771805b6b18dbdf6d501fb7ee122d43b21b6cde1b977c71d0f9de250fb52ff1` | [`X4_runs/reports/X4_GD_REPORT.json`](X4_runs/reports/X4_GD_REPORT.json) |
| `0bbf263cbfd152dfaa542a1404668b2993749bad04cfebb6be5176e437717ea0` | [`X4_runs/reports/X4_G_REPORT.json`](X4_runs/reports/X4_G_REPORT.json) |
| `55596d133899bddebf72b026dcf40de16a8e5d695fd81e50b62fbbef5479103a` | [`X4_runs/reports/X4_D_REPORT.json`](X4_runs/reports/X4_D_REPORT.json) |
| `8b15bada3cbb96b87ce86358ae417b1d72dc851936414f38369a0ec86c10a12f` | [`X5_runs/X5_F6_ELIGIBILITY.json`](X5_runs/X5_F6_ELIGIBILITY.json) |
| `84e550fa8646025a8558f3079ded44b256d96ece0d4d5171978ab6cf460aa942` | [`X5_runs/X5_GD_ELIGIBILITY.json`](X5_runs/X5_GD_ELIGIBILITY.json) |
| `913190a94ebe3d9feab400a681dcf94868a9158e4f77a929770e8d18313be23c` | [`X5_runs/X5_G_ELIGIBILITY.json`](X5_runs/X5_G_ELIGIBILITY.json) |
| `961713591f7fdc3e19e57c38fa5413761cea3d555925e37bac8f6d39fdff40c6` | [`X5_runs/X5_D_ELIGIBILITY.json`](X5_runs/X5_D_ELIGIBILITY.json) |
| `5ebc6b6edb43a1eaeed7324b3b812bf257cf6b7da810ccd44e9b3f0671ad56a3` | [`X4_runs/x4_execution/code_sha256.txt`](X4_runs/x4_execution/code_sha256.txt) |
| `0c3b490a88326b140f7922f315b3a01d8bc486dd750df33a3d14850bb659769d` | [`X4_runs/x4_execution/sacct.txt`](X4_runs/x4_execution/sacct.txt) |

## Current principal code snapshot

| SHA-256 | Code |
|---|---|
| `631e53337a5b51f76d0d7d8c7969abd9b9fb331e7117b0e9dc5925849016879e` | [`scale1m/hf_crawl.py`](../../scale1m/hf_crawl.py) |
| `9f48194aa4a38f749649489ffa06a201221ad3053dae47a282b5f23bf05133bf` | [`scale1m/hf_crawl_datasets.py`](../../scale1m/hf_crawl_datasets.py) |
| `03b9976c87d36ceff30eb18dcf0bf5f1bc826f7c5c1ffe527a953f622bc16190` | [`scale1m/match_dataset_cards.py`](../../scale1m/match_dataset_cards.py) |
| `4d0f9c7da2c6bf29ee7683794bf81e5d107c7323a04806438314551c0a328da5` | [`scale1m/canonicalize_rf.py`](../../scale1m/canonicalize_rf.py) |
| `04b2a7f5c4dc331ef70dbe243facf67d54af58ecd1491873c13008928d2d7316` | [`scale1m/metric_semantics.py`](../../scale1m/metric_semantics.py) |
| `72b2495fd5d27fb87d87b39ef232b6277c368bb4bc2032568459dc9821a75437` | [`scale1m/merge_supervision.py`](../../scale1m/merge_supervision.py) |
| `976c936b0d64b2fcd6b09854d5036e0865ed0a03fc005049e2be02537e33c32c` | [`scale1m/build_ladder_rf.py`](../../scale1m/build_ladder_rf.py) |
| `5a7dd784c1167c86572f916a5eaea651a7bb422231c42cc86d9a6eaa37a07032` | [`scale1m/embed_lake_rf.py`](../../scale1m/embed_lake_rf.py) |
| `704d5c1f53a66fa04b1502f8ab6c08705ebad64f0f7d2ed1526589be4edf31ec` | [`scale1m/build_graph_rf.py`](../../scale1m/build_graph_rf.py) |
| `488c472249b4060acf73866a8b3d29c9bb9d09bb2d7cc85a18d919cf2197e037` | [`scale1m/graph_store.py`](../../scale1m/graph_store.py) |
| `08904b015de386188b452b3a99f3b598207595460d605dd9452b6f4d05743a46` | [`scale1m/train_rung.py`](../../scale1m/train_rung.py), matches X4 remote hash |
| `a24a4505381f071c1162ae314c3b2e27cc5fb38a8839db998be6a1193454e20d` | [`scale1m/checkpoint.py`](../../scale1m/checkpoint.py) |
| `935b3d9ef30c64ccb4f33530bd540b2795662cab7f19fd63035a8b0f6bdaff65` | [`scale1m/export_rf.py`](../../scale1m/export_rf.py) |
| `6c61c24d6bb4c9883c5216c5137d11e7b602a5df6cb85e553f60788d24c1e8b3` | [`scale1m/eval_rf.py`](../../scale1m/eval_rf.py), includes post-X5 path/default hardening |
| `41069d9e1b203600eb227e7c87cb152d4b3a2392b8dd5580ba4cbc3372596f36` | [`scale1m/fast_lever_audit.py`](../../scale1m/fast_lever_audit.py) |
| `252f62b3152d8ca1bfd00e7aa3411d4898748d2338cf272f86545e5a2d0f573b` | [`scale1m/utility_scorecard.py`](../../scale1m/utility_scorecard.py) |
| `e0fa475bbdf2cc1040c9643b37a28d18d54f7cf08fd6b73f1af7650552a9191d` | [`stage2TrainGraphSAGE/ablation.py`](../../stage2TrainGraphSAGE/ablation.py), matches X4 remote hash |
| `c9b35119ac64471352dbc75da372a3b065d60f8b630b3954ed45c94711392b9c` | [`stage2TrainGraphSAGE/losses.py`](../../stage2TrainGraphSAGE/losses.py), matches X4 remote hash |
| `3cacbdd31f349200cbfcc654eb1824d4b01c9a22f615291525b4189686cd60db` | [`stage2TrainGraphSAGE/model.py`](../../stage2TrainGraphSAGE/model.py) |
| `b69a38f0dfc9eb00bbeac9729592be7ec02b58c5af244528dd563ffd2b7e123e` | [`stage2TrainGraphSAGE/sampling.py`](../../stage2TrainGraphSAGE/sampling.py) |
| `d43ce6f2b16d29242fc6548da133f8a73715eca83f60b5fc97e09ba16d42bc32` | [`stage2TrainGraphSAGE/train.py`](../../stage2TrainGraphSAGE/train.py) |
| `1bae594e39748d6542bd9e59942a5d1d4a0bb3c3892cfe84376d81a37526b856` | [`stage3HNSW/build_prior_sidecar.py`](../../stage3HNSW/build_prior_sidecar.py) |
| `30cd7a022cd94f0cf46f73e18e356b3c7e4542e3c7307b8eebde8e87bb04a167` | [`scale/modellens_build_graph.py`](../../scale/modellens_build_graph.py) |
| `deecddbc117970e9f62f41f3d7cb5c1c33dc3786a1e7873d3d75f3cb48606172` | [`scale/global_metrics.py`](../../scale/global_metrics.py) |
| `acb5c7e86ecc29f639e35b34e7a8b5b57493e42f11f36cf4a2f54f71bce2191d` | [`scale/export_ours.py`](../../scale/export_ours.py) |
| `de2a897296f2f069b98be772158202cde25652dcdbe667ad75f59640685bacfb` | [`stage1BuildTransferGraph/d0_build_graph.py`](../../stage1BuildTransferGraph/d0_build_graph.py) |
| `5ddb3ff9c7b0606585840f94c4a3e99ba09c98aa0257d8af249ed8d337bc7326` | [`stage1BuildTransferGraph/dataset_embed/model_node_encoder.py`](../../stage1BuildTransferGraph/dataset_embed/model_node_encoder.py) |
| `5bdf7239c29c2bfe2208fbf44124dcbab9ba3c2c1ce5b0e1e873273798f4710b` | [`stage2TrainGraphSAGE/d0_splits.py`](../../stage2TrainGraphSAGE/d0_splits.py) |
| `2f9c9bd36e39896fa1bd46e36cd10b096cfbfba039b6047a269999f05d225265` | [`scripts/watgpu/train_rung_rf.sbatch`](../../scripts/watgpu/train_rung_rf.sbatch) |
| `f1237a73f7ea6ccccc73fe68897297b51cfe69d2ce6bfb4d5d5af02cd4b67e9b` | [`scripts/watgpu/train_rung_x4.sbatch`](../../scripts/watgpu/train_rung_x4.sbatch), matches X4 formal-run hash |

## Large artifacts identified by embedded bindings

The large local/remote binaries are not duplicated in `docs/1M`. Their identities are carried by the reports and manifests:

| Object | Bound identity |
|---|---|
| Frozen model ladder | `full_model_ids.parquet` SHA-256 `fee360d1fef9afa4c446b70af35bcbf4ead2c75387f47abc3b0d21c944ae3efa` |
| Frozen dataset ladder | `full_dataset_ids.parquet` SHA-256 `31c027ff2eeb693aed6abaa5d235f4a6264bb4b3245bfd71cf1c8b85d32173cb` |
| Model feature matrix | `x_m.npy` SHA-256 `ba1020872ddb6e90726c755f9f08f236f4c73e506013c91d8a0075fc0c0361e6` |
| Family vocabulary | SHA-256 `00d304df9bd72acb2c2c1ca5b84ed0dc9d4adb0cbc9f098f32a9e62109d24005` |
| Sharded RF graph | `graph_digest` `0e80b8393846dcc4b9e354218fd5139a906d569125d2e8545e17ccd01612b76c` |
| Frozen gold rules | `rf-gold-2.0`, SHA-256 `be3fb05ecf92ea0112ba4de10b8191d71dfe096b8667a86911c2ebaeadb9cdb1` |
