# T2 v2 distribution report

Generated 2026-08-03T16:02:13+00:00. A = v1 downloads-desc head shard, B = v2 candidate pool, C = v2 selected HALO.

## Concentration

| metric | A: v1 150K | B: pool | C: selected |
|---|---|---|---|
| models | 150000 | 537025 | 69817 |
| author distinct | 25500 | 120767 | 30987 |
| author HHI | 0.06620 | 0.00681 | 0.00035 |
| author effective count | 15.1 | 146.7 | 2838.3 |
| author top-1 share | 0.2421 | 0.0740 | 0.0090 |
| author top-10 share | 0.4042 | 0.1608 | 0.0406 |
| author top-100 share | 0.5340 | 0.2679 | 0.1158 |
| family distinct | 84590 | 296567 | 45466 |
| family HHI | 0.03219 | 0.01337 | 0.00031 |
| family effective count | 31.1 | 74.8 | 3226.0 |
| family top-1 share | 0.1790 | 0.1127 | 0.0050 |
| family top-10 share | 0.2078 | 0.1719 | 0.0370 |
| family top-100 share | 0.2581 | 0.2678 | 0.1295 |
| largest author | mradermacher | mradermacher | WindstormLabs |

## supertask

| bucket | A share | B share | C share |
|---|---|---|---|
| text-generation-chat | 0.5077 | 0.2592 | 0.1800 |
| text-classification | 0.0245 | 0.0811 | 0.1147 |
| embedding-retrieval | 0.0819 | 0.0631 | 0.0770 |
| image-generation | 0.0455 | 0.0686 | 0.0740 |
| tabular-timeseries-rl | 0.0073 | 0.0799 | 0.0703 |
| vision-language | 0.0583 | 0.0582 | 0.0673 |
| image-classification | 0.0193 | 0.0455 | 0.0632 |
| translation | 0.0061 | 0.0335 | 0.0580 |
| speech-recognition | 0.0161 | 0.0358 | 0.0559 |
| token-classification | 0.0104 | 0.0339 | 0.0478 |
| audio-generation | 0.0153 | 0.0215 | 0.0406 |
| detection-segmentation | 0.0081 | 0.0195 | 0.0399 |
| question-answering | 0.0022 | 0.0258 | 0.0268 |
| audio-classification | 0.0024 | 0.0089 | 0.0235 |
| summarization | 0.0013 | 0.0060 | 0.0202 |
| unknown | 0.1693 | 0.1485 | 0.0200 |
| code | 0.0236 | 0.0096 | 0.0169 |
| other | 0.0006 | 0.0015 | 0.0040 |
| _entropy (bits)_ | 2.506 | 3.508 | 3.828 |
| _normalized entropy_ | 0.601 | 0.841 | 0.918 |

## language bucket

| bucket | A share | B share | C share |
|---|---|---|---|
| english-primary | 0.3479 | 0.1749 | 0.3955 |
| multilingual-with-english | 0.1241 | 0.0892 | 0.2345 |
| non-english | 0.0584 | 0.0859 | 0.1848 |
| language-neutral | 0.0608 | 0.1945 | 0.1600 |
| unknown | 0.4088 | 0.4555 | 0.0252 |
| _entropy (bits)_ | 1.916 | 2.031 | 2.027 |
| _normalized entropy_ | 0.825 | 0.875 | 0.873 |

## source type

| bucket | A share | B share | C share |
|---|---|---|---|
| original-base | 0.2175 | 0.4415 | 0.5427 |
| community-finetune | 0.1177 | 0.1852 | 0.1827 |
| quantized-conversion | 0.6198 | 0.2776 | 0.1500 |
| adapter-lora | 0.0256 | 0.0648 | 0.0500 |
| official-derivative | 0.0194 | 0.0189 | 0.0489 |
| official-quantized | -- | 0.0120 | 0.0257 |
| _entropy (bits)_ | 1.516 | 1.925 | 1.902 |
| _normalized entropy_ | 0.653 | 0.745 | 0.736 |

## family size stratum

| bucket | A share | B share | C share |
|---|---|---|---|
| singleton | 0.4701 | 0.5000 | 0.5016 |
| small | 0.2252 | 0.1301 | 0.2136 |
| large | 0.2523 | 0.3220 | 0.1918 |
| medium | 0.0524 | 0.0479 | 0.0930 |
| _entropy (bits)_ | 1.720 | 1.619 | 1.751 |
| _normalized entropy_ | 0.860 | 0.810 | 0.875 |

## popularity stratum

| bucket | A share | B share | C share |
|---|---|---|---|
| recent | 0.3061 | 0.3507 | 0.3357 |
| head | 0.1735 | 0.1623 | 0.3210 |
| mid | 0.2775 | 0.2598 | 0.2320 |
| long-tail | 0.2429 | 0.2271 | 0.1113 |
| _entropy (bits)_ | 1.970 | 1.947 | 1.897 |
| _normalized entropy_ | 0.985 | 0.973 | 0.948 |

## quantization type

| bucket | A share | B share | C share |
|---|---|---|---|
| none | 0.3654 | 0.7140 | 0.8169 |
| gguf | 0.5288 | 0.1695 | 0.0604 |
| onnx | 0.0205 | 0.0431 | 0.0404 |
| openvino | 0.0149 | 0.0249 | 0.0247 |
| mlx | 0.0406 | 0.0247 | 0.0236 |
| fp8 | 0.0132 | 0.0073 | 0.0135 |
| bitsandbytes | 0.0053 | 0.0073 | 0.0094 |
| awq | 0.0069 | 0.0031 | 0.0047 |
| gptq | 0.0041 | 0.0033 | 0.0040 |
| exl2 | 0.0000 | 0.0023 | 0.0017 |
| tensorrt | 0.0001 | 0.0003 | 0.0005 |
| torchao | 0.0001 | 0.0001 | 0.0003 |
| _entropy (bits)_ | 1.618 | 1.424 | 1.169 |
| _normalized entropy_ | 0.451 | 0.397 | 0.326 |

## Top 10 authors

| rank | A: v1 150K | B: pool | C: selected |
|---|---|---|---|
| 1 | mradermacher 24.21% | mradermacher 7.40% | WindstormLabs 0.90% |
| 2 | RichardErkhov 7.97% | RichardErkhov 1.97% | mradermacher 0.50% |
| 3 | mlx-community 1.22% | Muapi 1.76% | OpenMed 0.50% |
| 4 | bartowski 1.19% | alphaedge-ai 1.31% | Helsinki-NLP 0.41% |
| 5 | featherless-ai-quants 1.19% | mlx-community 1.00% | mlx-community 0.35% |
| 6 | timm 1.07% | ProbeX 0.75% | facebook 0.31% |
| 7 | alphaedge-ai 1.02% | WindstormLabs 0.60% | WindyWord 0.30% |
| 8 | facebook 1.01% | bartowski 0.45% | ILKT 0.27% |
| 9 | TheBloke 0.85% | TheBloke 0.43% | KoichiYasuoka 0.27% |
| 10 | unsloth 0.71% | OpenMed 0.42% | prithivMLmods 0.26% |

## Quality / lineage

| metric | A | B | C |
|---|---|---|---|
| unresolved family rate | 0.4642 | 0.4929 | 0.4905 |
| mirror-publisher share | 0.4336 | 0.1494 | 0.0261 |
| safetensors size known | 0.3500 | 0.4010 | 0.5775 |
| has model-index | 0.0379 | 0.0757 | 0.0948 |
| metadata quality mean | 0.3625 | 0.5564 | 0.7185 |
| metadata quality pass rate | 0.7045 | 0.8562 | 1.0000 |
