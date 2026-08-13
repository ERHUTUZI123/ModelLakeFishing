# T2 v2 distribution report

Generated 2026-08-03T02:46:16+00:00. A = v1 downloads-desc head shard, B = v2 candidate pool, C = v2 selected HALO.

## Concentration

| metric | A: v1 150K | B: pool | C: selected |
|---|---|---|---|
| models | 150000 | 43086 | 4000 |
| author distinct | 25500 | 14783 | 2407 |
| author HHI | 0.06620 | 0.00243 | 0.00101 |
| author effective count | 15.1 | 411.3 | 986.2 |
| author top-1 share | 0.2421 | 0.0299 | 0.0083 |
| author top-10 share | 0.4042 | 0.1175 | 0.0537 |
| author top-100 share | 0.5340 | 0.3008 | 0.2253 |
| family distinct | 84590 | 27806 | 2213 |
| family HHI | 0.03219 | 0.00048 | 0.00140 |
| family effective count | 31.1 | 2081.6 | 713.8 |
| family top-1 share | 0.1790 | 0.0064 | 0.0050 |
| family top-10 share | 0.2078 | 0.0445 | 0.0500 |
| family top-100 share | 0.2581 | 0.1693 | 0.3055 |
| largest author | mradermacher | mradermacher | Qwen |

## supertask

| bucket | A share | B share | C share |
|---|---|---|---|
| text-generation-chat | 0.5077 | 0.1597 | 0.1500 |
| vision-language | 0.0583 | 0.0942 | 0.1303 |
| image-generation | 0.0455 | 0.1081 | 0.0935 |
| embedding-retrieval | 0.0819 | 0.0621 | 0.0917 |
| text-classification | 0.0245 | 0.0702 | 0.0717 |
| tabular-timeseries-rl | 0.0073 | 0.0814 | 0.0550 |
| audio-generation | 0.0153 | 0.0436 | 0.0545 |
| detection-segmentation | 0.0081 | 0.0575 | 0.0475 |
| image-classification | 0.0193 | 0.0444 | 0.0423 |
| speech-recognition | 0.0161 | 0.0307 | 0.0420 |
| token-classification | 0.0104 | 0.0312 | 0.0360 |
| audio-classification | 0.0024 | 0.0187 | 0.0352 |
| summarization | 0.0013 | 0.0134 | 0.0328 |
| question-answering | 0.0022 | 0.0187 | 0.0328 |
| translation | 0.0061 | 0.0271 | 0.0315 |
| unknown | 0.1693 | 0.1115 | 0.0182 |
| other | 0.0006 | 0.0165 | 0.0175 |
| code | 0.0236 | 0.0109 | 0.0175 |
| _entropy (bits)_ | 2.506 | 3.808 | 3.888 |
| _normalized entropy_ | 0.601 | 0.913 | 0.932 |

## language bucket

| bucket | A share | B share | C share |
|---|---|---|---|
| english-primary | 0.3479 | 0.2254 | 0.4730 |
| multilingual-with-english | 0.1241 | 0.1194 | 0.2770 |
| non-english | 0.0584 | 0.0757 | 0.1235 |
| language-neutral | 0.0608 | 0.2251 | 0.1047 |
| unknown | 0.4088 | 0.3544 | 0.0217 |
| _entropy (bits)_ | 1.916 | 2.147 | 1.858 |
| _normalized entropy_ | 0.825 | 0.925 | 0.800 |

## source type

| bucket | A share | B share | C share |
|---|---|---|---|
| community-finetune | 0.1177 | 0.1444 | 0.3937 |
| original-base | 0.2175 | 0.5248 | 0.2585 |
| quantized-conversion | 0.6198 | 0.2539 | 0.1500 |
| official-derivative | 0.0194 | 0.0284 | 0.1230 |
| adapter-lora | 0.0256 | 0.0484 | 0.0747 |
| _entropy (bits)_ | 1.516 | 1.751 | 2.096 |
| _normalized entropy_ | 0.653 | 0.754 | 0.903 |

## family size stratum

| bucket | A share | B share | C share |
|---|---|---|---|
| small | 0.2252 | 0.1974 | 0.3130 |
| singleton | 0.4701 | 0.5683 | 0.2800 |
| medium | 0.0524 | 0.1056 | 0.2177 |
| large | 0.2523 | 0.1286 | 0.1893 |
| _entropy (bits)_ | 1.720 | 1.649 | 1.972 |
| _normalized entropy_ | 0.860 | 0.824 | 0.986 |

## popularity stratum

| bucket | A share | B share | C share |
|---|---|---|---|
| mid | 0.2775 | 0.2511 | 0.3395 |
| head | 0.1735 | 0.1573 | 0.3235 |
| recent | 0.3061 | 0.3716 | 0.2090 |
| long-tail | 0.2429 | 0.2200 | 0.1280 |
| _entropy (bits)_ | 1.970 | 1.932 | 1.907 |
| _normalized entropy_ | 0.985 | 0.966 | 0.954 |

## quantization type

| bucket | A share | B share | C share |
|---|---|---|---|
| none | 0.3654 | 0.7409 | 0.8430 |
| onnx | 0.0205 | 0.0558 | 0.0535 |
| gguf | 0.5288 | 0.1227 | 0.0333 |
| openvino | 0.0149 | 0.0284 | 0.0300 |
| fp8 | 0.0132 | 0.0156 | 0.0158 |
| mlx | 0.0406 | 0.0219 | 0.0075 |
| awq | 0.0069 | 0.0054 | 0.0073 |
| bitsandbytes | 0.0053 | 0.0054 | 0.0070 |
| gptq | 0.0041 | 0.0034 | 0.0027 |
| tensorrt | 0.0001 | 0.0003 | -- |
| exl2 | 0.0000 | 0.0001 | -- |
| torchao | 0.0001 | 0.0001 | -- |
| _entropy (bits)_ | 1.618 | 1.399 | 1.021 |
| _normalized entropy_ | 0.451 | 0.390 | 0.322 |

## Top 10 authors

| rank | A: v1 150K | B: pool | C: selected |
|---|---|---|---|
| 1 | mradermacher 24.21% | mradermacher 2.99% | Qwen 0.83% |
| 2 | RichardErkhov 7.97% | facebook 1.31% | OpenGVLab 0.65% |
| 3 | mlx-community 1.22% | timm 1.29% | zai-org 0.57% |
| 4 | bartowski 1.19% | OpenMed 1.19% | allenai 0.53% |
| 5 | featherless-ai-quants 1.19% | cstr 0.94% | mistralai 0.53% |
| 6 | timm 1.07% | unsloth 0.93% | OpenMOSS-Team 0.53% |
| 7 | alphaedge-ai 1.02% | nvidia 0.91% | microsoft 0.45% |
| 8 | facebook 1.01% | google 0.79% | nvidia 0.45% |
| 9 | TheBloke 0.85% | Qwen 0.71% | openai 0.43% |
| 10 | unsloth 0.71% | WindyTranslate 0.70% | LiquidAI 0.43% |

## Quality / lineage

| metric | A | B | C |
|---|---|---|---|
| unresolved family rate | 0.4642 | 0.5587 | 0.2682 |
| mirror-publisher share | 0.4336 | 0.0764 | 0.0030 |
| safetensors size known | 0.3500 | 0.4348 | 0.7598 |
| has model-index | 0.0379 | 0.0624 | 0.1207 |
| metadata quality mean | 0.3625 | 0.5836 | 0.8005 |
| metadata quality pass rate | 0.7045 | 0.8854 | 1.0000 |
