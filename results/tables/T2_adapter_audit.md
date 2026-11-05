# T2 Adapter audit: adapted tensors and LoRA parameters by group

Per adapter: LoRA tensors / trainable LoRA parameters in each audit group, all tensors with a nonzero gradient during training, and the audit gate. The gate fails if an expert arm (Gemma-EX) adapts no expert tensor or none of its expert tensors received a nonzero gradient, if a non-expert arm adapts experts, or if any vision tensor or anything outside the text-model groups is adapted (SCOPE §6). From `runs/results/adapter_audit/` (copies in `results/adapter_audit/`); all columns in `T2_adapter_audit.csv`.

| adapter | attention | dense mlp | router | experts | vision | other | LoRA tensors | LoRA params | % of base | nonzero-grad tensors | audit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qwen_w1 | 608 / 47.5M | 384 / 69.2M | 0 | 0 | 0 | 0 | 992 | 116,727,808 | 0.43% | 992 of 992 | PASS |
| qwen_w2 | 608 / 47.5M | 384 / 69.2M | 0 | 0 | 0 | 0 | 992 | 116,727,808 | 0.43% | 992 of 992 | PASS |
| qwen_w3 | 608 / 47.5M | 384 / 69.2M | 0 | 0 | 0 | 0 | 992 | 116,727,808 | 0.43% | 992 of 992 | PASS |
| gemma_al_w1 | 230 / 11.5M | 180 / 7.1M | 60 / 1.4M | 0 | 0 | 0 | 470 | 19,998,720 | 0.08% | 470 of 470 | PASS |
| gemma_al_w2 | 230 / 11.5M | 180 / 7.1M | 60 / 1.4M | 0 | 0 | 0 | 470 | 19,998,720 | 0.08% | 470 of 470 | PASS |
| gemma_al_w3 | 230 / 11.5M | 180 / 7.1M | 60 / 1.4M | 0 | 0 | 0 | 470 | 19,998,720 | 0.08% | 470 of 470 | PASS |
| gemma_ex_w1 | 230 / 11.5M | 180 / 7.1M | 60 / 1.4M | 120 / 475.8M | 0 | 0 | 590 | 495,790,080 | 1.92% | 590 of 590 | PASS |
| gemma_ex_w2 | 230 / 11.5M | 180 / 7.1M | 60 / 1.4M | 120 / 475.8M | 0 | 0 | 590 | 495,790,080 | 1.92% | 590 of 590 | PASS |
| gemma_ex_w3 | 230 / 11.5M | 180 / 7.1M | 60 / 1.4M | 120 / 475.8M | 0 | 0 | 590 | 495,790,080 | 1.92% | 590 of 590 | PASS |

Cells: LoRA tensors / LoRA parameters (M = million). `% of base` is LoRA parameters over the base model's parameters (vision tower included). The two Gemma arms share a base and differ only in the expert group: Gemma-EX also adapts the fused expert weights.
