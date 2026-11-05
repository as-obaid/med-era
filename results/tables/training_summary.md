# Training summary

Per adapter: optimizer steps, training loss (mean NLL per token of the batch) at the start, at the end of epoch 1 and at the end, the adapter's held-out NLL on its own era (T3), attempts with the steps each covered, GPU type, peak GPU memory, GPU-hours and the training commit. All columns in `training_summary.csv`.

| adapter | steps | epochs | loss, first 10 steps | loss, last 10 steps of epoch 1 | loss, last 10 steps | held-out NLL, own era | attempts (steps) | GPU | peak GB | GPU-h | commit |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|
| qwen_w1 | 196 | 2 | 1.602 | 1.640 | 1.516 | 1.482 | 2 (0-103, 103-196) | H100 | 63.3 | 1.34 | 242f148 |
| qwen_w2 | 196 | 2 | 1.622 | 1.596 | 1.486 | 1.528 | 1 (0-196) | H200 | 63.3 | 0.75 | 242f148 |
| qwen_w3 | 196 | 2 | 1.578 | 1.510 | 1.415 | 1.519 | 1 (0-196) | H100 | 63.3 | 0.60 | 242f148 |
| gemma_al_w1 | 189 | 2 | 1.640 | 1.648 | 1.629 | 1.523 | 1 (0-189) | H200 | 59.9 | 0.63 | 242f148 |
| gemma_al_w2 | 189 | 2 | 1.729 | 1.681 | 1.583 | 1.562 | 1 (0-189) | H100 | 59.9 | 0.61 | 242f148 |
| gemma_al_w3 | 189 | 2 | 1.597 | 1.613 | 1.555 | 1.576 | 1 (0-189) | H100 | 59.9 | 0.51 | 242f148 |
| gemma_ex_w1 | 189 | 2 | 1.639 | 1.640 | 1.552 | 1.526 | 2 (0-140, 140-189) | H100/H200 | 67.6 | 0.96 | 242f148 |
| gemma_ex_w2 | 189 | 2 | 1.728 | 1.673 | 1.503 | 1.568 | 2 (0-181, 181-189) | H100 | 67.6 | 0.82 | 242f148 |
| gemma_ex_w3 | 189 | 2 | 1.596 | 1.605 | 1.475 | 1.583 | 1 (0-189) | H200 | 67.6 | 0.70 | 242f148 |
| **total** | 1,722 | | | | | | 12 | H100/H200 | 67.6 (max) | **6.93** | |

- **GPU-hours** = the wall time of every optimizer step across all attempts (5.38 h) + the model load of every attempt (1.55 h): process start to base model loaded, from the run summary for the last attempt and from each earlier attempt's job log. Not counted: interpreter start-up, LoRA setup, checkpoint saves, the time between the stop deadline and job exit, and queueing. The first step of each attempt includes warm-up and is counted.
- **Epoch 1** ends inside a step when the blocks do not split evenly into steps of micro-batch x grad-accum; 'last 10 steps of epoch 1' are the last 10 steps entirely within it (qwen: steps 89-98 of 196, gemma_al: steps 85-94 of 189, gemma_ex: steps 85-94 of 189).
- **Attempts**: a job that reaches its time limit saves a checkpoint and the next attempt resumes from that step; the step spans are contiguous, so no step is repeated or missing. **Peak GB** is the last attempt's peak allocated GPU memory. Training loss and held-out NLL compare within a model only (Qwen and Gemma tokenize differently).
