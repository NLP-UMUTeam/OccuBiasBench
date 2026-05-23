# Evaluation

This folder contains the scripts used to compute the evaluation metrics reported in the paper from the raw model outputs in `../results/`.

The scripts are intentionally kept focused on **machine-readable outputs**. They generate `.csv` and `.json` files only.
## Scripts

| Script                     | Purpose                                                      | Main outputs                                                 |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `compute_bias_metrics.py`  | Computes final-answer metrics for explicit and indirect earning-comparison prompts. | `normalized_predictions.csv`, `overall_bias_summary.csv`, `mspb_summary.csv`, `mspb_implicit_explicit_summary.csv`, `paired_bootstrap_deltas.csv` |
| `compute_reasoning_ras.py` | Computes Reasoning Attribute Score Bias (RAS) from structured reasoning outputs. | `reasoning_score_rows.csv`, `reasoning_ras_summary.csv`, `reasoning_parse_errors.csv`, `reasoning_ras_diagnostics.json` |

## Expected input layout

The scripts expect the following repository layout:

```text
repo/
├── dataset/
│   └── selected_images.txt
├── results/
│   ├── explicit/
│   │   ├── predictions_explicit_prompt_gemma3_12b_it.jsonl
│   │   ├── predictions_explicit_prompt_gemma3_27b_it.jsonl
│   │   ├── predictions_explicit_prompt_qwen3_vl_4b.jsonl
│   │   ├── predictions_explicit_prompt_qwen3_vl_8b.jsonl
│   │   └── predictions_explicit_prompt_qwen_3_vl_32b.jsonl
│   ├── indirect/
│   │   ├── predictions_indirect_prompt_gemma3_12b_it.jsonl
│   │   ├── predictions_indirect_prompt_gemma3_27b_it.jsonl
│   │   ├── predictions_indirect_prompt_qwen3_vl_4b.jsonl
│   │   ├── predictions_indirect_prompt_qwen3_vl_8b.jsonl
│   │   └── predictions_indirect_prompt_qwen_3_vl_32b.jsonl
│   └── reasoning/
│       ├── spatial/
│       │   ├── predictions_reasoning_spatial_thinking_gemma3_12b_it.jsonl
│       │   ├── predictions_reasoning_spatial_thinking_gemma3_27b_it.jsonl
│       │   ├── predictions_reasoning_spatial_qwen3_vl_4b_analysis.jsonl
│       │   ├── predictions_reasoning_spatial_qwen3_vl_8b_analysis.jsonl
│       │   └── predictions_reasoning_spatial_qwen_3_vl_32b_analysis.jsonl
│       └── gender/
│           ├ predictions_reasoning_gender_prompt_gemma3_12b_it_analysis.jsonl
│           ├ predictions_reasoning_gender_prompt_gemma3_27b_it_analysis.jsonl
│           ├ predictions_reasoning_gender_qwen3_vl_4b_analysis.jsonl
│           ├ predictions_reasoning_gender_qwen3_vl_8b_analysis.jsonl
│           └ predictions_reasoning_gender_prompt_qwen_3_vl_32_analysis.jsonl
└── evaluation/
    ├── compute_bias_metrics.py
    ├── compute_reasoning_ras.py
    └── README.md
```

By default, both scripts use:

```text
--results-dir ../results
--exclude-file ../dataset/selected_images.txt
```

## Human-validation filtering

The reported paper results exclude images that failed human validation. These filenames are stored in:

```text
../dataset/selected_images.txt
```

Both scripts support filtering with:

```bash
--exclude-file ../dataset/selected_images.txt
```

This file contains image filenames to remove from the analysis. The scripts compare the filename from each prediction record against this exclusion list.

## Final-answer bias metrics

Run:

```bash
python compute_bias_metrics.py \
  --results-dir ../results \
  --exclude-file ../dataset/selected_images.txt \
  --n-bootstrap 10000 \
  --seed 42 \
  --out-dir outputs/bias_metrics
```

This script reads the prediction files in:

```text
../results/explicit/
../results/indirect/
```

and computes:

- **Male Preference (MP)** over determined `man`/`woman` responses.
- **Abstention Rate (AR)** using `cannot be determined` responses.
- **Male Selection Position Bias (MSPB)** for position-sensitive male selections.
- **Paired bootstrap deltas** between explicit and indirect settings.

### Outputs

```text
outputs/bias_metrics/
├── normalized_predictions.csv
├── overall_bias_summary.csv
├── mspb_summary.csv
├── mspb_implicit_explicit_summary.csv
└── paired_bootstrap_deltas.csv
```

## Reasoning Attribute Score Bias

Run:

```bash
python compute_reasoning_ras.py \
  --results-dir ../results \
  --exclude-file ../dataset/selected_images.txt \
  --out-dir outputs/reasoning_ras
```

This script reads the structured reasoning outputs in:

```text
../results/reasoning/spatial/
../results/reasoning/gender/
```

and computes **Reasoning Attribute Score Bias (RAS)** for six attributes:

- competence
- perceived authority
- seniority
- warmth
- promotion likelihood
- perceived earning potential

RAS is computed from the numeric 1--5 scores assigned to the man and the woman in each image.