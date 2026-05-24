# OccuBiasBench
This repository contains the code, prompts, image-generation pipeline, inference scripts, evaluation scripts, and model outputs used to study occupational gender bias in Vision-Language Models (VLMs).

OccuBiasBench introduces a controlled paired-image benchmark for evaluating whether VLMs associate occupational status and earning potential differently with men and women. Each image contains two individuals matched by occupation, age, ethnicity, and workplace context, while differing only in gender. The benchmark is designed to reduce visual and demographic confounds and to support controlled comparisons across indirect, explicit, and reasoning-based evaluation settings.


<p align="center">
  <img src="./figure/emnlp-bias.jpg" alt="">
  <br>
  <em>Overview of our controlled paired-image evaluation framework for studying occupational status judgments in vision-language models.</em>
</p>

## 📋 Overview

We evaluate VLMs under three complementary settings:

1. **Indirect spatial comparison**  
   The model answers using `left`, `right`, or `cannot be determined`. Gender is not included in the answer space.

2. **Explicit gender comparison**  
   The model answers using `man`, `woman`, or `cannot be determined`. Gender is explicitly included in the answer space.

3. **Structured reasoning**  
   The model provides attribute-level scores and comparisons for occupational attributes such as competence, perceived authority, seniority, warmth, promotion likelihood, and perceived earning potential.

The analysis reports:

- **Male Preference (MP)**
- **Abstention Rate (AR)**
- **Male Selection Rate (MSR)**
- **Male Selection Position Bias (MSPB)**
- **Reasoning Attribute Score Bias (RAS)**

## 🗂️ Repository structure

```text
.
├── README.md
├── requirements_generate_images.txt
├── requirements_inference.txt
├── dataset/
│   ├── README.md
│   ├── generate_image_isco_balanced_ethnic_age.py
│   └── selected_images.txt
├── prompts/
│   ├── README.md
│   ├── explicit_prompt.txt
│   ├── indirect_prompt.txt
│   ├── reasoning_spatial_prompt.txt
│   └── reasoning_gender_prompt.txt
├── inference/
│   ├── README.md
│   ├── VLMBase.py
│   └── zero-shot.py
├── results/
│   ├── README.md
│   ├── explicit/
│   ├── indirect/
│   └── reasoning/
│       ├── spatial/
│       └── gender/
├── evaluation/
│   ├── README.md
│   ├── compute_bias_metrics.py
│   └── compute_reasoning_ras.py
└── figures/
```

## ⚙️ Installation

We provide separate requirement files for image generation and VLM inference, since these steps may require different environments.

### Image generation environment
Use this environment to regenerate the synthetic image dataset with Z-Image-Turbo:

```bash
pip install -r requirements_generate_images.txt
```

This environment is needed for: `dataset generate_image_isco_balanced_ethnic_age.py`

For image generation with Z-Image-Turbo, a GPU environment with PyTorch, CUDA, and `diffusers` is recommended. 

### VLM inference and evaluation environment
Use this environment to run VLM inference and compute evaluation metrics:

```bash
pip install -r requirements_inference.txt
```
This environment is needed for: `inference/zero-shot.py`. `evaluation/compute_bias_metrics.py`, and `evaluation/compute_reasoning_ras.py`  

For VLM inference, GPU memory requirements depend on the selected model. Quantization can be enabled with:

```text
--quantization_config 4bits
```

or:

```text
--quantization_config 8bits
```

Some Hugging Face models may require accepting model licenses before download.


## 📊 Dataset

The dataset is generated using controlled synthetic image generation. The generation script is located at:

```text
dataset/generate_image_isco_balanced_ethnic_age.py
```

It creates balanced paired images across:

- 41 ISCO-08 sub-major occupational groups
- 10 ISCO-08 major occupational categories
- 7 ethnicity categories
- 5 age groups
- 2 exact ages per age group
- 2 gender orders: `woman-man` and `man-woman`

Due to the large size of the generated image set, the full `images/` directory is not included in this repository. The images can be regenerated using the script in `dataset/`.

After generation, apply:

```text
dataset/selected_images.txt
```

to exclude images removed during human validation. These excluded images correspond to cases with visual artifacts, demographic mismatch, occupational inconsistency, or lack of visual parity.

See `dataset/README.md` for full details.

## 💬 Prompts

The prompt templates are stored in:

```text
prompts/
```

The folder contains four prompt files:

| File                           | Setting                                                      |
| ------------------------------ | ------------------------------------------------------------ |
| `indirect_prompt.txt`          | Indirect spatial earning comparison                          |
| `explicit_prompt.txt`          | Explicit gender earning comparison                           |
| `reasoning_spatial_prompt.txt` | Structured reasoning with `left`, `right`, or `similar` comparisons |
| `reasoning_gender_prompt.txt`  | Structured reasoning with `man`, `woman`, or `similar` comparisons |

See `prompts/README.md` for details about each prompt and its expected output format.

## 🧠 Inference

Inference scripts are stored in:

```text
inference/
```

The folder contains:

| File           | Purpose                                                      |
| -------------- | ------------------------------------------------------------ |
| `VLMBase.py`   | Generic VLM wrapper for Gemma-3, Qwen3-VL, LLaVA, and InternVL models. |
| `zero-shot.py` | Runs zero-shot inference over generated image metadata and writes JSONL predictions. |

The VLM wrapper maps short aliases to Hugging Face model IDs, for example:

| Alias                   | Hugging Face model ID        |
| ----------------------- | ---------------------------- |
| `gemma3-12b-it`         | `google/gemma-3-12b-it`      |
| `gemma3-27b-it`         | `google/gemma-3-27b-it`      |
| `Qwen3-VL-4B-Instruct`  | `Qwen/Qwen3-VL-4B-Instruct`  |
| `Qwen3-VL-8B-Instruct`  | `Qwen/Qwen3-VL-8B-Instruct`  |
| `Qwen3-VL-32B-Instruct` | `Qwen/Qwen3-VL-32B-Instruct` |
| `llava-1.5-7b`          | `llava-hf/llava-1.5-7b-hf`   |
| `InternVL3_5-8B`        | `OpenGVLab/InternVL3_5-8B`   |

Example for final comparison prompts:

```bash
cd inference

python zero-shot.py \
  --input_jsonl ../dataset/metadata.jsonl \
  --image_base_dir ../dataset \
  --prompt_file_1 ../prompts/indirect_prompt.txt \
  --prompt_file_2 ../prompts/explicit_prompt.txt \
  --output_jsonl_prompt1 ../results/indirect/predictions_indirect_prompt_qwen3_vl_4b.jsonl \
  --output_jsonl_prompt2 ../results/explicit/predictions_explicit_prompt_qwen3_vl_4b.jsonl \
  --model_name Qwen3-VL-4B-Instruct \
  --quantization_config none \
  --max_new_tokens 1024
```

Example for structured reasoning prompts:

```bash
cd inference

python zero-shot.py \
  --input_jsonl ../dataset/metadata.jsonl \
  --image_base_dir ../dataset \
  --prompt_file_1 ../prompts/reasoning_spatial_prompt.txt \
  --prompt_file_2 ../prompts/reasoning_gender_prompt.txt \
  --output_jsonl_prompt1 ../results/reasoning/spatial/predictions_reasoning_spatial_qwen3_vl_4b_analysis.jsonl \
  --output_jsonl_prompt2 ../results/reasoning/gender/predictions_reasoning_gender_qwen3_vl_4b_analysis.jsonl \
  --model_name Qwen3-VL-4B-Instruct \
  --quantization_config none \
  --max_new_tokens 4096
```

See `inference/README.md` for additional details.

## 📈 Results

Model prediction outputs are stored in:

```text
results/
```

The expected structure is:

```text
results/
├── explicit/
├── indirect/
└── reasoning/
    ├── spatial/
    └── gender/
```

The `explicit/` and `indirect/` folders contain final earning-comparison predictions. The `reasoning/` folder contains structured JSON-style outputs used for attribute-level analysis.

The raw `.jsonl` files contain model outputs before human-validation filtering. To reproduce the reported results, remove the filenames listed in:

```text
dataset/selected_images.txt
```

See `results/README.md` for file descriptions and expected JSONL fields.

## 🧪 Evaluation

Evaluation scripts are located in:

```text
evaluation/
```

The main scripts are:

| Script                     | Purpose                                                      |
| -------------------------- | ------------------------------------------------------------ |
| `compute_bias_metrics.py`  | Computes MP, AR, MSR, MSPB, confidence intervals, and prompt-setting comparisons from explicit and indirect predictions. |
| `compute_reasoning_ras.py` | Computes Reasoning Attribute Score Bias (RAS) from structured reasoning outputs. |

The scripts expect result files under:

```text
../results/explicit/
../results/indirect/
../results/reasoning/spatial/
../results/reasoning/gender/
```

from inside the `evaluation/` directory.

Example usage:

```bash
cd evaluation

python compute_bias_metrics.py \
  --results-dir ../results \
  --exclude-file ../dataset/selected_images.txt \
  --out-dir ../outputs/bias_metrics

python compute_reasoning_ras.py \
  --results-dir ../results/reasoning \
  --exclude-file ../dataset/selected_images.txt \
  --out-dir ../outputs/reasoning_ras
```

See `evaluation/README.md` for more details.

## 🔁 Reproducing the benchmark

A typical workflow is:

### 1. Generate images

```bash
python generate_image_isco_balanced_ethnic_age.py \
  --output_dir ./images \
  --model_id Tongyi-MAI/Z-Image-Turbo 
```

### 2. Filter invalid images

Use `dataset/selected_images.txt` to exclude images removed during human validation.

### 3. Run VLM inference

Use the scripts in `inference/` and the prompt templates in `prompts/`. Save model outputs under:

```text
results/explicit/
results/indirect/
results/reasoning/spatial/
results/reasoning/gender/
```

### 4. Compute metrics

Run the scripts in `evaluation/`.

### 5. Generate paper tables and figures

Use the CSV/JSON outputs from `evaluation/` to create the paper tables and figures.

## Metrics

### Male Preference (MP)

Measures directional preference among determined responses:

```text
MP = (N_man - N_woman) / (N_man + N_woman)
```

Positive values indicate male-oriented associations.

### Abstention Rate (AR)

Measures how often the model avoids making a comparison:

```text
AR = N_cannot / (N_man + N_woman + N_cannot)
```

### Male Selection Rate (MSR)

Includes abstentions in the denominator:

```text
MSR = N_man / (N_man + N_woman + N_cannot)
```

### Male Selection Position Bias (MSPB)

Measures whether male selections are more frequent when the man appears on the right or left:

```text
MSPB = (N_man@right - N_man@left) / (N_man@right + N_man@left)
```

### Reasoning Attribute Score Bias (RAS)

Measures attribute-level score differences between the man and the woman:

```text
RAS_a = mean_i((s_man,a_i - s_woman,a_i) / 4)
```

Scores are normalized because each attribute is rated from 1 to 5.

## 📝 Notes on released files

- The generated image files are not included due to size.
- The generation code and metadata format are provided to support reproducibility.
- `selected_images.txt` contains images excluded after human validation.
- Raw model outputs are kept in JSONL format to preserve auditable responses.
- Evaluation scripts produce reusable `.csv` and `.json` outputs.

## ⚖️ Ethical use

This repository is intended for research on bias evaluation and auditing of multimodal models. The dataset and outputs should not be used to:

- rank real people,
- infer sensitive attributes in deployment,
- make hiring or salary decisions,
- evaluate the professional ability of individuals.

The benchmark is diagnostic and controlled. It is designed to isolate occupational gender associations under matched visual conditions.
