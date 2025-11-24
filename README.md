# Attack_Embeddings

A framework for building and evaluating embedding-space adversarial attacks on vision & multimodal models.

## Overview

This project provides tools and scripts for:

* Generating adversarial perturbations in embedding space (rather than just input space)
* Training models / embedding networks under attack scenarios
* Defending models with embedding-space and input-space strategies
* Evaluating robustness of embedding & multimodal pipelines

## Key Features

* Attack modules: e.g., embedding-space attacks such as for CLIP models.
* Defense modules: input transformations (e.g., JPEG compression) and embedding regularisation.
* Full training / evaluation pipelines (see `attack_mmeb_train.py`, `attack_mmeb_eval.py`).
* Modular design: add new attack / defense strategies under `src/pipe/`.
* Logging, experiment tracking and visualization support (via `logs/`, `update_visualization.py`).
* Notebook demo to walk through key components: `adversarial_demo.ipynb`.

## Repository Structure

```
.
├── src/pipe/                  # core attack / defense pipelines  
├── logs/                      # experiment logs  
├── failed_experiments/        # archive of experiments that didn’t succeed  
├── adversarial_demo.ipynb     # notebook demo  
├── training.md                # detailed training instructions  
├── walkthrough.md             # step-by-step project use guide  
├── writeup.md                 # project write-up / background  
├── requirements.txt           # Python dependencies  
├── *.py                      # utility & main scripts:
│   ├── add_comparison.py
│   ├── apply_jpeg_defense.py
│   ├── attack_mmeb_train.py
│   ├── attack_mmeb_eval.py
│   ├── defend_clip_mat_v2.py
│   ├── defend_clip_pgd_v2.py
│   ├── download_mscoco.py
│   ├── eval_defenses_batch.py
│   └── update_visualization.py
```

## Getting Started

### Prerequisites

* Python 3.x (tested on 3.8+)
* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```
* Download required dataset(s) (e.g., COCO) via `download_mscoco.py`.

### Running an Attack + Defense Workflow

1. Choose an attack scenario in `src/pipe/`.
2. Use `attack_mmeb_train.py` to train the embedding-model under adversarial regime.
3. Apply defenses (e.g., `apply_jpeg_defense.py`, `defend_clip_pgd_v2.py`).
4. Evaluate robustness using `attack_mmeb_eval.py` and optionally `eval_defenses_batch.py`.
5. Visualise results via `update_visualization.py`.

### Notebook Demo

Open `adversarial_demo.ipynb` to get an interactive walkthrough of:

* how attacks are constructed
* how to plug in new loss / perturbation functions
* how embeddings move in response to attacks.

## How to Add Your Own Attack / Defense

1. Create a new module under `src/pipe/`, e.g., `my_embedding_attack.py` or `my_defense.py`.
2. Derive from existing base classes (if present) or follow pattern of existing attacks.
3. Add argument parsing and integration into training/eval scripts.
4. Update `walkthrough.md` if adding a major new functionality.

## Use Cases

* Research in adversarial robustness for multimodal / embedding models.
* Benchmarking embedding‐space perturbations across vision or text modalities.
* Prototyping novel defense strategies in embedding space.
