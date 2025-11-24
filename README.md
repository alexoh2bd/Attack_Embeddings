# Adversarial Attacks and Defenses on CLIP Models

## Group Members

Alex Oh
Sasank Garimella
Jai Pruthi

## Project Overview
This project investigates the vulnerability of Vision-Language Models (specifically OpenAI's CLIP ViT-L/14) to adversarial attacks and evaluates the effectiveness of various defense mechanisms. We demonstrate that imperceptible perturbations can catastrophically break the alignment between images and their corresponding text descriptions.


## AI Usage Disclaimer

All AI work was directed and augemented by the team members.

This project utilized Artificial Intelligence tools to assist in development:
*   **Antigravity (Google DeepMind):** Used for the implementation, coding, and optimization of the **Defensive** pipelines (MAT, PGD Training, JPEG evaluation) and data analysis. - Export of chat is attached as Antigravity-Agentic-Coding-Chat-History.md. (5000 lines)

*   **ChatGPT (OpenAI):** Used for  ideation, conceptualization of attack strategies, and development of the **Adversarial Attack** prototypes.

    - https://chatgpt.com/share/6923f1e0-3be0-800c-a19f-7d6061f5b335
    - https://chatgpt.com/share/6923f1f8-0420-800c-85c1-6def1ba19147
    - https://chatgpt.com/share/6923f28c-16a8-800c-9e61-b3ae67d7eb20

## Methodologies

### Dataset Used

#### Data Leakage Issue
**Initial Problem:** Training on `MMEB-eval` (test set) caused data leakage.
- Models appeared robust by memorizing test examples
- Zero generalization to unseen attacks

**Solution:** Strict train/test separation
- **Training:** `TIGER-Lab/MMEB-train` (split="original") - COCO train2014 images
- **Evaluation:** `TIGER-Lab/MMEB-eval` (split="test") - COCO val2014 images
- **Dataset Size:** 5,000 training images (downloaded from 113k available)

#### Dataset Documentation

Both training and evaluation datasets are thoroughly documented:
- **MMEB-train:** [TIGER-Lab/MMEB-train on HuggingFace](https://huggingface.co/datasets/TIGER-Lab/MMEB-train)
- **MMEB-eval:** [TIGER-Lab/MMEB-eval on HuggingFace](https://huggingface.co/datasets/TIGER-Lab/MMEB-eval)

**Key Dataset Details:**
- **Source:** Microsoft COCO (Common Objects in Context)
- **Task:** Image-to-Text retrieval (MSCOCO_i2t)
- **Image Distribution:**
  - Train: COCO train2014 (released 2014, 82k images total)
  - Eval: COCO val2014 (released 2014, 40k images total)
- **No overlap:** train2014 and val2014 contain completely different images
- **MMEB-train size:** 113,287 image-text pairs (we used 5,000)
- **MMEB-eval size:** 5,000 image-text pairs (we used 100 for evaluation)

**Column Naming:**
- MMEB-train uses: `qry_image_path`, `pos_text`
- MMEB-eval uses: `qry_img_path`, `tgt_text`
- Our scripts handle both formats automatically


#### Adversarial Attacks
We explored two primary attack strategies to understand model robustness:

1.  **Caption-Aware PGD (Untargeted / Dissimilarity Maximization)**
    *   **Goal:** Maximize the distance (dissimilarity) between the image embedding and its **ground truth** text embedding.
    *   **Mechanism:** Uses the true label to define the "correct" direction and pushes the image embedding directly away from it.
    *   **Result:** Finds the most efficient "escape route" for the embedding, achieving massive similarity drops (e.g., +0.25 to -0.18) with imperceptible noise ($\epsilon=8/255$). This serves as our primary benchmark for defense evaluation.

2.  **Targeted PGD (Caption-Unaware)**
    *   **Goal:** Maximize similarity to a specific, incorrect target caption (e.g., "a photo of a toaster").
    *   **Mechanism:** Ignores the true label and attempts to force the image embedding into a specific region of the latent space.
    *   **Result:** Requires significantly higher perturbation budgets to force specific semantic shifts, often resulting in visible noise. Used primarily for qualitative exploration of decision boundaries.

### Defenses Evaluated
We implemented and benchmarked three defense strategies against the Caption-Aware PGD attack:

1.  **Mixed Adversarial Training (MAT)**
    *   **Concept:** Training on clean images + heavy augmentations (blur, noise, color jitter).
    *   **Performance:** +68.6% improvement in robustness.
    *   **Pros:** Fast training (~2 hours), improves general robustness.

2.  **PGD Adversarial Training**
    *   **Concept:** Fine-tuning the model on a mix of clean images and pre-generated PGD adversarial examples.
    *   **Performance:** +96.3% improvement (Best Model Defense).
    *   **Pros:** "Vaccinates" the model against the specific attack vector.

3.  **JPEG Compression (Preprocessing)**
    *   **Concept:** Using lossy compression to remove high-frequency adversarial perturbations before inference.
    *   **Performance:** +95.7% improvement (at Quality=30).
    *   **Pros:** Zero training cost, instant deployment.

## Key Results Summary

| Model | Clean Image Cosine Similarity | Adversarial Image Cosine Similarity | Drop | Improvement | Training Time |
|:---|:---|:---|:---|:---|:---|
| **Original CLIP Model** | 0.2513 | -0.1783 | 171.0% | - | N/A |
| **Original CLIP Model JPEG (q=30)** | 0.2513 | 0.2329 | 7.3% | **+95.7%** | None |
| **Original CLIP Model JPEG (q=75)** | 0.2513 | 0.1504 | 40.2% | **+76.5%** | None |
| **Original CLIP Model JPEG (q=50)** | 0.2513 | 0.2147 | 14.6% | **+91.5%** | None |
| **Mixed Adversarial Trained Model** | 0.2417 | 0.1067 | 55.9% | **+68.6%** | ~2 hours |
| **Caption Aware Adversarial Trained Model** | 0.2507 | 0.2349 | 6.3% | **+96.3%** | ~6 hours* |

## How to Run

### 1. Environment Setup
Create a conda environment and install dependencies:
```bash
conda create -n ae python=3.12
conda activate ae
pip install -r requirements.txt
```

### 2. Download Data
Download the necessary training and evaluation images from MSCOCO:
```bash
python download_mscoco.py --num_images 5000
```

### 3. Generate Adversarial Examples
Generate adversarial examples for training (5,000 images) and evaluation (100 images):
```bash
# For training set (used by PGD defense)
python attack_mmeb_train.py --dataset MSCOCO_i2t --eps 0.0314 --steps 20 --max_samples 5000

# For evaluation set (used for benchmarking)
python attack_mmeb_eval.py --dataset MSCOCO_i2t --eps 0.0314 --steps 20 --max_samples 100
```

### 4. Train Defenses
Train the defense models:

**Mixed Adversarial Training (MAT):**
```bash
python defend_clip_mat_v2.py --batch_size 16 --lr 1e-5 --epochs 3
```

**Caption Aware Adversarial Training:**
```bash
python defend_clip_pgd_v2.py --batch_size 16 --lr 1e-5 --epochs 3 --loss_threshold 0.1
```

### 5. Apply JPEG Defense
Apply JPEG compression to the adversarial evaluation set:
```bash
python apply_jpeg_defense.py --attacked_dir mmeb_eval_attacked --qualities 75 50 30
```

### 6. Run Evaluation
Evaluate all models and defenses:
```bash
python eval_defenses_batch.py --attacked_dir mmeb_eval_attacked
```
This will generate the final comparison table and `defense_results.json`.
