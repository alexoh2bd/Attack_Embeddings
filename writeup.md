# Adversarial Robustness Evaluation for CLIP Models

## 1. Problem Statement
**Objective:** Develop and evaluate adversarial defenses for CLIP ViT-L/14 models against Caption Aware PGD attacks that cause the model to completely fail at image-text matching.

**Attack Severity:** Without defense, PGD attacks cause a **171% performance drop**, flipping positive cosine similarity (0.25) to negative (-0.18).

## 2. Dataset & Methodology

### 2.1 Data Leakage & Solution
**Issue:** Initial training on `MMEB-eval` caused massive overfitting and data leakage.
**Solution:** We implemented strict train/test separation:
*   **Training:** 5,000 images from `TIGER-Lab/MMEB-train` (COCO train2014).
*   **Evaluation:** 100 images from `TIGER-Lab/MMEB-eval` (COCO val2014).

### 2.2 Attack Specification (Caption Aware PGD)
We implemented a PGD attack maximizing the dissimilarity between images and their captions.
*   **Epsilon (ε):** 0.0314 (8/255)
*   **Step size (α):** 0.0078 (2/255)
*   **Iterations:** 20 steps
*   **Constraint:** Attack applied in **[0,1] pixel space** to ensure visual imperceptibility before CLIP normalization.

## 3. Defense Implementation

### 3.1 Mixed Adversarial Training (MAT)
**Concept:** Train on clean images + randomly augmented versions (noise, blur, jitter).

**Training Configuration:**
```bash
python defend_clip_mat_v2.py --batch_size 16 --lr 1e-5 --epochs 3
```
*   **Augmentations:** Random brightness/contrast, Gaussian blur (r=0.1-2.0), Gaussian noise (σ=15).
*   **Optimizer:** AdamW (lr=1e-5, weight_decay=0.1).
*   **Architecture:** Visual encoder trainable, text encoder frozen.

### 3.2 Caption Aware Adversarial Training
**Concept:** Train on clean images + actual PGD adversarial examples generated offline.

**Process:**
1.  **Generate:** Created 5,000 adversarial examples using the attack parameters above.
2.  **Train:**
    ```bash
    python defend_clip_pgd_v2.py --batch_size 16 --lr 1e-5 --epochs 3 --loss_threshold 0.1
    ```
*   **Loss Function:** Average of clean loss and adversarial loss.
*   **Stopping Criterion:** Training stops when loss < 0.1 to prevent overfitting.

### 3.3 JPEG Compression Defense
**Concept:** Preprocessing defense to remove high-frequency adversarial noise.
*   **Qualities Tested:** 75 (mild), 50 (moderate), 30 (aggressive).
*   **Mechanism:** Standard JPEG compression applied before model inference.

## 4. Results

| Model / Defense | Clean Similarity | Adversarial Similarity | Improvement | Training Time |
| :--- | :--- | :--- | :--- | :--- |
| **Original CLIP** | 0.2513 | -0.1783 | - | N/A |
| **MAT Defense** | 0.2417 | 0.1067 | +68.6% | ~2 hours |
| **Caption Aware Training** | 0.2507 | 0.2349 | **+96.3%** | ~6 hours* |
| **JPEG (Q=30)** | 0.2513 | 0.2329 | +95.7% | None |

*\*Includes ~2.7 hours for attack generation + ~3 hours training.*


## 6. Conclusion
**Caption Aware Adversarial Training** proved to be the superior defense, recovering 96.3% of the baseline performance. However, **JPEG compression (q=30)** offers a surprisingly effective (95.7%) and zero-cost alternative for deployment scenarios where model retraining is not feasible.
