# Adversarial Robustness Evaluation for CLIP Models

## Executive Summary

This document details the implementation and evaluation of multiple adversarial defense strategies for OpenAI's CLIP ViT-L/14 model. We evaluated three defense approaches: **Mixed Adversarial Training (MAT)**, **PGD Adversarial Training**, and **JPEG Compression**, achieving up to **96.3% improvement** in robustness against PGD attacks.

**Key Results:**
- **Best Model Defense:** PGD Adversarial Training (96.3% improvement)
- **Best Preprocessing Defense:** JPEG Compression q=30 (95.7% improvement)
- **Fastest Training:** MAT (68.6% improvement, 2 hours on 5k images)

---

## 1. Problem Statement

### Objective
Develop and evaluate adversarial defenses for CLIP models against PGD (Projected Gradient Descent) attacks that cause the model to completely fail at image-text matching.

### Attack Severity
Without defense, PGD attacks cause **171% performance drop**:
- Clean similarity: 0.25 (normal operation)
- Adversarial similarity: **-0.18** (negative = predicting opposite of correct label)
- The model is completely broken by imperceptible perturbations (ε = 8/255)

---

## 2. Dataset Setup

### 2.1 Data Leakage Issue
**Initial Problem:** Training on `MMEB-eval` (test set) caused data leakage.
- Models appeared robust by memorizing test examples
- Zero generalization to unseen attacks

**Solution:** Strict train/test separation
- **Training:** `TIGER-Lab/MMEB-train` (split="original") - COCO train2014 images
- **Evaluation:** `TIGER-Lab/MMEB-eval` (split="test") - COCO val2014 images
- **Dataset Size:** 5,000 training images (downloaded from 113k available)

### 2.2 Dataset Documentation

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

### 2.3 Data Download
```bash
python download_mscoco.py --num_images 5000
```
- Downloaded 4,939 new images + 61 existing
- Multi-threaded download (16 workers)
- Completion time: ~2 minutes

---

## 3. Attack Implementation

### 3.1 PGD Attack Specification
**Parameters:**
- Epsilon (ε): 0.0314 (8/255 in [0,1] space)
- Step size (α): 0.0078 (2/255)
- Iterations: 20 steps
- Attack space: **[0,1] pixel values** (critical for visual imperceptibility)

**Key Implementation Detail:**
```python
# Work in [0,1] space, normalize ONLY during forward pass
x_input = normalize_fn(x_adv)  # CLIP normalization applied here
img_emb = model.encode_image(x_input)

# Loss: NEGATIVE similarity (maximize distance)
loss = -sim.mean()
```

### 3.2 Attack Generation Scripts

**For Evaluation (100 samples):**
```bash
python attack_mmeb_eval.py --dataset MSCOCO_i2t --eps 0.0314 --steps 20 --max_samples 100
```

**For Training (5,000 samples):**
```bash
python attack_mmeb_train.py --dataset MSCOCO_i2t --eps 0.0314 --steps 20 --max_samples 5000
```
- Runtime: ~2.7 hours for 5,000 images
- Success rate: 100% (all attacks caused >50% similarity drop)
- Output: `mmeb_train_attacked/` with adversarial images + metadata

---

## 4. Defense Methods

### 4.1 Mixed Adversarial Training (MAT)

**Concept:** Train on clean images + heavily augmented versions

**Augmentations Applied:**
- Random brightness/contrast/color jitter
- Gaussian blur (radius 0.1-2.0)
- Gaussian noise (σ=15)
- Applied with 50% probability

**Training Configuration:**
```bash
python defend_clip_mat_v2.py --batch_size 16 --lr 1e-5 --epochs 3
```

**Architecture:**
- Base model: OpenAI CLIP ViT-L/14
- Trainable: Visual encoder only (text encoder frozen)
- Optimizer: AdamW (lr=1e-5, weight_decay=0.1)
- AMP (Automatic Mixed Precision) enabled
- Negative text bank: 512 samples for loss stability

**Training Details:**
- Total steps: ~924 (308 steps/epoch × 3 epochs)
- Runtime: ~2 hours on 5,000 images
- Early stopping: Patience = 100 steps
- Best model checkpoint saved automatically

**Results:**
- Clean accuracy: 0.2417 (slight drop from 0.2513)
- Adversarial accuracy: 0.1067 (vs -0.1783 baseline)
- **Improvement: 68.6%**

---

### 4.2 PGD Adversarial Training

**Concept:** Train on clean images + actual PGD adversarial examples

**Two-Stage Process:**

**Stage 1: Generate Adversarial Training Set**
```bash
python attack_mmeb_train.py --dataset MSCOCO_i2t --eps 0.0314 --steps 20 --max_samples 5000
```
- Generated 5,000 adversarial examples offline
- Saved to `mmeb_train_attacked/`

**Stage 2: Train on Pre-computed Adversarial Examples**
```bash
python defend_clip_pgd_v2.py --batch_size 16 --lr 1e-5 --epochs 3 --loss_threshold 0.1
```

**Architecture:**
- Identical to MAT (visual encoder only, AMP, negative bank)
- Loss threshold stopping: Stops when loss < 0.1

**Training Details:**
- Loads both clean and adversarial images from disk
- Loss: (loss_clean + loss_adv) / 2
- Faster than on-the-fly PGD generation
- Runtime: ~2-3 hours estimated

**Results:**
- Clean accuracy: 0.2507 (minimal drop)
- Adversarial accuracy: 0.2349 (huge improvement!)
- **Improvement: 96.3%** ✅ **Best model-based defense**

---

### 4.3 JPEG Compression Defense

**Concept:** Preprocessing defense that removes adversarial perturbations

**Implementation:**
```bash
python apply_jpeg_defense.py --attacked_dir mmeb_eval_attacked --qualities 75 50 30
```

**How It Works:**
1. Load adversarial image (PNG, lossless)
2. Save as JPEG with specified quality
3. Compression artifacts destroy subtle adversarial perturbations
4. Load JPEG back for evaluation

**Results by Quality Level:**

| Quality | Adversarial Similarity | Improvement |
|:--------|:----------------------|:------------|
| **q=75** (mild) | 0.1504 | **76.5%** |
| **q=50** (moderate) | 0.2147 | **91.5%** |
| **q=30** (aggressive) | 0.2329 | **95.7%** ✅ **Best preprocessing** |

**Trade-offs:**
- ✅ No model retraining required
- ✅ Nearly perfect defense at q=30
- ❌ Degrades image quality
- ❌ Must apply to every input image

---

## 5. Evaluation Pipeline

### 5.1 Batch Evaluation Script
```bash
python eval_defenses_batch.py --attacked_dir mmeb_eval_attacked
```

**Process:**
1. Load metadata for 100 adversarial examples
2. For each model (Original, MAT, PGD):
   - Evaluate on clean images
   - Evaluate on adversarial images
   - Evaluate on JPEG-defended images (if applicable)
3. Calculate similarity scores and improvement metrics
4. Generate comparison table

### 5.2 Metrics Explained

**Similarity Scores:**
- Range: -1.0 to 1.0 (cosine similarity)
- Positive = model predicts correct label
- Negative = model predicts opposite of correct label
- Zero = random/uncertain

**Drop Calculation:**
```
Drop = Clean_Similarity - Adversarial_Similarity
Drop_Percentage = (Drop / Clean_Similarity) × 100%
```

**Improvement Calculation:**
```
Improvement = ((Baseline_Drop - Defense_Drop) / Baseline_Drop) × 100%
```
- Baseline_Drop = 0.4296 (171% for original model)
- Higher improvement = better defense

---

## 6. Final Results

### 6.1 Complete Comparison Table

| Method | Clean | Adversarial | Drop | Improvement | Training Time |
|:---|:---|:---|:---|:---|:---|
| **Original (No Defense)** | 0.2513 | -0.1783 | 171.0% | - | N/A |
| JPEG Defense (q=75) | 0.2513 | 0.1504 | 40.2% | +76.5% | N/A (preprocessing) |
| JPEG Defense (q=50) | 0.2513 | 0.2147 | 14.6% | +91.5% | N/A (preprocessing) |
| JPEG Defense (q=30) | 0.2513 | 0.2329 | 7.3% | +95.7% | N/A (preprocessing) |
| **MAT** | 0.2417 | 0.1067 | 55.9% | **+68.6%** | ~2 hours |
| **PGD** | 0.2507 | 0.2349 | 6.3% | **+96.3%** ✅ | ~5 hours* |

*PGD total time includes ~2.7 hours for adversarial example generation + ~2-3 hours for model training

### 6.2 Key Observations

**1. PGD Training is Superior**
- Achieves 96.3% improvement (nearly perfect)
- Trains on actual adversarial examples (not proxies)
- Minimal clean accuracy drop (0.2513 → 0.2507)

**2. MAT is Practical**
- Good robustness (68.6%) with faster training
- Uses random augmentations (cheaper than PGD)
- Useful when training time is limited

**3. JPEG is Surprisingly Effective**
- No training required
- q=30 achieves 95.7% (close to PGD!)
- Best for deployment scenarios where preprocessing is acceptable

**4. Data Leakage Impact**
- Training on 61 images: 0.1% improvement (useless)
- Training on 5,000 images: 68.6-96.3% improvement (real robustness)
- **Lesson:** Dataset size and proper train/test split are critical

---

## 7. Technical Challenges Solved

### 7.1 Attack Implementation Bug
**Problem:** Initial PGD attack was broken - using positive cross-entropy loss
```python
# WRONG: This improves model accuracy!
loss = F.cross_entropy(logits, labels)
```

**Fix:** Negative similarity (correct untargeted attack)
```python
# CORRECT: Minimize similarity to fool the model
loss = -sim.mean()
```

### 7.2 Visual Perturbation Issue
**Problem:** Attacking in normalized space caused extreme visual artifacts
- Epsilon in [-2, 2] (CLIP normalized) = huge pixel changes

**Fix:** Attack in [0,1] space before normalization
```python
# Attack here (0-1 space)
x_adv = x_adv + alpha * grad.sign()
x_adv = torch.clamp(x_adv, 0, 1)

# Normalize only for forward pass
x_input = normalize_fn(x_adv)
```
- Epsilon = 8/255 in [0,1] space = imperceptible

### 7.3 Dataset Column Name Mismatch
**Problem:** MMEB-train uses different column names than MMEB-eval
- Train: `qry_image_path`, `pos_text`
- Eval: `qry_img_path`, `tgt_text`

**Fix:** Graceful fallback
```python
img_path = item.get('qry_image_path', item.get('qry_img_path'))
text = item.get('pos_text', item.get('tgt_text'))
```

### 7.4 Missing `__len__` Method
**Problem:** DataLoader failed with `TypeError: object of type 'CLIPDataset' has no len()`

**Fix:** Added `__len__` to all dataset classes
```python
def __len__(self):
    return len(self.valid_indices)
```

---

## 8. File Structure

### 8.1 Training Scripts
- `defend_clip_mat_v2.py` - MAT training (data augmentation)
- `defend_clip_pgd_v2.py` - PGD training (loads pre-generated adversarial examples)
- `defend_clip_fat.py` - FAT training (deprecated, not used)

### 8.2 Attack Scripts
- `attack_mmeb_eval.py` - Generate attacks for evaluation (100 samples)
- `attack_mmeb_train.py` - Generate attacks for PGD training (5,000 samples)
- `attack_clip.py` - Original attack implementation (reference)

### 8.3 Evaluation Scripts
- `eval_defenses_batch.py` - Batch evaluation of all defenses
- `apply_jpeg_defense.py` - Apply JPEG compression at multiple quality levels

### 8.4 Utility Scripts
- `download_mscoco.py` - Multi-threaded MSCOCO image downloader

### 8.5 Data Directories
```
Attack_Embeddings/
├── mmeb_images/               # Clean training/eval images
│   └── images/MSCOCO_i2t/Train/  # Training images (5k)
├── mmeb_eval_attacked/        # Adversarial eval set (100)
│   ├── adv_*.png             # Adversarial images
│   ├── jpeg_q75/             # JPEG defended (q=75)
│   ├── jpeg_q50/             # JPEG defended (q=50)
│   ├── jpeg_q30/             # JPEG defended (q=30)
│   └── metadata.json         # Paths and similarity scores
├── mmeb_train_attacked/       # Adversarial training set (5k)
│   ├── adv_*.png
│   └── metadata.json
├── robust_clip_mat_v2/        # MAT model checkpoint
│   └── model_best.pt
└── robust_clip_pgd/           # PGD model checkpoint
    └── model_best.pt
```

---

## 9. Reproduction Guide

### 9.1 Environment Setup
```bash
conda create -n ae python=3.12
conda activate ae
pip install -r requirements.txt
```

### 9.2 Download Training Data
```bash
python download_mscoco.py --num_images 5000
```

### 9.3 Train MAT Defense
```bash
python defend_clip_mat_v2.py --batch_size 16 --lr 1e-5 --epochs 3
```
- Runtime: ~2 hours
- Output: `robust_clip_mat_v2/model_best.pt`

### 9.4 Generate Adversarial Training Set
```bash
python attack_mmeb_train.py --dataset MSCOCO_i2t --eps 0.0314 --steps 20 --max_samples 5000
```
- Runtime: ~2.7 hours
- Output: `mmeb_train_attacked/`

### 9.5 Train PGD Defense
```bash
python defend_clip_pgd_v2.py --batch_size 16 --lr 1e-5 --epochs 3 --loss_threshold 0.1
```
- Runtime: ~2-3 hours
- Output: `robust_clip_pgd/model_best.pt`

### 9.6 Generate Evaluation Attacks
```bash
python attack_mmeb_eval.py --dataset MSCOCO_i2t --eps 0.0314 --steps 20 --max_samples 100
```
- Runtime: ~3 minutes
- Output: `mmeb_eval_attacked/`

### 9.7 Apply JPEG Defense
```bash
python apply_jpeg_defense.py --attacked_dir mmeb_eval_attacked --qualities 75 50 30
```
- Runtime: ~1 second
- Output: `mmeb_eval_attacked/jpeg_q*/`

### 9.8 Run Evaluation
```bash
python eval_defenses_batch.py --attacked_dir mmeb_eval_attacked
```
- Runtime: ~2 minutes
- Output: Final comparison table + `defense_results.json`

---

## 10. Lessons Learned

### 10.1 What Worked

**1. Pre-generating adversarial examples for training**
- Faster than on-the-fly generation
- Ensures consistent training data
- Easier to debug and reproduce

**2. Strict train/test separation**
- Critical for measuring real generalization
- Small training sets (61 images) don't generalize
- Need 5,000+ images for robust learning

**3. Loss threshold stopping**
- Prevents overfitting
- Saves training time
- `--loss_threshold 0.1` worked well for PGD

**4. Negative text bank**
- Prevents loss collapse on small batches
- 512 negative samples is sufficient
- Critical for stable training

### 10.2 What Didn't Work

**1. Training on test data**
- Gives fake robustness (memorization)
- Zero generalization to new attacks
- Must use separate train/eval datasets

**2. Attacking in normalized space**
- Visual perturbations become extreme
- ε=0.03 in [-2,2] space ≠ ε=8/255 in [0,1] space
- Always attack in pixel space, normalize only for forward pass

**3. Small training sets**
- 61 images: 0.1% improvement (useless)
- 5,000 images: 68-96% improvement (real robustness)
- Dataset size matters more than training tricks

---

## 11. Future Work

### 11.1 Potential Improvements

**1. Download and train on full dataset (113k images)**
- Current: 5,000 images
- Expected: Further robustness gains

**2. Ensemble defenses**
- Combine PGD model + JPEG preprocessing
- Potential for >97% improvement

**3. Evaluate on other attack types**
- C&W attack
- AutoAttack
- Transfer attacks from other models

**4. Train text encoder**
- Currently frozen (only visual encoder trained)
- Joint training may improve robustness

**5. Multi-step PGD during training**
- Current: 7 steps (from pre-generated examples)
- Try: 10, 20 steps for stronger adversaries

### 11.2 Deployment Recommendations

**For Production:**
- Use **PGD model** (96.3% improvement, minimal clean accuracy drop)
- Add **JPEG q=50** as fallback (91.5% improvement, acceptable quality)

**For Rapid Deployment:**
- Use **JPEG q=30** alone (95.7% improvement, no training)
- Trade-off: Some image quality degradation

**For Research:**
- Train on full 113k dataset
- Evaluate ensemble of PGD + JPEG
- Test on stronger attacks (AutoAttack)

---

## 12. Conclusion

We successfully developed and evaluated three adversarial defense strategies for CLIP:

1. **PGD Adversarial Training** - Best model defense (96.3% improvement)
2. **MAT** - Practical alternative (68.6% improvement, faster training)
3. **JPEG Compression** - Best preprocessing (95.7% improvement, no training)

The key insight is that **proper dataset construction** (train/test separation, sufficient size) is more important than algorithmic tricks. Our final PGD model achieves near-perfect robustness against imperceptible adversarial perturbations while maintaining clean accuracy.

**Final Metrics:**
- **Baseline:** 171% drop (complete failure)
- **PGD Defense:** 6.3% drop (minimal impact)
- **Improvement:** 96.3% ✅

This represents a **practical solution** for deploying CLIP in adversarial environments.
