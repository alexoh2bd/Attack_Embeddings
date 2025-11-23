# Failed Experiments Archive

This directory contains deprecated experiments, old scripts, and reference implementations that are no longer actively used.

---

## Currently Deprecated (Moved 2025-11-23)

### Defense Scripts (Superseded)

**`defend_clip_pgd_old.py`**
- **Status:** ❌ Replaced by `defend_clip_pgd_v2.py`
- **Issue:** On-the-fly PGD generation during training (slow), incorrect attack loss function
- **Replaced by:** Pre-generated adversarial examples + verified attack logic

**`defend_clip_fat.py`**
- **Status:** ❌ Not used (FAT training abandoned)
- **Issue:** Slower than MAT, minimal robustness gains
- **Alternative:** Use MAT (faster) or PGD (stronger)

**`defend_clip_jpeg.py`**
- **Status:** ❌ Stub/incomplete
- **Replaced by:** `apply_jpeg_defense.py` (working implementation)

### Attack Scripts (Reference)

**`attack_clip_reference.py`**
- **Status:** ✅ Reference implementation (working)
- **Note:** Original attack script, kept for reference
- **Actively used:** `attack_mmeb_eval.py` and `attack_mmeb_train.py` (derived from this)

### Evaluation Scripts (Old)

**`eval_robust_openai.py`**
- **Status:** ❌ Old evaluation script
- **Replaced by:** `eval_defenses_batch.py` (batch evaluation with JPEG support)

### Utility Scripts (Deprecated)

**`run_all_training.py`**
- **Status:** ❌ Old batch training runner
- **Issue:** Attempted to train all defenses sequentially
- **Current approach:** Train individually as needed

**`match_file.py`**
- **Status:** ❌ Unknown purpose/debugging
- **Deprecated:** No longer needed

**`inspect_dataset.py`**
- **Status:** ✅ Debugging utility (no longer needed)
- **Purpose:** Inspected HuggingFace dataset structure

### Test Files (Deprecated)

**`adv.png`, `adv_jpeg.png`**
- **Status:** ✅ Test images from early experiments
- **Note:** Replaced by systematic evaluation in `mmeb_eval_attacked/`

**`sample_image.jpg`**
- **Status:** ✅ Test image
- **Deprecated:** Using MMEB datasets now

**`labir.ipynb`**
- **Status:** ❌ Old/unknown notebook
- **Replaced by:** Python scripts

**`run_ir.sh`**
- **Status:** ❌ Old shell script
- **Deprecated:** Unknown purpose

**`pip-install.log`**
- **Status:** ✅ Installation log
- **Deprecated:** No longer needed

---

## Previous Failed Experiments

### `huggingface_mat/`
**Status:** ❌ Model architecture mismatch  
**Issue:** Trained on `vit-base-patch32` but evaluated against attacks from `ViT-L/14`  
**Result:** Apparent robustness was due to poor attack transferability, not actual training  
**Files:**
- `defend_clip_mat.py` - Training script
- `eval_robust_simple.py` - Evaluation script
- `attack_clip_hf.py` - HuggingFace attack generator
- `robust_clip_mat/` - Trained model checkpoint
- `adv_hf*.png` - Generated adversarial images

### `pgd_training/`
**Status:** ❌ Unstable training  
**Issue:** PGD-based adversarial training caused loss divergence  
**Files:**
- `defend_clip_comprehensive.py` - Unstable PGD training
- `defend_clip.py` - Demo training script
- `robust_clip_model/` - Old checkpoint
- `eval_robust_model.py` - Old evaluation script

### `test_scripts/`
**Status:** ✅ Debugging utilities (no longer needed)  
**Files:**
- `test_import.py` - Import debugging
- `test_print.py` - Output buffering test
- `test_pgd_fixed.py` - Converted notebook

### `old_notebooks/`
**Status:** ✅ Superseded by Python scripts  
**Files:**
- `PGD_Demo-Copy1.ipynb` - Original notebook
- `PGD_Demo_Fixed.ipynb` - Fixed version

---

## Active Scripts (Main Directory)

### ✅ Working Defense Scripts
- `defend_clip_mat_v2.py` - MAT training (68.6% improvement, ~2 hours)
- `defend_clip_pgd_v2.py` - PGD training (96.3% improvement, ~5 hours)

### ✅ Working Attack Scripts
- `attack_mmeb_eval.py` - Generate attacks for evaluation
- `attack_mmeb_train.py` - Generate attacks for training

### ✅ Working Evaluation Scripts
- `eval_defenses_batch.py` - Batch evaluation of all defenses
- `apply_jpeg_defense.py` - JPEG compression defense

### ✅ Utilities
- `download_mscoco.py` - Multi-threaded image downloader
- `writeup.md` - Complete experiment documentation
- `training.md` - Training documentation
- `requirements.txt` - Dependencies

---

## Lessons Learned

### What Failed

1. **On-the-fly PGD generation** - Too slow and buggy. Pre-generate adversarial examples instead.
2. **FAT (Fast Adversarial Training)** - Not worth the complexity. Use MAT or PGD.
3. **Attacking in normalized space** - Causes extreme visual artifacts. Always attack in [0,1] pixel space.
4. **Training on test data** - Gives fake robustness via memorization. Use proper train/test split.
5. **Small datasets** - 61 images = 0.1% improvement. Need 5,000+ for real robustness.

### What Worked

1. **Pre-generated adversarial examples** - Fast, consistent, reproducible.
2. **MAT with data augmentation** - Simple, effective (68.6% improvement).
3. **PGD adversarial training** - Best model defense (96.3% improvement).
4. **JPEG compression** - Best preprocessing defense (95.7% improvement, no training).
5. **Negative text bank** - Prevents loss collapse on small batches.
6. **Loss threshold stopping** - Prevents overfitting, saves time.

---

## Final Results Summary

| Method | Improvement | Training Time | Status |
|:---|:---|:---|:---|
| **PGD** | 96.3% | ~5 hours | ✅ Best model defense |
| **JPEG (q=30)** | 95.7% | None | ✅ Best preprocessing |
| **MAT** | 68.6% | ~2 hours | ✅ Fast alternative |
| FAT | Not measured | ~6-8 hours (est.) | ❌ Abandoned |
| HuggingFace MAT | Fake (mismatch) | N/A | ❌ Architecture error |

**Recommendation:** Use PGD for production, JPEG for rapid deployment, MAT for research/prototyping.
