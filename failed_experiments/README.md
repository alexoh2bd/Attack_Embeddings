# Failed Experiments Archive

This directory contains experiments that didn't work as expected or are no longer relevant.

## Directory Structure

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

## Lessons Learned

1. **Architecture consistency matters** - Always use the same model for attack and defense
2. **Augmentation ≠ Adversarial training** - Need actual adversarial perturbations
3. **JPEG defense is surprisingly effective** - 98% recovery with no training
4. **Clean accuracy is sacred** - Don't sacrifice >5% for marginal gains

## Next Steps

Use **FAT (Fast Adversarial Training)** with actual FGSM/PGD examples instead of data augmentation.
