# Defense Training Commands

This document lists all available defense training scripts and their commands.

---

## 1. MAT (Mixed Adversarial Training)

**Script:** `defend_clip_mat.py`

**Description:** Original MAT implementation with data augmentation and gradient accumulation.

**Command:**
```bash
python defend_clip_mat.py
```

**Output:** `robust_clip_openai/model.pt`

**Features:**
- Aggressive data augmentation (color jitter, blur, noise)
- Gradient accumulation (micro_batch_size=2, accumulation_steps=8)
- Negative Text Bank (512 samples)
- Early stopping and best model selection

---

## 2. MAT V2 (Optimized)

**Script:** `defend_clip_mat_v2.py`

**Description:** Re-implementation from first principles with AMP and DataLoader.

**Command:**
```bash
python defend_clip_mat_v2.py --batch_size 16 --lr 1e-5 --epochs 3
```

**Parameters:**
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-6, recommended: 1e-5)
- `--epochs`: Number of epochs (default: 1)
- `--max_steps`: Max training steps (default: 1000)
- `--output_dir`: Output directory (default: robust_clip_mat_v2)

**Output:** `robust_clip_mat_v2/model_best.pt`

**Features:**
- Automatic Mixed Precision (AMP)
- PyTorch DataLoader with 2 workers
- Negative Text Bank (512 samples)
- Early stopping (patience=100)
- Per-step timing logs

---

## 3. FAT (Fast Adversarial Training)

**Script:** `defend_clip_fat.py`

**Description:** FGSM-based adversarial training for faster convergence.

**Command:**
```bash
python defend_clip_fat.py
```

**Output:** `robust_clip_fat/model.pt`

**Features:**
- Single-step FGSM attacks (epsilon=8/255)
- Gradient accumulation
- Negative Text Bank (512 samples)
- Early stopping and best model selection

---

## 4. PGD (Full PGD Adversarial Training)

**Script:** `defend_clip_pgd.py`

**Description:** Strongest defense using iterative PGD attacks.

**Command:**
```bash
python defend_clip_pgd.py --batch_size 16 --epochs 1
```

**Parameters:**
- `--batch_size`: Batch size (default: 16)
- `--lr`: Learning rate (default: 1e-6)
- `--epochs`: Number of epochs (default: 1)
- `--max_steps`: Max training steps (default: 1000)
- `--output_dir`: Output directory (default: robust_clip_pgd)

**Output:** `robust_clip_pgd/model_best.pt`

**Features:**
- 7-step PGD attacks (epsilon=8/255, alpha=2/255)
- AMP for efficiency
- Negative Text Bank (512 samples)
- Early stopping (patience=50, lower due to slower training)

---

## Orchestration Script

**Script:** `run_all_training.py`

**Description:** Sequential execution of MAT and FAT training with logging.

**Command:**
```bash
python run_all_training.py
```

**Output:**
- Logs in `logs/` directory
- Models in respective directories

---

## Recommended Training Order

1. **MAT V2** (balanced speed and effectiveness)
2. **FAT** (fast baseline)
3. **PGD** (strongest but slowest)

---

## Tips

- Use `--lr 1e-5` for MAT V2 (default 1e-6 is too low)
- Monitor GPU memory with `nvidia-smi`
- Check training logs for loss trends
- PGD training is ~7x slower than MAT due to iterative attacks
- Best models are saved as `model_best.pt`, final models as `model_final.pt`
