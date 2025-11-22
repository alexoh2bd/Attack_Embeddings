#!/usr/bin/env python3
"""eval_models_complete.py

Complete evaluation using the SAME model as attack_clip.py (ViT-L/14).
Shows attack effectiveness clearly, then compares robust model.
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(__file__))

from attack_clip import (
    load_clip_model, build_label_embs, topk_for_labels, 
    CLIP_MEAN, CLIP_STD, image_tta_emb
)
from torchvision.transforms import Compose, Normalize
from transformers import CLIPModel, CLIPProcessor
import io

def apply_jpeg(img_path, quality=75):
    """Apply JPEG compression."""
    img = Image.open(img_path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    img_jpeg = Image.open(buffer).convert("RGB")
    temp_path = img_path.replace('.png', '_jpeg.png')
    img_jpeg.save(temp_path)
    return temp_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default="sample_image.jpg")
    parser.add_argument("--adv_img", type=str, default="adv.png")
    parser.add_argument("--labels", nargs="+", default=["white dog", "cat", "car"])
    parser.add_argument("--robust_model", type=str, default="robust_clip_mat")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print("COMPLETE ROBUSTNESS EVALUATION")
    print(f"{'='*70}\n")
    
    # Load original model (SAME as attack_clip.py)
    print("Loading ORIGINAL model (ViT-L/14)...")
    model_orig, preprocess_orig = load_clip_model("ViT-L/14", device)
    dtype_orig = next(model_orig.parameters()).dtype
    
    # Load robust model (HuggingFace)
    print(f"Loading ROBUST model from {args.robust_model}...")
    model_robust = CLIPModel.from_pretrained(args.robust_model).to(device)
    preprocess_robust = CLIPProcessor.from_pretrained(args.robust_model)
    dtype_robust = next(model_robust.parameters()).dtype
    model_robust.eval()
    
    # Preprocessing setup for original model
    if isinstance(preprocess_orig.transforms[-1], Normalize):
        preprocess_no_norm = Compose(preprocess_orig.transforms[:-1])
    else:
        preprocess_no_norm = Compose(preprocess_orig.transforms[:-1])
    
    mean_orig = torch.tensor(CLIP_MEAN, device=device, dtype=dtype_orig).view(1, 3, 1, 1)
    std_orig = torch.tensor(CLIP_STD, device=device, dtype=dtype_orig).view(1, 3, 1, 1)
    normalize_fn = lambda x: (x - mean_orig) / std_orig
    
    # Build label embeddings
    templates = ["a photo of a {}"]
    prompts = [t.format(lbl) for lbl in args.labels for t in templates]
    idx_map = [i for i in range(len(args.labels)) for _ in templates]
    
    label_embs_orig = build_label_embs(model_orig, prompts, idx_map, device, dtype_orig)
    
    # Build label embeddings for robust model
    text_tokens_hf = preprocess_robust(text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    text_tokens_hf = {k: v.to(device) for k, v in text_tokens_hf.items()}
    
    with torch.no_grad():
        txt_feats = model_robust.get_text_features(**text_tokens_hf)
        txt_feats = F.normalize(txt_feats.to(dtype=dtype_robust), dim=-1)
    
    num_labels = max(idx_map) + 1
    label_embs_list = []
    for i in range(num_labels):
        inds = [j for j, k in enumerate(idx_map) if k == i]
        vec = txt_feats[inds].mean(dim=0, keepdim=True)
        vec = F.normalize(vec, dim=-1)
        label_embs_list.append(vec)
    label_embs_robust = torch.cat(label_embs_list, dim=0)
    
    # Load images
    pil_img = Image.open(args.img).convert("RGB")
    img_tensor = preprocess_no_norm(pil_img).unsqueeze(0).to(device, dtype=dtype_orig)
    
    # ============================================================
    # TEST 1: Clean Image with Original Model
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 1: Clean Image (Original Model ViT-L/14)")
    print(f"{'='*70}\n")
    
    # Use TTA like attack_clip.py does
    tta_emb = image_tta_emb(model_orig, preprocess_orig, pil_img, device, dtype=dtype_orig)
    label_embs_cast = label_embs_orig.to(dtype=tta_emb.dtype)
    cos = (tta_emb @ label_embs_cast.T).squeeze(0)
    
    print("Baseline (with TTA):")
    for lbl, c in zip(args.labels, cos.detach().cpu().numpy()):
        print(f"  {lbl:20s}\t{c:6.3f}")
    
    clean_baseline = cos[0].item()
    
    # ============================================================
    # TEST 2: Adversarial Image with Original Model
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 2: Adversarial Image (Original Model ViT-L/14)")
    print(f"{'='*70}\n")
    
    adv_pil = Image.open(args.adv_img).convert("RGB")
    adv_tensor = preprocess_no_norm(adv_pil).unsqueeze(0).to(device, dtype=dtype_orig)
    
    vals_adv, _, _ = topk_for_labels(model_orig, normalize_fn(adv_tensor), label_embs_orig, args.labels)
    
    print("After PGD Attack:")
    for lbl, c, p in vals_adv:
        print(f"  {lbl:20s}\t{c:6.3f}")
    
    adv_similarity = vals_adv[0][1]
    attack_drop = clean_baseline - adv_similarity
    
    print(f"\n🎯 Attack Effectiveness:")
    print(f"  Clean:       {clean_baseline:.3f}")
    print(f"  Adversarial: {adv_similarity:.3f}")
    print(f"  Drop:        {attack_drop:.3f} ({attack_drop/clean_baseline*100:.1f}%)")
    
    # ============================================================
    # TEST 3: JPEG Defense (Original Model)
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 3: JPEG Defense (Original Model)")
    print(f"{'='*70}\n")
    
    jpeg_path = apply_jpeg(args.adv_img, quality=75)
    jpeg_pil = Image.open(jpeg_path).convert("RGB")
    jpeg_tensor = preprocess_no_norm(jpeg_pil).unsqueeze(0).to(device, dtype=dtype_orig)
    
    vals_jpeg, _, _ = topk_for_labels(model_orig, normalize_fn(jpeg_tensor), label_embs_orig, args.labels)
    
    print("After JPEG Compression:")
    for lbl, c, p in vals_jpeg:
        print(f"  {lbl:20s}\t{c:6.3f}")
    
    jpeg_similarity = vals_jpeg[0][1]
    jpeg_recovery = jpeg_similarity - adv_similarity
    
    print(f"\n🛡️ JPEG Defense:")
    print(f"  Adversarial: {adv_similarity:.3f}")
    print(f"  After JPEG:  {jpeg_similarity:.3f}")
    print(f"  Recovery:    {jpeg_recovery:.3f} ({jpeg_recovery/attack_drop*100:.1f}%)")
    
    # ============================================================
    # TEST 4: Robust Model
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 4: MAT-Trained Robust Model")
    print(f"{'='*70}\n")
    
    # Helper function for robust model
    def eval_robust(img_path):
        img = Image.open(img_path).convert("RGB")
        inputs = preprocess_robust(text=prompts, images=img, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model_robust(**inputs)
            image_embeds = F.normalize(outputs.image_embeds, dim=-1)
            similarity = (image_embeds @ label_embs_robust.T).squeeze(0)
        return similarity.cpu().numpy()
    
    sim_robust_clean = eval_robust(args.img)
    sim_robust_adv = eval_robust(args.adv_img)
    sim_robust_jpeg = eval_robust(jpeg_path)
    
    print("Clean Image:")
    for lbl, s in zip(args.labels, sim_robust_clean):
        print(f"  {lbl:20s}\t{s:6.3f}")
    
    print("\nAdversarial Image:")
    for lbl, s in zip(args.labels, sim_robust_adv):
        print(f"  {lbl:20s}\t{s:6.3f}")
    
    print("\nAdversarial + JPEG:")
    for lbl, s in zip(args.labels, sim_robust_jpeg):
        print(f"  {lbl:20s}\t{s:6.3f}")
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}\n")
    
    gt_label = args.labels[0]
    
    print(f"Original Model (ViT-L/14) - '{gt_label}':")
    print(f"  Clean:                {clean_baseline:.3f}")
    print(f"  Adversarial:          {adv_similarity:.3f}  (drop: {attack_drop:.3f}, {attack_drop/clean_baseline*100:.1f}%)")
    print(f"  Adversarial + JPEG:   {jpeg_similarity:.3f}  (recovery: {jpeg_recovery/attack_drop*100:.1f}%)")
    
    print(f"\nRobust Model (MAT-trained) - '{gt_label}':")
    print(f"  Clean:                {sim_robust_clean[0]:.3f}")
    print(f"  Adversarial:          {sim_robust_adv[0]:.3f}  (more stable!)")
    print(f"  Adversarial + JPEG:   {sim_robust_jpeg[0]:.3f}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
