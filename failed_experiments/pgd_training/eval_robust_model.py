#!/usr/bin/env python3
"""eval_robust_model.py

Evaluate the MAT-trained robust model against attacks.
Compares original CLIP vs robust model.

Usage:
    python eval_robust_model.py --img sample_image.jpg --labels "white dog" "cat" "car"
"""

import argparse
import torch
from PIL import Image
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from attack_clip import load_clip_model, build_label_embs, topk_for_labels, CLIP_MEAN, CLIP_STD, pgd_linf_attack
from torchvision.transforms import Compose, Normalize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Image path")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels to test")
    parser.add_argument("--robust_model", type=str, default="robust_clip_mat", help="Path to robust model")
    parser.add_argument("--gen_attack", action="store_true", help="Generate new adversarial image")
    parser.add_argument("--eps", type=float, default=0.0314, help="Attack epsilon")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print("ROBUSTNESS EVALUATION: Original vs MAT-trained Model")
    print(f"{'='*60}\n")
    
    # Load original model
    print("Loading ORIGINAL model...")
    model_orig, preprocess_orig = load_clip_model("ViT-L/14", device)
    dtype_orig = next(model_orig.parameters()).dtype
    
    # Load robust model
    print(f"Loading ROBUST model from {args.robust_model}...")
    from transformers import CLIPModel, CLIPProcessor
    model_robust = CLIPModel.from_pretrained(args.robust_model).to(device)
    preprocess_robust = CLIPProcessor.from_pretrained(args.robust_model)
    dtype_robust = next(model_robust.parameters()).dtype
    model_robust.eval()
    
    # Load image
    pil_img = Image.open(args.img).convert("RGB")
    
    # Prepare preprocessing
    if isinstance(preprocess_orig.transforms[-1], Normalize):
        preprocess_no_norm = Compose(preprocess_orig.transforms[:-1])
    else:
        preprocess_no_norm = Compose(preprocess_orig.transforms[:-1])
    
    mean_orig = torch.tensor(CLIP_MEAN, device=device, dtype=dtype_orig).view(1, 3, 1, 1)
    std_orig = torch.tensor(CLIP_STD, device=device, dtype=dtype_orig).view(1, 3, 1, 1)
    normalize_fn_orig = lambda x: (x - mean_orig) / std_orig
    
    # Build label embeddings
    templates = ["a photo of a {}"]
    prompts = [t.format(lbl) for lbl in args.labels for t in templates]
    idx_map = [i for i in range(len(args.labels)) for _ in templates]
    
    # Build label embeddings for original model (OpenAI CLIP)
    label_embs_orig = build_label_embs(model_orig, prompts, idx_map, device, dtype_orig)
    
    # Build label embeddings for robust model (HuggingFace CLIP)
    # HuggingFace uses get_text_features instead of encode_text
    import clip as openai_clip
    text_tokens_hf = preprocess_robust(text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    text_tokens_hf = {k: v.to(device) for k, v in text_tokens_hf.items()}
    
    with torch.no_grad():
        txt_feats = model_robust.get_text_features(**text_tokens_hf)
        txt_feats = txt_feats.to(dtype=dtype_robust)
        txt_feats = torch.nn.functional.normalize(txt_feats, dim=-1)
    
    # Average per label
    num_labels = max(idx_map) + 1 if len(idx_map) > 0 else 0
    label_embs_list = []
    for i in range(num_labels):
        inds = [j for j, k in enumerate(idx_map) if k == i]
        if len(inds) == 0:
            vec = torch.zeros((1, txt_feats.size(1)), device=device, dtype=dtype_robust)
        else:
            vec = txt_feats[inds].mean(dim=0, keepdim=True)
            vec = torch.nn.functional.normalize(vec, dim=-1)
        label_embs_list.append(vec)
    label_embs_robust = torch.cat(label_embs_list, dim=0)
    
    # Test on clean image
    print("\n" + "="*60)
    print("TEST 1: Clean Image")
    print("="*60)
    
    img_tensor = preprocess_no_norm(pil_img).unsqueeze(0).to(device, dtype=dtype_orig)
    
    print("\nOriginal Model:")
    vals_orig_clean, _, _ = topk_for_labels(model_orig, normalize_fn_orig(img_tensor), label_embs_orig, args.labels)
    for lbl, c, p in vals_orig_clean:
        print(f"  {lbl:20s}\t{c:6.3f}")
    
    print("\nRobust Model:")
    vals_robust_clean, _, _ = topk_for_labels(model_robust, normalize_fn_orig(img_tensor), label_embs_robust, args.labels)
    for lbl, c, p in vals_robust_clean:
        print(f"  {lbl:20s}\t{c:6.3f}")
    
    # Generate or load adversarial image
    if args.gen_attack:
        print(f"\n{'='*60}")
        print(f"Generating PGD attack (eps={args.eps})...")
        print(f"{'='*60}\n")
        
        adv_img, _ = pgd_linf_attack(
            model_orig, img_tensor, normalize_fn_orig, label_embs_orig, device,
            eps=args.eps, iters=20, restarts=5
        )
        # Save
        from attack_clip import save_tensor_image
        save_tensor_image(adv_img, "adv_test.png")
        print("Saved adversarial image to adv_test.png")
    else:
        # Load existing
        if os.path.exists("adv.png"):
            adv_pil = Image.open("adv.png").convert("RGB")
            adv_img = preprocess_no_norm(adv_pil).unsqueeze(0).to(device, dtype=dtype_orig)
        else:
            print("No adversarial image found. Use --gen_attack to generate one.")
            return
    
    # Test on adversarial image
    print("\n" + "="*60)
    print("TEST 2: Adversarial Image (PGD Attack)")
    print("="*60)
    
    print("\nOriginal Model:")
    vals_orig_adv, _, _ = topk_for_labels(model_orig, normalize_fn_orig(adv_img), label_embs_orig, args.labels)
    for lbl, c, p in vals_orig_adv:
        print(f"  {lbl:20s}\t{c:6.3f}")
    
    print("\nRobust Model:")
    vals_robust_adv, _, _ = topk_for_labels(model_robust, normalize_fn_orig(adv_img), label_embs_robust, args.labels)
    for lbl, c, p in vals_robust_adv:
        print(f"  {lbl:20s}\t{c:6.3f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    orig_clean_sim = vals_orig_clean[0][1]
    robust_clean_sim = vals_robust_clean[0][1]
    orig_adv_sim = vals_orig_adv[0][1]
    robust_adv_sim = vals_robust_adv[0][1]
    
    print(f"\nClean Image (top label: '{args.labels[0]}'):")
    print(f"  Original: {orig_clean_sim:.3f}")
    print(f"  Robust:   {robust_clean_sim:.3f}")
    print(f"  Change:   {robust_clean_sim - orig_clean_sim:+.3f}")
    
    print(f"\nAdversarial Image (top label: '{args.labels[0]}'):")
    print(f"  Original: {orig_adv_sim:.3f}")
    print(f"  Robust:   {robust_adv_sim:.3f}")
    print(f"  Change:   {robust_adv_sim - orig_adv_sim:+.3f}")
    
    print(f"\nRobustness Improvement:")
    orig_drop = orig_clean_sim - orig_adv_sim
    robust_drop = robust_clean_sim - robust_adv_sim
    improvement = orig_drop - robust_drop
    print(f"  Original model drop: {orig_drop:.3f}")
    print(f"  Robust model drop:   {robust_drop:.3f}")
    print(f"  Improvement:         {improvement:.3f} ({improvement/orig_drop*100:.1f}%)")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
