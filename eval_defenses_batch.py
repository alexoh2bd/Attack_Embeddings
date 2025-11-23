#!/usr/bin/env python3
"""eval_defenses_batch.py

Evaluate all defense models on the pre-attacked MMEB-eval dataset.

Usage:
    python eval_defenses_batch.py --attacked_dir mmeb_eval_attacked
"""

import os
import argparse
import json
import torch
import torch.nn.functional as F
from PIL import Image
import clip
import numpy as np
from tqdm import tqdm

def load_model(model_type, checkpoint_path=None, device='cuda'):
    """Load a CLIP model (original or robust)."""
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load robust model weights
        torch.serialization.add_safe_globals([argparse.Namespace])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    model.float()
    return model, preprocess

def evaluate_model(model, preprocess, image_path, text, device='cuda'):
    """Evaluate a single image-text pair."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    text_tokens = clip.tokenize([text], truncate=True).to(device)
    
    with torch.no_grad():
        img_feat = model.encode_image(image_tensor)
        txt_feat = model.encode_text(text_tokens)
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        similarity = (img_feat @ txt_feat.T).item()
    
    return similarity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacked_dir", type=str, default="mmeb_eval_attacked")
    parser.add_argument("--base_img_path", type=str, default="mmeb_images")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load metadata
    metadata_path = os.path.join(args.attacked_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Loaded {len(metadata)} attacked samples from {args.attacked_dir}")
    
    # Define models to evaluate
    models_to_eval = [
        ("Original", None),
        ("MAT", "robust_clip_mat_v2/model_best.pt"),
        ("PGD", "robust_clip_pgd/model_best.pt"),
    ]
    
    results = {}
    
    for model_name, checkpoint_path in models_to_eval:
        # Check if model exists
        if checkpoint_path and not os.path.exists(checkpoint_path):
            print(f"⚠️  {model_name} model not found at {checkpoint_path}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name} model...")
        print(f"{'='*60}")
        
        model, preprocess = load_model(model_name, checkpoint_path, device)
        
        clean_sims = []
        adv_sims = []
        # Dictionary to store sims for each quality
        jpeg_sims = {} 
        
        for item in tqdm(metadata, desc=f"{model_name}"):
            # Evaluate on clean image
            clean_img_path = os.path.join(args.base_img_path, item['original_path'])
            if os.path.exists(clean_img_path):
                clean_sim = evaluate_model(model, preprocess, clean_img_path, item['text'], device)
                clean_sims.append(clean_sim)
            
            # Evaluate on adversarial image
            adv_img_path = os.path.join(args.attacked_dir, item['adv_path'])
            adv_sim = evaluate_model(model, preprocess, adv_img_path, item['text'], device)
            adv_sims.append(adv_sim)
            
            # Evaluate on JPEG-defended images
            if 'jpeg_paths' in item:
                for q, path in item['jpeg_paths'].items():
                    if q not in jpeg_sims:
                        jpeg_sims[q] = []
                    
                    jpeg_img_path = os.path.join(args.attacked_dir, path)
                    if os.path.exists(jpeg_img_path):
                        jpeg_sim = evaluate_model(model, preprocess, jpeg_img_path, item['text'], device)
                        jpeg_sims[q].append(jpeg_sim)
        
        # Calculate metrics
        avg_clean = np.mean(clean_sims) if clean_sims else 0
        avg_adv = np.mean(adv_sims)
        avg_drop = avg_clean - avg_adv
        drop_pct = (avg_drop / avg_clean * 100) if avg_clean > 0 else 0
        
        results[model_name] = {
            "clean": avg_clean,
            "adv": avg_adv,
            "drop": avg_drop,
            "drop_pct": drop_pct,
            "jpeg_results": {}
        }
        
        for q, sims in jpeg_sims.items():
            avg_jpeg = np.mean(sims)
            jpeg_drop = avg_clean - avg_jpeg
            jpeg_drop_pct = (jpeg_drop / avg_clean * 100) if avg_clean > 0 else 0
            results[model_name]["jpeg_results"][q] = {
                "avg": avg_jpeg,
                "drop": jpeg_drop,
                "drop_pct": jpeg_drop_pct
            }
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"DEFENSE COMPARISON TABLE (MMEB-eval)")
    print(f"{'='*80}")
    print(f"\n{'Method':<25} {'Clean':<12} {'Adv':<12} {'Drop':<20} {'Improvement'}")
    print(f"{'-'*80}")
    
    baseline_drop = results.get("Original", {}).get("drop", 0)
    
    # Original (No Defense)
    if "Original" in results:
        r = results["Original"]
        print(f"{'Original (No Defense)':<25} {r['clean']:<12.4f} {r['adv']:<12.4f} {r['drop']:.4f} ({r['drop_pct']:.1f}%)")
    
    # JPEG Defenses
    if "Original" in results:
        r = results["Original"]
        for q in sorted(r.get("jpeg_results", {}).keys(), key=lambda x: int(x), reverse=True):
            res = r["jpeg_results"][q]
            jpeg_improvement = ((baseline_drop - res['drop']) / baseline_drop * 100) if baseline_drop > 0 else 0
            status = "✅" if jpeg_improvement > 0 else "❌"
            print(f"{f'JPEG Defense (q={q})':<25} {r['clean']:<12.4f} {res['avg']:<12.4f} {res['drop']:.4f} ({res['drop_pct']:.1f}%)  {status} {jpeg_improvement:+.1f}%")
    
    # Trained models
    for model_name in ["MAT", "PGD"]:
        if model_name not in results:
            print(f"{model_name:<25} {'N/A':<12} {'N/A':<12} {'N/A':<20} (not trained)")
            continue
        
        r = results[model_name]
        improvement = ((baseline_drop - r['drop']) / baseline_drop * 100) if baseline_drop > 0 else 0
        status = "✅" if improvement > 0 else "❌"
        print(f"{model_name:<25} {r['clean']:<12.4f} {r['adv']:<12.4f} {r['drop']:.4f} ({r['drop_pct']:.1f}%)  {status} {improvement:+.1f}%")
    
    print(f"\n{'='*80}\n")
    
    # Save results
    results_path = os.path.join(args.attacked_dir, "defense_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
