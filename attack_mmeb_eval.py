#!/usr/bin/env python3
"""attack_mmeb_eval.py

Generation augmented by the help of Google Antigravity - model Gemini 3 Pro and Claude Sonnet 4.5 Thinking
Saves attacked images and metadata to mmeb_eval_attacked/ folder.

Usage:
    python attack_mmeb_eval.py --dataset MSCOCO_i2t --eps 0.0314 --steps 20
"""

import os
import argparse
import json
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from datasets import load_dataset
import clip
from tqdm import tqdm

def save_tensor_image(x, out_path):
    """Save tensor as PNG image."""
    if x.dim() == 4:
        x = x[0]
    arr = (x.detach().cpu().permute(1,2,0).numpy() * 255.0).clip(0,255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(out_path)

def pgd_attack(model, image_tensor_01, text_tokens, normalize_fn, epsilon=0.0314, alpha=0.0078, steps=20, device='cuda'):
    """
    PGD attack - exact copy from attack_clip.py
    """
    x0 = image_tensor_01.clone().detach()
    
    # Random initialization
    x_adv = x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(0, 1)
    
    for _ in range(steps):
        x_adv.requires_grad_(True)
        
        # Normalize for CLIP
        x_input = normalize_fn(x_adv)
        
        # Get embeddings
        img_emb = model.encode_image(x_input)
        img_emb = F.normalize(img_emb, dim=-1)
        
        text_emb = model.encode_text(text_tokens)
        text_emb = F.normalize(text_emb, dim=-1)
        
        # Similarity
        sim = img_emb @ text_emb.T  # (B, L)
        
        # Loss: NEGATIVE similarity (we want to minimize it)
        loss = -sim.mean()
        
        # Backward
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = x_adv.grad
            # PGD step
            x_adv = x_adv + alpha * grad.sign()
            # Project to epsilon ball
            x_adv = torch.max(torch.min(x_adv, x0 + epsilon), x0 - epsilon)
            x_adv = x_adv.clamp(0, 1)
        
        x_adv.requires_grad_(True)
    
    return x_adv.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MSCOCO_i2t")
    parser.add_argument("--eps", type=float, default=0.0314, help="Epsilon (8/255)")
    parser.add_argument("--alpha", type=float, default=0.0078, help="Step size (2/255)")
    parser.add_argument("--steps", type=int, default=20, help="PGD steps")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to attack")
    parser.add_argument("--output_dir", type=str, default="mmeb_eval_attacked")
    parser.add_argument("--base_img_path", type=str, default="mmeb_images")
    parser.add_argument("--model", type=str, default="ViT-L/14")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading CLIP model: {args.model}")
    model, preprocess = clip.load(args.model, device=device, jit=False)
    model.eval()
    model.float()
    
    # Extract CLIP normalization
    CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    normalize_fn = lambda x: (x - CLIP_MEAN) / CLIP_STD
    
    print(f"Loading MMEB-eval dataset: {args.dataset}")
    ds = load_dataset("TIGER-Lab/MMEB-eval", args.dataset, split="test")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Metadata list
    metadata = []
    
    print(f"\nGenerating adversarial examples...")
    print(f"Epsilon: {args.eps:.4f}, Alpha: {args.alpha:.4f}, Steps: {args.steps}")
    
    successful_attacks = 0
    total_samples = 0
    
    num_to_process = args.max_samples if args.max_samples else len(ds)
    
    for idx in tqdm(range(num_to_process)):
        item = ds[idx]
        img_path = item['qry_img_path']
        text = item['tgt_text']
        if isinstance(text, list):
            text = text[0]
        
        full_img_path = os.path.join(args.base_img_path, img_path)
        
        # Skip if image doesn't exist
        if not os.path.exists(full_img_path):
            continue
        
        try:
            # Load image as [0,1] tensor (no normalization yet)
            image = Image.open(full_img_path).convert("RGB").resize((224, 224))
            image_tensor_01 = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image_tensor_01 = image_tensor_01.unsqueeze(0).to(device)
            text_tokens = clip.tokenize([text], truncate=True).to(device)
            
            # Check clean accuracy
            with torch.no_grad():
                clean_normalized = normalize_fn(image_tensor_01)
                img_feat = model.encode_image(clean_normalized)
                txt_feat = model.encode_text(text_tokens)
                img_feat = F.normalize(img_feat, dim=-1)
                txt_feat = F.normalize(txt_feat, dim=-1)
                clean_sim = (img_feat @ txt_feat.T).item()
            
            # Generate adversarial example
            adv_tensor = pgd_attack(
                model, image_tensor_01, text_tokens, normalize_fn,
                epsilon=args.eps, alpha=args.alpha, steps=args.steps, device=device
            )
            
            # Check adversarial accuracy
            with torch.no_grad():
                adv_normalized = normalize_fn(adv_tensor)
                adv_feat = model.encode_image(adv_normalized)
                adv_feat = F.normalize(adv_feat, dim=-1)
                adv_sim = (adv_feat @ txt_feat.T).item()
            
            # Save adversarial image
            adv_filename = f"adv_{idx:05d}.png"
            adv_path = os.path.join(args.output_dir, adv_filename)
            save_tensor_image(adv_tensor, adv_path)
            
            # Save metadata
            metadata.append({
                "idx": idx,
                "original_path": img_path,
                "adv_path": adv_filename,
                "text": text,
                "clean_similarity": float(clean_sim),
                "adv_similarity": float(adv_sim),
                "drop": float(clean_sim - adv_sim)
            })
            
            total_samples += 1
            if adv_sim < clean_sim * 0.5:  # Consider it successful if similarity drops by 50%+
                successful_attacks += 1
                
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue
    
    # Save metadata
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Attack Generation Complete!")
    print(f"{'='*60}")
    print(f"Total samples processed: {total_samples}")
    print(f"Successful attacks (>50% drop): {successful_attacks} ({successful_attacks/total_samples*100:.1f}%)")
    print(f"Output directory: {args.output_dir}/")
    print(f"Metadata saved to: {metadata_path}")
    
    # Calculate average metrics
    avg_clean = np.mean([m['clean_similarity'] for m in metadata])
    avg_adv = np.mean([m['adv_similarity'] for m in metadata])
    avg_drop = np.mean([m['drop'] for m in metadata])
    
    print(f"\nAverage Metrics:")
    print(f"  Clean Similarity: {avg_clean:.4f}")
    print(f"  Adversarial Similarity: {avg_adv:.4f}")
    print(f"  Average Drop: {avg_drop:.4f} ({avg_drop/avg_clean*100:.1f}%)")

if __name__ == "__main__":
    main()
