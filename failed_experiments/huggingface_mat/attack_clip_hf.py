#!/usr/bin/env python3
"""attack_clip_hf.py

PGD Attack using Hugging Face CLIP model (openai/clip-vit-base-patch32).
This matches the architecture of our robust model for fair comparison.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import os

def pgd_attack(model, image_tensor, text_embeddings, epsilon=0.0314, alpha=0.005, steps=20):
    """
    PGD Attack for Hugging Face CLIP.
    Untargeted attack: Minimize similarity to ground truth text.
    """
    # Clone image and enable gradients
    adv_image = image_tensor.clone().detach()
    adv_image.requires_grad = True
    
    for i in range(steps):
        # Forward pass
        image_features = model.get_image_features(pixel_values=adv_image)
        image_features = F.normalize(image_features, dim=-1)
        
        # Calculate similarity
        similarity = (image_features @ text_embeddings.T).squeeze()
        
        # We want to MINIMIZE similarity to the correct label (untargeted attack)
        # So we maximize -similarity (or minimize similarity)
        loss = similarity.mean()
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update image
        grad = adv_image.grad.data
        adv_image.data = adv_image.data - alpha * grad.sign()
        
        # Projection (L-inf constraint)
        eta = torch.clamp(adv_image.data - image_tensor.data, -epsilon, epsilon)
        adv_image.data = torch.clamp(image_tensor.data + eta, 0.0, 1.0) # Assuming normalized 0-1 input
        
        adv_image.grad.zero_()
        
    return adv_image.detach()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=str, default="adv_hf.png")
    parser.add_argument("--eps", type=float, default=0.0628)  # Increased from 0.0314
    parser.add_argument("--steps", type=int, default=50)      # Increased from 20
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"
    
    print(f"\n{'='*60}")
    print(f"Attacking Hugging Face CLIP: {model_name}")
    print(f"{'='*60}\n")
    
    # Load model
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    # Load image
    image = Image.open(args.img).convert("RGB")
    
    # Process inputs
    inputs = processor(text=args.labels, images=image, return_tensors="pt", padding=True)
    pixel_values = inputs['pixel_values'].to(device)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Get text embeddings (target)
    with torch.no_grad():
        text_features = model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        text_features = F.normalize(text_features, dim=-1)
    
    # Initial prediction
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=pixel_values)
        image_features = F.normalize(image_features, dim=-1)
        sim_clean = (image_features @ text_features.T).squeeze()
        
    print("Clean Image Similarities:")
    for lbl, s in zip(args.labels, sim_clean.cpu().numpy()):
        print(f"  {lbl:20s}: {s:.4f}")
        
    # Run Attack
    print(f"\nRunning PGD Attack (eps={args.eps}, steps={args.steps})...")
    adv_pixel_values = pgd_attack(model, pixel_values, text_features, epsilon=args.eps, steps=args.steps)
    
    # Final prediction
    with torch.no_grad():
        adv_features = model.get_image_features(pixel_values=adv_pixel_values)
        adv_features = F.normalize(adv_features, dim=-1)
        sim_adv = (adv_features @ text_features.T).squeeze()
        
    print("\nAdversarial Image Similarities:")
    for lbl, s in zip(args.labels, sim_adv.cpu().numpy()):
        print(f"  {lbl:20s}: {s:.4f}")
        
    # Save adversarial image
    # Note: HF processor normalizes images. We need to un-normalize to save.
    # But for simplicity, we'll just save the tensor directly if possible or approximation
    # Actually, let's just use the robust eval script which handles loading
    print(f"\nSaving adversarial image to {args.output}...")
    
    # Approximate reconstruction (since we don't have exact un-normalize params handy easily)
    # Standard ImageNet mean/std
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)
    
    adv_img_denorm = adv_pixel_values * std + mean
    adv_img_denorm = torch.clamp(adv_img_denorm, 0, 1)
    
    adv_img_np = adv_img_denorm.squeeze().permute(1, 2, 0).cpu().numpy()
    adv_img_pil = Image.fromarray((adv_img_np * 255).astype(np.uint8))
    adv_img_pil.save(args.output)
    print("Done.")

if __name__ == "__main__":
    main()
