#!/usr/bin/env python3
"""eval_robust_simple.py

Simple evaluation of MAT-trained robust model.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import argparse
import io

def evaluate_model(model, processor, img_path, labels, device):
    """Evaluate a CLIP model on an image."""
    # Load image
    img = Image.open(img_path).convert("RGB")
    
    # Prepare inputs
    texts = [f"a photo of a {label}" for label in labels]
    inputs = processor(text=texts, images=img, return_tensors="pt", padding=True).to(device)
    
    # Get features
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        
        # Normalize
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Compute similarity
        similarity = (image_embeds @ text_embeds.T).squeeze(0)
    
    return similarity.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--adv_img", type=str, default="adv.png")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--robust_model", type=str, default="robust_clip_mat")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print("ROBUSTNESS EVALUATION")
    print(f"{'='*60}\n")
    
    # Load original model
    print("Loading ORIGINAL model...")
    model_orig = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor_orig = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model_orig.eval()
    
    # Load robust model
    print(f"Loading ROBUST model from {args.robust_model}...")
    model_robust = CLIPModel.from_pretrained(args.robust_model).to(device)
    processor_robust = CLIPProcessor.from_pretrained(args.robust_model)
    model_robust.eval()
    
    # Evaluate on clean image
    print(f"\n{'='*60}")
    print("TEST 1: Clean Image")
    print(f"{'='*60}\n")
    
    sim_orig_clean = evaluate_model(model_orig, processor_orig, args.img, args.labels, device)
    sim_robust_clean = evaluate_model(model_robust, processor_robust, args.img, args.labels, device)
    
    print("Original Model:")
    for label, sim in zip(args.labels, sim_orig_clean):
        print(f"  {label:20s}\t{sim:6.3f}")
    
    print("\nRobust Model:")
    for label, sim in zip(args.labels, sim_robust_clean):
        print(f"  {label:20s}\t{sim:6.3f}")
    
    # Evaluate on adversarial image
    print(f"\n{'='*60}")
    print("TEST 2: Adversarial Image")
    print(f"{'='*60}\n")
    
    sim_orig_adv = evaluate_model(model_orig, processor_orig, args.adv_img, args.labels, device)
    sim_robust_adv = evaluate_model(model_robust, processor_robust, args.adv_img, args.labels, device)
    
    print("Original Model:")
    for label, sim in zip(args.labels, sim_orig_adv):
        print(f"  {label:20s}\t{sim:6.3f}")
    
    print("\nRobust Model:")
    for label, sim in zip(args.labels, sim_robust_adv):
        print(f"  {label:20s}\t{sim:6.3f}")
    
    # Evaluate on adversarial image with JPEG defense
    print(f"\n{'='*60}")
    print("TEST 3: Adversarial Image + JPEG Defense")
    print(f"{'='*60}\n")
    
    # Apply JPEG compression
    adv_img = Image.open(args.adv_img).convert("RGB")
    buffer = io.BytesIO()
    adv_img.save(buffer, format="JPEG", quality=75)
    buffer.seek(0)
    adv_img_jpeg = Image.open(buffer).convert("RGB")
    adv_img_jpeg.save("adv_jpeg.png")  # Save for reference
    
    sim_orig_jpeg = evaluate_model(model_orig, processor_orig, "adv_jpeg.png", args.labels, device)
    sim_robust_jpeg = evaluate_model(model_robust, processor_robust, "adv_jpeg.png", args.labels, device)
    
    print("Original Model + JPEG:")
    for label, sim in zip(args.labels, sim_orig_jpeg):
        print(f"  {label:20s}\t{sim:6.3f}")
    
    print("\nRobust Model + JPEG:")
    for label, sim in zip(args.labels, sim_robust_jpeg):
        print(f"  {label:20s}\t{sim:6.3f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    gt_label = args.labels[0]
    orig_clean = sim_orig_clean[0]
    robust_clean = sim_robust_clean[0]
    orig_adv = sim_orig_adv[0]
    robust_adv = sim_robust_adv[0]
    orig_jpeg = sim_orig_jpeg[0]
    robust_jpeg = sim_robust_jpeg[0]
    
    print(f"Clean Image ('{gt_label}'):")
    print(f"  Original: {orig_clean:.3f}")
    print(f"  Robust:   {robust_clean:.3f}")
    
    print(f"\nAdversarial Image ('{gt_label}'):")
    print(f"  Original:        {orig_adv:.3f}")
    print(f"  Robust:          {robust_adv:.3f}")
    print(f"  Orig + JPEG:     {orig_jpeg:.3f}")
    print(f"  Robust + JPEG:   {robust_jpeg:.3f}")
    
    print(f"\nDefense Effectiveness:")
    print(f"  MAT Training:    {robust_adv:.3f} (change: {robust_adv - orig_adv:+.3f})")
    print(f"  JPEG Defense:    {orig_jpeg:.3f} (change: {orig_jpeg - orig_adv:+.3f})")
    print(f"  Combined:        {robust_jpeg:.3f} (change: {robust_jpeg - orig_adv:+.3f})")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
