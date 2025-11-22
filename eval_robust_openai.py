#!/usr/bin/env python3
"""eval_robust_openai.py

Evaluate the robust OpenAI CLIP model (ViT-L/14) against PGD attacks.
Compares original vs. trained robust model.

Usage:
    python eval_robust_openai.py --img sample_image.jpg --adv_img adv.png --labels "white dog" "cat" "car"
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import clip
import os

def load_original_model(device="cuda"):
    """Load the original OpenAI CLIP ViT-L/14 model."""
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    model.eval()
    model.float()
    return model, preprocess

def load_robust_model(checkpoint_path, device="cuda"):
    """Load the trained robust OpenAI CLIP model."""
    # Load original architecture
    model, preprocess = clip.load("ViT-L/14", device=device, jit=False)
    
    # Load trained weights
    checkpoint = torch.load(os.path.join(checkpoint_path, "model.pt"), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.float()
    
    return model, preprocess

def evaluate_model(model, preprocess, image_path, labels, device="cuda"):
    """Evaluate a model on an image with given labels."""
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Tokenize text
    text_tokens = clip.tokenize(labels).to(device)
    
    # Get features
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)
        
        # Normalize
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # Compute similarity
        similarity = (image_features @ text_features.T).squeeze()
    
    return similarity.cpu().numpy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Clean image path")
    parser.add_argument("--adv_img", type=str, required=True, help="Adversarial image path")
    parser.add_argument("--labels", nargs="+", required=True, help="Text labels")
    parser.add_argument("--robust_path", type=str, default="robust_clip_openai", help="Path to robust model")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print("OpenAI CLIP (ViT-L/14) Robustness Evaluation")
    print(f"{'='*60}\n")
    
    # Load models
    print("Loading original model...")
    original_model, original_preprocess = load_original_model(device)
    
    print("\n" + "="*60)
    print("CLEAN IMAGE EVALUATION")
    print("="*60)
    
    # Evaluate on clean image
    print("\nOriginal Model:")
    original_clean = evaluate_model(original_model, original_preprocess, args.img, args.labels, device)
    for label, sim in zip(args.labels, original_clean):
        print(f"  {label:20s}: {sim:.4f}")
    
    print("\n" + "="*60)
    print("ADVERSARIAL IMAGE EVALUATION")
    print("="*60)
    
    # Evaluate on adversarial image
    print("\nOriginal Model:")
    original_adv = evaluate_model(original_model, original_preprocess, args.adv_img, args.labels, device)
    for label, sim in zip(args.labels, original_adv):
        print(f"  {label:20s}: {sim:.4f}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Calculate drops
    original_drop = original_clean[0] - original_adv[0]
    
    print(f"\nTarget Label: '{args.labels[0]}'")
    print(f"\n{'='*60}")
    print("DEFENSE COMPARISON TABLE")
    print("="*60)
    print(f"\n{'Method':<25} {'Clean':<10} {'Adv':<10} {'Drop':<15} {'Status'}")
    print("-" * 70)
    
    # Original (No Defense)
    print(f"{'Original (No Defense)':<25} {original_clean[0]:<10.4f} {original_adv[0]:<10.4f} {original_drop:.4f} ({original_drop/original_clean[0]*100:.1f}%)")
    
    # JPEG Defense (if adv_jpeg exists)
    jpeg_path = args.adv_img.replace('.png', '_jpeg.png') if not args.adv_img.endswith('_jpeg.png') else args.adv_img
    if os.path.exists(jpeg_path):
        jpeg_sim = evaluate_model(original_model, original_preprocess, jpeg_path, args.labels, device)[0]
        jpeg_drop = original_clean[0] - jpeg_sim
        print(f"{'JPEG Defense (q=75)':<25} {original_clean[0]:<10.4f} {jpeg_sim:<10.4f} {jpeg_drop:.4f} ({jpeg_drop/original_clean[0]*100:.1f}%)")
    else:
        print(f"{'JPEG Defense (q=75)':<25} {'N/A':<10} {'N/A':<10} {'N/A':<15} (run apply_jpeg.py)")
    
    # MAT (Augmentation)
    mat_path = "robust_clip_openai"
    if os.path.exists(os.path.join(mat_path, "model.pt")):
        print(f"Loading MAT model from {mat_path}...")
        try:
            mat_model, mat_preprocess = load_robust_model(mat_path, device)
            mat_clean = evaluate_model(mat_model, mat_preprocess, args.img, args.labels, device)[0]
            mat_adv = evaluate_model(mat_model, mat_preprocess, args.adv_img, args.labels, device)[0]
            
            # Cleanup MAT model to free memory
            del mat_model
            torch.cuda.empty_cache()
            
            mat_drop = mat_clean - mat_adv
            mat_imp = ((original_drop - mat_drop) / original_drop * 100) if original_drop > 0 else 0
            status = "✅" if mat_drop < original_drop else "❌"
            print(f"{'MAT (Augmentation)':<25} {mat_clean:<10.4f} {mat_adv:<10.4f} {mat_drop:.4f} ({mat_drop/mat_clean*100:.1f}%)   {status} {mat_imp:+.1f}%")
        except Exception as e:
            print(f"{'MAT (Augmentation)':<25} ERROR: {str(e)}")
    else:
        print(f"{'MAT (Augmentation)':<25} {'N/A':<10} {'N/A':<10} {'N/A':<15} (not trained)")

    # MAT V2 (First Principles)
    mat_v2_path = "robust_clip_mat_v2"
    # Check for best model first, then fallback to final
    mat_v2_model_path = os.path.join(mat_v2_path, "model_best.pt")
    if not os.path.exists(mat_v2_model_path):
        mat_v2_model_path = os.path.join(mat_v2_path, "model.pt")
        
    if os.path.exists(mat_v2_model_path):
        print(f"Loading MAT V2 model from {mat_v2_path}...")
        try:
            # Helper to load from specific file
            model_v2, preprocess_v2 = clip.load("ViT-L/14", device=device, jit=False)
            
            # Fix for WeightsUnpickler error with argparse.Namespace
            torch.serialization.add_safe_globals([argparse.Namespace])
            
            checkpoint = torch.load(mat_v2_model_path, map_location=device, weights_only=False)
            model_v2.load_state_dict(checkpoint['model_state_dict'])
            model_v2.eval()
            model_v2.float()
            
            mat_v2_clean = evaluate_model(model_v2, preprocess_v2, args.img, args.labels, device)[0]
            mat_v2_adv = evaluate_model(model_v2, preprocess_v2, args.adv_img, args.labels, device)[0]
            
            del model_v2
            torch.cuda.empty_cache()
            
            mat_v2_drop = mat_v2_clean - mat_v2_adv
            mat_v2_imp = ((original_drop - mat_v2_drop) / original_drop * 100) if original_drop > 0 else 0
            status = "✅" if mat_v2_drop < original_drop else "❌"
            print(f"{'MAT V2 (Optimized)':<25} {mat_v2_clean:<10.4f} {mat_v2_adv:<10.4f} {mat_v2_drop:.4f} ({mat_v2_drop/mat_v2_clean*100:.1f}%)   {status} {mat_v2_imp:+.1f}%")
        except Exception as e:
            print(f"{'MAT V2 (Optimized)':<25} ERROR: {str(e)}")
    else:
        print(f"{'MAT V2 (Optimized)':<25} {'N/A':<10} {'N/A':<10} {'N/A':<15} (not trained)")

    # FAT (Fast Adversarial Training)
    fat_path = "robust_clip_fat"
    if os.path.exists(os.path.join(fat_path, "model.pt")):
        print(f"Loading FAT model from {fat_path}...")
        try:
            fat_model, fat_preprocess = load_robust_model(fat_path, device)
            fat_clean = evaluate_model(fat_model, fat_preprocess, args.img, args.labels, device)[0]
            fat_adv = evaluate_model(fat_model, fat_preprocess, args.adv_img, args.labels, device)[0]
            
            # Cleanup FAT model to free memory
            del fat_model
            torch.cuda.empty_cache()
            
            fat_drop = fat_clean - fat_adv
            fat_imp = ((original_drop - fat_drop) / original_drop * 100) if original_drop > 0 else 0
            status = "✅" if fat_drop < original_drop else "❌"
            print(f"{'FAT (FGSM Training)':<25} {fat_clean:<10.4f} {fat_adv:<10.4f} {fat_drop:.4f} ({fat_drop/fat_clean*100:.1f}%)   {status} {fat_imp:+.1f}%")
        except Exception as e:
            print(f"{'FAT (FGSM Training)':<25} ERROR: {str(e)}")
    else:
        print(f"{'FAT (FGSM Training)':<25} {'N/A':<10} {'N/A':<10} {'N/A':<15} (not trained)")
    
    print("\n" + "="*60)
    print("INSIGHTS")
    print("="*60)
    
    print(f"\n📊 Attack Effectiveness:")
    print(f"   Original model drop: {original_drop:.4f} ({original_drop/original_clean[0]*100:.1f}%)")
    
    print(f"\n🛡️  Defense Performance:")
    if os.path.exists(jpeg_path):
        jpeg_recovery = (1 - jpeg_drop/original_drop) * 100 if original_drop > 0 else 0
        print(f"   JPEG recovery: {jpeg_recovery:.1f}%")
    
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    main()
