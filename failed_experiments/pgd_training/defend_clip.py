#!/usr/bin/env python3
"""defend_clip.py

Implements defenses against adversarial attacks on CLIP:
1. Inference-time defenses (Data Augmentation / Purification):
   - JPEG Compression
   - Random Resizing
   - Test-Time Augmentation (TTA) / Combined DA
2. Adversarial Training (FAT/MAT) logic (requires dataset, demo on single image provided).

Usage:
    # Inference Defense (Eval)
    python defend_clip.py --mode eval --img adv.png --labels "white dog" "cat" "car"

    # Adversarial Training (Demo)
    python defend_clip.py --mode train --img sample_image.jpg --labels "white dog" --epochs 5
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, RandomHorizontalFlip, ToTensor, ToPILImage
import io

# Import from attack_clip to reuse logic
from attack_clip import load_clip_model, topk_for_labels, build_label_embs, pgd_linf_attack, CLIP_MEAN, CLIP_STD

# --- Inference Defenses (Purification) ---

class JPEGDefense:
    def __init__(self, quality=75):
        self.quality = quality

    def __call__(self, img_tensor):
        # Input: (1, C, H, W) tensor in [0, 1]
        # Output: (1, C, H, W) tensor in [0, 1] after JPEG compression
        img_pil = ToPILImage()(img_tensor.squeeze(0).cpu())
        buffer = io.BytesIO()
        img_pil.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        img_jpeg = Image.open(buffer).convert("RGB")
        return ToTensor()(img_jpeg).unsqueeze(0).to(img_tensor.device)

class RandomResizeDefense:
    def __init__(self, size=224, scale=(0.8, 1.0)):
        self.transform = RandomResizedCrop(size, scale=scale)

    def __call__(self, img_tensor):
        return self.transform(img_tensor)

# --- Adversarial Training Logic ---

class AdversarialTrainer:
    def __init__(self, model, preprocess, device, defense_mode="FAT"):
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.defense_mode = defense_mode # FAT or MAT
        
        # Prepare normalization function for attack
        dtype = next(model.parameters()).dtype
        self.mean = torch.tensor(CLIP_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(CLIP_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
        self.normalize_fn = lambda x: (x - self.mean) / self.std
        
        # Optimizer (only training visual encoder for demo)
        self.optimizer = optim.AdamW(self.model.visual.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()

    def train_step(self, images, texts, text_embs):
        # images: (B, C, H, W) raw tensors [0,1]
        # texts: tokenized text
        # text_embs: precomputed embeddings for attack guidance
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Generate Adversarial Examples
        # Note: In real training, we'd generate on the fly. Here we use the PGD function.
        # We assume batch size 1 for this demo script simplicity, but logic holds.
        
        # For MAT, we mix clean and adv. For FAT, only adv.
        
        adv_images_list = []
        clean_images_list = []
        
        for i in range(images.size(0)):
            img = images[i].unsqueeze(0)
            
            # Generate attack
            # We need text_embs for the specific label of this image for untargeted attack (reduce sim)
            # But standard AT usually tries to MAINTAIN similarity to correct label under attack.
            # The attack_clip.py does UNTARGETED attack (reduce sim to ground truth).
            # So we generate an example that minimizes similarity to GT, then train to MAXIMIZE it.
            
            # PGD Attack (Untargeted)
            adv_img, _ = pgd_linf_attack(
                self.model, img, self.normalize_fn, text_embs[i].unsqueeze(0), self.device,
                eps=4.0/255.0, iters=3, restarts=1 # Fast attack for training
            )
            adv_images_list.append(adv_img)
            clean_images_list.append(img)
            
        adv_batch = torch.cat(adv_images_list, dim=0)
        clean_batch = torch.cat(clean_images_list, dim=0)
        
        if self.defense_mode == "MAT":
            train_batch = torch.cat([clean_batch, adv_batch], dim=0)
            # Duplicate texts for MAT
            train_texts = torch.cat([texts, texts], dim=0)
        else: # FAT
            train_batch = adv_batch
            train_texts = texts

        # Normalize for model input
        train_batch_norm = self.normalize_fn(train_batch)
        
        # Forward pass
        image_logits, text_logits = self.model(train_batch_norm, train_texts)
        
        # Create ground truth labels (contrastive loss)
        # CLIP loss is symmetric CE
        ground_truth = torch.arange(len(train_batch), dtype=torch.long, device=self.device)
        
        loss = (self.loss_img(image_logits, ground_truth) + self.loss_txt(text_logits, ground_truth)) / 2
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["eval", "train"], help="Mode: eval (inference defense) or train (adversarial training demo)")
    parser.add_argument("--img", type=str, required=True, help="Input image path")
    parser.add_argument("--labels", nargs="+", required=True, help="Labels (first one is ground truth)")
    parser.add_argument("--model", type=str, default="ViT-L/14", help="CLIP model name")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (for train mode)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model {args.model} on {device}...")
    model, preprocess = load_clip_model(args.model, device)
    dtype = next(model.parameters()).dtype
    
    # Load image
    pil_img = Image.open(args.img).convert("RGB")
    
    # Prepare prompts
    templates = ["a photo of a {}"]
    prompts = [t.format(lbl) for lbl in args.labels for t in templates]
    idx_map = [i for i in range(len(args.labels)) for _ in templates]
    
    # Label embeddings
    label_embs = build_label_embs(model, prompts, idx_map, device, dtype)
    
    # Preprocessing setup (no norm for raw access)
    if isinstance(preprocess.transforms[-1], Normalize):
        preprocess_no_norm = Compose(preprocess.transforms[:-1])
    else:
        preprocess_no_norm = Compose(preprocess.transforms[:-1])
        
    mean = torch.tensor(CLIP_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    normalize_fn = lambda x: (x - mean) / std

    if args.mode == "eval":
        print("\n--- Inference Defense Evaluation ---")
        img_tensor = preprocess_no_norm(pil_img).unsqueeze(0).to(device, dtype=dtype)
        
        # 1. No Defense
        print("\n[No Defense]")
        vals, _, _ = topk_for_labels(model, normalize_fn(img_tensor), label_embs, args.labels)
        print(f"Top-1: {vals[0][0]} ({vals[0][1]:.3f})")
        
        # 2. JPEG Defense
        print("\n[JPEG Defense (Quality=75)]")
        jpeg_def = JPEGDefense(quality=75)
        img_jpeg = jpeg_def(img_tensor)
        vals, _, _ = topk_for_labels(model, normalize_fn(img_jpeg), label_embs, args.labels)
        print(f"Top-1: {vals[0][0]} ({vals[0][1]:.3f})")
        
        # 3. Random Resize (Simulated TTA)
        print("\n[Random Resize Defense (1 crop)]")
        resize_def = RandomResizeDefense(size=224)
        # Need to resize to 224 for CLIP if we crop, but here we just scale
        # Actually CLIP preprocess does resize/crop. Let's just use a random crop pipeline.
        # We'll use the TTA logic from attack_clip but just 1 random crop to simulate "Data Augmentation" defense
        # Actually, let's use the full TTA as "Combined Data Augmentation" defense
        
        print("\n[Combined Data Augmentation (TTA) Defense]")
        # We use the image_tta_emb function which does multiple crops/flips
        # This is effectively "CDA" at inference time.
        from attack_clip import image_tta_emb
        tta_emb = image_tta_emb(model, preprocess, pil_img, device, dtype=dtype)
        
        # Compute similarity manually
        with torch.no_grad():
            label_embs_cast = label_embs.to(dtype=tta_emb.dtype)
            cos = (tta_emb @ label_embs_cast.T).squeeze(0)
        
        vals = list(zip(args.labels, cos.detach().cpu().numpy()))
        vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)
        print(f"Top-1: {vals_sorted[0][0]} ({vals_sorted[0][1]:.3f})")

    elif args.mode == "train":
        print("\n--- Adversarial Training Demo (FAT) ---")
        print("Note: Training on a single image is for demonstration only and will overfit.")
        
        trainer = AdversarialTrainer(model, preprocess, device, defense_mode="FAT")
        
        # Prepare data for training loop
        # We need tokenized text for the ground truth label
        import clip
        gt_text = clip.tokenize([f"a photo of a {args.labels[0]}"]).to(device)
        
        # We use the raw tensor
        img_tensor = preprocess_no_norm(pil_img).unsqueeze(0).to(device, dtype=dtype)
        
        # GT embedding for attack guidance
        gt_emb = label_embs[0].unsqueeze(0)
        
        for epoch in range(args.epochs):
            loss = trainer.train_step(img_tensor, gt_text, gt_emb)
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")
            
        print("\nTraining complete. Evaluating on original image...")
        vals, _, _ = topk_for_labels(model, normalize_fn(img_tensor), label_embs, args.labels)
        print(f"Top-1: {vals[0][0]} ({vals[0][1]:.3f})")
        
        # Save model (optional)
        # torch.save(model.state_dict(), "clip_robust.pt")

if __name__ == "__main__":
    main()
