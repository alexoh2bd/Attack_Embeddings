#!/usr/bin/env python3
"""defend_clip_mat_v2.py

Re-implemented Mixed Adversarial Training (MAT) for OpenAI CLIP ViT-L/14.
Built from first principles for stability, efficiency, and correctness.

Key Features:
- Automatic Mixed Precision (AMP) for memory efficiency and speed.
- PyTorch DataLoader with multi-processing for fast data loading.
- Negative Text Bank to prevent loss collapse on small batches.
- Modular design with robust logging.
- Heavy Data Augmentation (MAT strategy).

Usage:
    python defend_clip_mat_v2.py --batch_size 16 --epochs 1
"""

import os
import sys
import time
import random
import argparse
import copy
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from datasets import load_dataset
import clip

# --- Configuration ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# --- Augmentation Logic ---
class Augmentations:
    """Robust Data Augmentations for MAT."""
    
    @staticmethod
    def apply(img):
        # 50% chance to apply any augmentation chain
        if random.random() > 0.9:
            return img
            
        # Random Color Jitter
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
            img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4))
            img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.5))
            
        # Random Blur
        if random.random() < 0.3:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
            
        # Random Noise
        if random.random() < 0.3:
            img_arr = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 15, img_arr.shape) # Sigma=15
            img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr)
            
        return img

# --- Dataset ---
class CLIPDataset(Dataset):
    def __init__(self, dataset_name, preprocess, base_img_path, max_samples=None):
        self.preprocess = preprocess
        self.base_img_path = base_img_path
        
        log(f"Loading HF Dataset: {dataset_name}...")
        self.ds = load_dataset("TIGER-Lab/MMEB-eval", dataset_name, split="test")
        
        if max_samples:
            self.ds = self.ds.select(range(min(len(self.ds), max_samples)))
            
        self.valid_indices = self._filter_valid_images()
        log(f"Found {len(self.valid_indices)} valid images out of {len(self.ds)}")

    def _filter_valid_images(self):
        """Pre-scan dataset to find valid images."""
        valid = []
        for i in range(len(self.ds)):
            img_path = self.ds[i]['qry_img_path']
            full_path = os.path.join(self.base_img_path, img_path)
            if os.path.exists(full_path):
                valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        item = self.ds[real_idx]
        
        img_path = item['qry_img_path']
        text = item['tgt_text']
        if isinstance(text, list): text = text[0]
        
        full_path = os.path.join(self.base_img_path, img_path)
        
        try:
            image = Image.open(full_path).convert("RGB")
            
            # 1. Clean Image
            clean_tensor = self.preprocess(image)
            
            # 2. Augmented Image (MAT)
            aug_image = Augmentations.apply(image.copy())
            aug_tensor = self.preprocess(aug_image)
            
            return {
                "clean_image": clean_tensor,
                "aug_image": aug_tensor,
                "text": text
            }
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            # Return a dummy item to prevent crash (handled by collate ideally, but simple here)
            return self.__getitem__((idx + 1) % len(self))

# --- Trainer ---
class MATTrainerV2:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        log(f"Initializing Model: {args.model_name}")
        self.model, self.preprocess = clip.load(args.model_name, device=self.device, jit=False)
        self.model.float() # Convert to float32 initially
        
        # Freeze non-visual parameters
        for name, param in self.model.named_parameters():
            if "visual" not in name:
                param.requires_grad = False
                
        self.optimizer = optim.AdamW(
            [p for p in self.model.visual.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.scaler = GradScaler() # For AMP
        self.loss_fn = nn.CrossEntropyLoss()
        
    def get_negative_bank(self, dataset):
        """Create a bank of negative texts for robust loss calculation."""
        log("Creating Negative Text Bank...")
        all_texts = [dataset[i]['text'] for i in range(len(dataset))]
        
        # Sample negatives
        num_neg = min(len(all_texts), 512)
        neg_texts = random.sample(all_texts, num_neg)
        
        with torch.no_grad():
            neg_tokens = clip.tokenize(neg_texts, truncate=True).to(self.device)
            neg_features = self.model.encode_text(neg_tokens)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
            
        log(f"Negative Bank Size: {neg_features.shape}")
        return neg_features.detach()

    def train(self):
        # Setup Data
        dataset = CLIPDataset(
            self.args.dataset, 
            self.preprocess, 
            os.path.join(os.getcwd(), 'mmeb_images/')
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=True,
            drop_last=True
        )
        
        # Negative Bank
        neg_features_bank = self.get_negative_bank(dataset)
        
        log(f"Starting Training for {self.args.epochs} epochs...")
        
        step = 0
        best_loss = float('inf')
        patience = 100
        patience_counter = 0
        
        for epoch in range(self.args.epochs):
            self.model.train()
            
            end = time.time()
            for i, batch in enumerate(dataloader):
                data_time = time.time() - end
                step += 1
                
                clean_imgs = batch['clean_image'].to(self.device)
                aug_imgs = batch['aug_image'].to(self.device)
                texts = clip.tokenize(batch['text'], truncate=True).to(self.device)
                
                self.optimizer.zero_grad()
                
                with autocast():
                    # 1. Encode Texts (Shared)
                    text_features = self.model.encode_text(texts)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    # Combine with Negatives
                    all_text_features = torch.cat([text_features, neg_features_bank], dim=0)
                    
                    # 2. Clean Loss
                    img_features_clean = self.model.encode_image(clean_imgs)
                    img_features_clean = img_features_clean / img_features_clean.norm(dim=-1, keepdim=True)
                    
                    logit_scale = self.model.logit_scale.exp()
                    
                    # Image-to-Text (Batch vs All Texts)
                    logits_clean = logit_scale * img_features_clean @ all_text_features.t()
                    labels = torch.arange(len(clean_imgs)).to(self.device)
                    
                    loss_clean = self.loss_fn(logits_clean, labels)
                    
                    # 3. Augmented Loss
                    img_features_aug = self.model.encode_image(aug_imgs)
                    img_features_aug = img_features_aug / img_features_aug.norm(dim=-1, keepdim=True)
                    
                    logits_aug = logit_scale * img_features_aug @ all_text_features.t()
                    loss_aug = self.loss_fn(logits_aug, labels)
                    
                    # Total Loss
                    loss = (loss_clean + loss_aug) / 2
                
                # Backward with AMP
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                batch_time = time.time() - end
                end = time.time()
                
                current_loss = loss.item()
                
                # Best Model & Early Stopping Logic
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                    self.save_model(best=True, loss=best_loss)
                    print(f"\rStep {step} | Loss: {current_loss:.4f} | Data: {data_time:.3f}s | Batch: {batch_time:.3f}s | 🌟 New Best!", flush=True)
                else:
                    patience_counter += 1
                    # Print every step for visibility
                    print(f"\rStep {step} | Loss: {current_loss:.4f} | Data: {data_time:.3f}s | Batch: {batch_time:.3f}s | Pat: {patience_counter}", end="", flush=True)
                
                if patience_counter >= patience:
                    log(f"\n🛑 Early stopping triggered after {patience} steps without improvement.")
                    break
                
                if step >= self.args.max_steps:
                    break
            
            if patience_counter >= patience or step >= self.args.max_steps:
                break
                
        log("\nTraining Complete.")

    def save_model(self, best=False, loss=None):
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        
        filename = "model_best.pt" if best else "model_final.pt"
        path = os.path.join(self.args.output_dir, filename)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': self.args,
            'loss': loss
        }, path)
        # Only log if it's the final save to avoid spamming
        if not best:
            log(f"Model saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ViT-L/14")
    parser.add_argument("--dataset", type=str, default="MSCOCO_i2t")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="robust_clip_mat_v2")
    
    args = parser.parse_args()
    
    trainer = MATTrainerV2(args)
    trainer.train()
