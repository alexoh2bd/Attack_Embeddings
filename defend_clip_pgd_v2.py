#!/usr/bin/env python3
"""defend_clip_pgd_v2.py
Generation augmented by the help of Google Antigravity - model Gemini 3 Pro and Claude Sonnet 4.5 Thinking
CHAT HISTORY ATTACHED as Antigravity-Agentic-Coding-Chat-History.md

Command to train:
    python defend_clip_pgd_v2.py --batch_size 16 --epochs 3 --loss_threshold 0.1
"""

import os
import sys
import time
import random
import argparse
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import clip

# --- Configuration ---
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# --- Dataset ---
class PGDDataset(Dataset):
    def __init__(self, metadata_path, base_img_path, attacked_dir, preprocess):
        self.base_img_path = base_img_path
        self.attacked_dir = attacked_dir
        self.preprocess = preprocess
        
        log(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        log(f"Found {len(self.metadata)} adversarial examples")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        
        # Load clean image
        clean_path = os.path.join(self.base_img_path, item['original_path'])
        # Load adversarial image
        adv_path = os.path.join(self.attacked_dir, item['adv_path'])
        
        try:
            clean_img = Image.open(clean_path).convert("RGB")
            adv_img = Image.open(adv_path).convert("RGB")
            
            clean_tensor = self.preprocess(clean_img)
            adv_tensor = self.preprocess(adv_img)
            
            return {
                "clean_image": clean_tensor,
                "adv_image": adv_tensor,
                "text": item['text']
            }
        except Exception as e:
            print(f"Error loading {clean_path} or {adv_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

# --- Trainer ---
class PGDTrainerV2:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        log(f"Initializing Model: {args.model_name}")
        self.model, self.preprocess = clip.load(args.model_name, device=self.device, jit=False)
        self.model.float()
        
        # Freeze non-visual parameters
        for name, param in self.model.named_parameters():
            if "visual" not in name:
                param.requires_grad = False
                
        self.optimizer = optim.AdamW(
            [p for p in self.model.visual.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        self.scaler = GradScaler()
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
        dataset = PGDDataset(
            metadata_path=os.path.join(self.args.attacked_dir, "metadata.json"),
            base_img_path=self.args.base_img_path,
            attacked_dir=self.args.attacked_dir,
            preprocess=self.preprocess
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
        
        log(f"Starting PGD Training for {self.args.epochs} epochs...")
        log(f"Loss threshold for early stop: {self.args.loss_threshold}")
        
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
                adv_imgs = batch['adv_image'].to(self.device)
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
                    
                    # 3. Adversarial Loss
                    img_features_adv = self.model.encode_image(adv_imgs)
                    img_features_adv = img_features_adv / img_features_adv.norm(dim=-1, keepdim=True)
                    
                    logits_adv = logit_scale * img_features_adv @ all_text_features.t()
                    loss_adv = self.loss_fn(logits_adv, labels)
                    
                    # Total Loss
                    loss = (loss_clean + loss_adv) / 2
                
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
                    print(f"\rStep {step} | Loss: {current_loss:.4f} | Data: {data_time:.3f}s | Batch: {batch_time:.3f}s | Pat: {patience_counter}", end="", flush=True)
                
                # Check loss threshold
                if current_loss < self.args.loss_threshold:
                    log(f"\n🎯 Loss threshold {self.args.loss_threshold} reached! Stopping training.")
                    self.save_model()
                    return
                
                if patience_counter >= patience:
                    log(f"\n🛑 Early stopping triggered after {patience} steps without improvement.")
                    break
                
                if step >= self.args.max_steps:
                    break
            
            if patience_counter >= patience or step >= self.args.max_steps:
                break
                
        log("\nTraining Complete.")
        self.save_model()

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
        if not best:
            log(f"Model saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ViT-L/14")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--loss_threshold", type=float, default=0.1, help="Stop training when loss drops below this")
    parser.add_argument("--attacked_dir", type=str, default="mmeb_train_attacked")
    parser.add_argument("--base_img_path", type=str, default="mmeb_images")
    parser.add_argument("--output_dir", type=str, default="robust_clip_pgd")
    
    args = parser.parse_args()
    
    trainer = PGDTrainerV2(args)
    trainer.train()
