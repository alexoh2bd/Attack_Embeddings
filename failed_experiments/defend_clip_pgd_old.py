#!/usr/bin/env python3
"""defend_clip_pgd.py

Full PGD Adversarial Training for OpenAI CLIP ViT-L/14.
Based on the robust V2 architecture (AMP, DataLoader, Negative Bank).

Key Features:
- Iterative PGD Attack (7 steps) during training.
- Automatic Mixed Precision (AMP).
- Negative Text Bank for stable loss.
- Early Stopping & Best Model Saving.

Usage:
    python defend_clip_pgd.py --batch_size 16 --epochs 1
"""

import os
import sys
import time
import random
import argparse
import copy
import numpy as np
from PIL import Image
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

# --- PGD Attack Logic ---
def pgd_attack(model, images, texts, epsilon=0.0314, alpha=0.0078, steps=7):
    """
    Generates PGD adversarial examples.
    epsilon: Max perturbation (8/255 approx 0.0314)
    alpha: Step size (2/255 approx 0.0078)
    steps: Number of iterations
    """
    # Clone images to avoid modifying original
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1).detach()
    
    for _ in range(steps):
        adv_images.requires_grad = True
        
        # Forward pass
        image_features = model.encode_image(adv_images)
        text_features = model.encode_text(texts)
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Loss
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        labels = torch.arange(len(images)).to(images.device)
        
        loss = (nn.CrossEntropyLoss()(logits_per_image, labels) + 
                nn.CrossEntropyLoss()(logits_per_text, labels)) / 2
        
        # Gradient
        grad = torch.autograd.grad(loss, adv_images)[0]
        
        # PGD Step
        adv_images = adv_images.detach() + alpha * grad.sign()
        
        # Projection
        delta = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + delta, 0, 1).detach()
        
    return adv_images

# --- Dataset ---
class CLIPDataset(Dataset):
    def __init__(self, dataset_name, preprocess, base_img_path, max_samples=None):
        self.preprocess = preprocess
        self.base_img_path = base_img_path
        
        log(f"Loading HF Dataset: {dataset_name}...")
        self.ds = load_dataset("TIGER-Lab/MMEB-train", dataset_name, split="original")
        
        
        if max_samples:
            self.ds = self.ds.select(range(min(len(self.ds), max_samples)))
            
        self.valid_indices = self._filter_valid_images()
        log(f"Found {len(self.valid_indices)} valid images out of {len(self.ds)}")

    def _filter_valid_images(self):
        valid_indices = []
        for i in range(len(self.ds)):
            # MMEB-train uses 'qry_image_path', MMEB-eval uses 'qry_img_path'
            img_path = self.ds[i].get('qry_image_path', self.ds[i].get('qry_img_path'))
            full_path = os.path.join(self.base_img_path, img_path)
            if os.path.exists(full_path):
                valid_indices.append(i)
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        item = self.ds[real_idx]
        
        # Handle different column names
        img_path = item.get('qry_image_path', item.get('qry_img_path'))
        text = item.get('pos_text', item.get('tgt_text'))
        
        if isinstance(text, list):
            text = text[0]
        
        full_path = os.path.join(self.base_img_path, img_path)
        
        try:
            image = Image.open(full_path).convert("RGB")
            # For PGD, we need the raw tensor before normalization for the attack loop
            # But CLIP's preprocess usually includes normalization.
            # We will use the standard preprocess and assume it handles it, 
            # OR we might need to manually handle normalization inside the attack if preprocess does it.
            # Standard CLIP preprocess: Resize -> CenterCrop -> ToTensor -> Normalize
            # We need: Resize -> CenterCrop -> ToTensor -> [Attack] -> Normalize
            
            # For simplicity in this script, we'll rely on the fact that PGD usually works on [0,1] tensors.
            # If CLIP preprocess normalizes, we need to un-normalize or implement custom transform.
            # Let's check standard CLIP preprocess. It DOES normalize.
            # So we will use a custom transform that returns [0,1] tensor.
            
            return {
                "image_path": full_path,
                "text": text
            }
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

# --- Trainer ---
class PGDTrainer:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        log(f"Initializing Model: {args.model_name}")
        self.model, self.clip_preprocess = clip.load(args.model_name, device=self.device, jit=False)
        self.model.float()
        
        # Freeze non-visual
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
        
        # Extract normalization stats from CLIP preprocess
        # transforms: Resize, CenterCrop, ToTensor, Normalize
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(self.device).view(1, 3, 1, 1)

    def preprocess_for_pgd(self, img_path):
        """Custom preprocess: Resize/Crop -> Tensor [0,1]. No normalization yet."""
        img = Image.open(img_path).convert("RGB")
        # Resize and Crop manually to match CLIP
        img = img.resize((224, 224), Image.BICUBIC) # Simplified resize
        # Ideally use CLIP's transforms but without Normalize
        
        # Convert to tensor [0,1]
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        return img_tensor

    def normalize(self, tensor):
        """Apply CLIP normalization."""
        return (tensor - self.mean) / self.std

    def get_negative_bank(self, dataset):
        log("Creating Negative Text Bank...")
        all_texts = [dataset[i]['text'] for i in range(len(dataset))]
        num_neg = min(len(all_texts), 512)
        neg_texts = random.sample(all_texts, num_neg)
        
        with torch.no_grad():
            neg_tokens = clip.tokenize(neg_texts, truncate=True).to(self.device)
            neg_features = self.model.encode_text(neg_tokens)
            neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
        return neg_features.detach()

    def train(self):
        dataset = CLIPDataset(
            self.args.dataset, 
            self.clip_preprocess, 
            os.path.join(os.getcwd(), 'mmeb_images/')
        )
        
        # Custom collate to handle raw image paths
        def collate_fn(batch):
            return batch
            
        dataloader = DataLoader(
            dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True, 
            num_workers=2,
            collate_fn=collate_fn,
            drop_last=True
        )
        
        neg_features_bank = self.get_negative_bank(dataset)
        
        log(f"Starting PGD Training for {self.args.epochs} epochs...")
        
        step = 0
        best_loss = float('inf')
        patience = 50 # Lower patience for PGD as it's slower
        patience_counter = 0
        
        for epoch in range(self.args.epochs):
            self.model.train()
            end = time.time()
            
            for batch in dataloader:
                data_time = time.time() - end
                step += 1
                
                # Prepare Batch
                img_paths = [item['image_path'] for item in batch]
                raw_texts = [item['text'] for item in batch]
                
                # Load images to [0,1] tensors
                clean_tensors = torch.stack([self.preprocess_for_pgd(p) for p in img_paths]).to(self.device)
                text_tokens = clip.tokenize(raw_texts, truncate=True).to(self.device)
                
                # --- PGD Attack Generation ---
                # We need to attack the model to find worst-case examples
                # This requires gradients, so we enable grad for inputs
                # Note: We attack the *normalized* input usually, but here we have [0,1]
                # We will define a wrapper to normalize on the fly
                
                self.model.eval() # Eval mode for attack generation
                
                # PGD Loop
                adv_tensors = clean_tensors.clone().detach()
                epsilon = 8/255
                alpha = 2/255
                pgd_steps = 7
                
                adv_tensors = adv_tensors + torch.empty_like(adv_tensors).uniform_(-epsilon, epsilon)
                adv_tensors = torch.clamp(adv_tensors, 0, 1).detach()
                
                for _ in range(pgd_steps):
                    adv_tensors.requires_grad = True
                    
                    # Normalize before passing to CLIP
                    norm_adv = self.normalize(adv_tensors)
                    
                    img_emb = self.model.encode_image(norm_adv)
                    txt_emb = self.model.encode_text(text_tokens)
                    
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                    txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                    
                    logits = self.model.logit_scale.exp() * img_emb @ txt_emb.t()
                    labels = torch.arange(len(adv_tensors)).to(self.device)
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    
                    grad = torch.autograd.grad(loss, adv_tensors)[0]
                    adv_tensors = adv_tensors.detach() + alpha * grad.sign()
                    delta = torch.clamp(adv_tensors - clean_tensors, -epsilon, epsilon)
                    adv_tensors = torch.clamp(clean_tensors + delta, 0, 1).detach()
                
                self.model.train() # Back to train mode
                
                # --- Training Step ---
                self.optimizer.zero_grad()
                
                with autocast():
                    # Encode Texts
                    text_features = self.model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    all_text_features = torch.cat([text_features, neg_features_bank], dim=0)
                    
                    # 1. Clean Loss
                    norm_clean = self.normalize(clean_tensors)
                    img_feat_clean = self.model.encode_image(norm_clean)
                    img_feat_clean = img_feat_clean / img_feat_clean.norm(dim=-1, keepdim=True)
                    
                    logit_scale = self.model.logit_scale.exp()
                    logits_clean = logit_scale * img_feat_clean @ all_text_features.t()
                    labels = torch.arange(len(clean_tensors)).to(self.device)
                    loss_clean = self.loss_fn(logits_clean, labels)
                    
                    # 2. Adversarial Loss
                    norm_adv = self.normalize(adv_tensors)
                    img_feat_adv = self.model.encode_image(norm_adv)
                    img_feat_adv = img_feat_adv / img_feat_adv.norm(dim=-1, keepdim=True)
                    
                    logits_adv = logit_scale * img_feat_adv @ all_text_features.t()
                    loss_adv = self.loss_fn(logits_adv, labels)
                    
                    loss = (loss_clean + loss_adv) / 2
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                batch_time = time.time() - end
                end = time.time()
                
                current_loss = loss.item()
                
                if current_loss < best_loss:
                    best_loss = current_loss
                    patience_counter = 0
                    self.save_model(best=True, loss=best_loss)
                    print(f"\rStep {step} | Loss: {current_loss:.4f} | Data: {data_time:.3f}s | Batch: {batch_time:.3f}s | 🌟 New Best!", flush=True)
                else:
                    patience_counter += 1
                    print(f"\rStep {step} | Loss: {current_loss:.4f} | Data: {data_time:.3f}s | Batch: {batch_time:.3f}s | Pat: {patience_counter}", end="", flush=True)
                
                if patience_counter >= patience or step >= self.args.max_steps:
                    break
            
            if patience_counter >= patience or step >= self.args.max_steps:
                break
                
        log("\nPGD Training Complete.")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="ViT-L/14")
    parser.add_argument("--dataset", type=str, default="MSCOCO_i2t")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="robust_clip_pgd")
    
    args = parser.parse_args()
    
    # Check for deprecation warning fix
    import warnings
    warnings.filterwarnings("ignore")
    
    trainer = PGDTrainer(args)
    trainer.train()
