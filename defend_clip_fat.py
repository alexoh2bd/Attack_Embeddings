#!/usr/bin/env python3
"""defend_clip_fat.py

Fast Adversarial Training (FAT) for OpenAI CLIP ViT-L/14.
Uses FGSM (Fast Gradient Sign Method) to generate adversarial examples during training.
Includes Best Model Selection and Early Stopping.

Usage:
    python defend_clip_fat.py --effective_batch_size 16 --micro_batch_size 2 --max_steps 1000
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from PIL import Image
import numpy as np
import os
import sys
import clip
import time
import copy

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def fgsm_attack(model, image_tensor, text_tokens, epsilon=0.0314):
    """Generate FGSM adversarial example."""
    delta = torch.zeros_like(image_tensor, requires_grad=True)
    
    # Forward pass
    image_features = model.encode_image(image_tensor + delta)
    text_features = model.encode_text(text_tokens)
    
    # Normalize
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Loss
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    labels = torch.arange(len(image_tensor)).to(image_tensor.device)
    
    loss = (nn.CrossEntropyLoss()(logits_per_image, labels) + 
            nn.CrossEntropyLoss()(logits_per_text, labels)) / 2
            
    # Backward
    loss.backward()
    
    # FGSM step
    noise = epsilon * delta.grad.sign()
    return (image_tensor + noise).detach()

class FATTrainer:
    def __init__(self, model_name="ViT-L/14", output_dir="robust_clip_fat"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = output_dir
        
        log(f"Initializing FAT Trainer on {self.device}")
        log(f"Loading OpenAI CLIP model: {model_name}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        log("Model loaded.")
        
        self.model.float()
        self.model.train()
        
        # Freeze non-visual parameters
        for name, param in self.model.named_parameters():
            if not name.startswith('visual'):
                param.requires_grad = False
        
        self.optimizer = optim.AdamW(
            [p for p in self.model.visual.parameters() if p.requires_grad],
            lr=1e-6,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.1
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-8
        )
        
    def train(self, dataset_name="MSCOCO_i2t", effective_batch_size=16, micro_batch_size=2, max_steps=1000):
        log(f"Loading dataset {dataset_name}...")
        ds = load_dataset("TIGER-Lab/MMEB-eval", dataset_name, split="test")
        base_img_url = os.path.join(os.getcwd(), 'mmeb_images/')
        
        # Initialize Negative Text Bank
        log("Initializing Negative Text Bank (to prevent zero loss on small batches)...")
        all_texts = []
        for i in range(len(ds)):
            t = ds[i]['tgt_text']
            if isinstance(t, list): t = t[0]
            all_texts.append(t)
        
        # Sample 512 negatives
        num_negatives = 512
        if len(all_texts) > num_negatives:
            negative_texts = random.sample(all_texts, num_negatives)
        else:
            negative_texts = all_texts
            
        # Pre-compute negative features
        with torch.no_grad():
            neg_tokens = clip.tokenize(negative_texts, truncate=True).to(self.device)
            # Process in chunks to avoid OOM
            neg_features_list = []
            chunk_size = 128
            for i in range(0, len(neg_tokens), chunk_size):
                chunk = neg_tokens[i:i+chunk_size]
                neg_features_list.append(self.model.encode_text(chunk))
            
            negative_text_features = torch.cat(neg_features_list, dim=0)
            negative_text_features = negative_text_features / negative_text_features.norm(dim=-1, keepdim=True)
            negative_text_features = negative_text_features.detach()
            
        log(f"Negative Text Bank created with {len(negative_texts)} samples.")

        accumulation_steps = effective_batch_size // micro_batch_size
        log(f"FAT Configuration: Batch={effective_batch_size}, Micro={micro_batch_size}, Steps={max_steps}")
        
        step = 0
        epoch = 0
        global_step = 0
        
        # Best Model Tracking
        best_loss = float('inf')
        best_model_state = None
        patience = 100
        patience_counter = 0
        
        self.optimizer.zero_grad()
        
        while global_step < max_steps:
            epoch += 1
            log(f"Starting Epoch {epoch}")
            ds = ds.shuffle()
            
            for i in range(0, len(ds), micro_batch_size):
                if global_step >= max_steps:
                    break
                
                iter_start = time.time()
                batch = ds[i:i+micro_batch_size]
                
                images = []
                texts = []
                
                for j in range(len(batch['qry_img_path'])):
                    try:
                        img_path = batch['qry_img_path'][j]
                        text = batch['tgt_text'][j]
                        if isinstance(text, list): text = text[0]
                        
                        img_full_path = os.path.join(base_img_url, img_path)
                        if not os.path.exists(img_full_path): continue
                        
                        images.append(Image.open(img_full_path).convert("RGB"))
                        texts.append(text)
                    except: continue
                
                if not images: continue
                
                # Prepare inputs
                clean_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)
                text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
                
                # Generate Adversarial Examples (FAT)
                adv_tensors = fgsm_attack(self.model, clean_tensors, text_tokens)
                
                # Train on Mix (50% Clean, 50% Adv) - Compute losses separately to avoid duplicate text issue
                
                logit_scale = self.model.logit_scale.exp()
                
                # --- 1. Clean Loss ---
                image_features_clean = self.model.encode_image(clean_tensors)
                text_features_clean = self.model.encode_text(text_tokens)
                
                image_features_clean = image_features_clean / image_features_clean.norm(dim=-1, keepdim=True)
                text_features_clean = text_features_clean / text_features_clean.norm(dim=-1, keepdim=True)
                
                # Combine batch texts with negative texts for i2t loss
                all_text_features_clean = torch.cat([text_features_clean, negative_text_features], dim=0)
                
                logits_i2t_clean = logit_scale * image_features_clean @ all_text_features_clean.t()
                logits_t2i_clean = logit_scale * text_features_clean @ image_features_clean.t()
                
                labels = torch.arange(len(clean_tensors)).to(self.device)
                
                loss_clean = (nn.CrossEntropyLoss()(logits_i2t_clean, labels) + 
                              nn.CrossEntropyLoss()(logits_t2i_clean, labels)) / 2

                # --- 2. Adversarial Loss ---
                image_features_adv = self.model.encode_image(adv_tensors)
                # We can reuse text_features_clean since text is the same, but let's recompute to be safe/standard
                text_features_adv = self.model.encode_text(text_tokens) 
                
                image_features_adv = image_features_adv / image_features_adv.norm(dim=-1, keepdim=True)
                text_features_adv = text_features_adv / text_features_adv.norm(dim=-1, keepdim=True)
                
                # Combine batch texts with negative texts for i2t loss
                all_text_features_adv = torch.cat([text_features_adv, negative_text_features], dim=0)
                
                logits_i2t_adv = logit_scale * image_features_adv @ all_text_features_adv.t()
                logits_t2i_adv = logit_scale * text_features_adv @ image_features_adv.t()
                
                loss_adv = (nn.CrossEntropyLoss()(logits_i2t_adv, labels) + 
                            nn.CrossEntropyLoss()(logits_t2i_adv, labels)) / 2
                
                # Total Loss
                loss = (loss_clean + loss_adv) / 2
                
                loss = loss / accumulation_steps
                loss.backward()
                
                step += 1
                
                if step % 1 == 0:
                    print(f"\rMicro-batch {step}... (Acc {step % accumulation_steps}/{accumulation_steps})", end="", flush=True)
                
                if step % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.visual.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    actual_loss = loss.item() * accumulation_steps
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    # Best Model Logic
                    if actual_loss < best_loss:
                        best_loss = actual_loss
                        best_model_state = copy.deepcopy(self.model.state_dict())
                        patience_counter = 0
                        print(f"\n[{time.strftime('%H:%M:%S')}] Step {global_step}/{max_steps} | Loss: {actual_loss:.4f} | LR: {lr:.2e} | 🌟 New Best!")
                    else:
                        patience_counter += 1
                        print(f"\n[{time.strftime('%H:%M:%S')}] Step {global_step}/{max_steps} | Loss: {actual_loss:.4f} | LR: {lr:.2e} | Patience: {patience_counter}/{patience}")
                    
                    # Early Stopping
                    if patience_counter >= patience:
                        log(f"🛑 No improvement for {patience} steps. Stopping early.")
                        global_step = max_steps + 1 # Force exit
                        break
                
                del clean_tensors, adv_tensors, text_tokens
                del image_features_clean, text_features_clean, logits_i2t_clean, logits_t2i_clean, loss_clean
                del image_features_adv, text_features_adv, logits_i2t_adv, logits_t2i_adv, loss_adv
                del loss
                torch.cuda.empty_cache()
        
        log(f"Training finished. Saving BEST model (Loss: {best_loss:.4f}) to {self.output_dir}...")
        if not os.path.exists(self.output_dir): os.makedirs(self.output_dir)
        
        # Save Best Model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': best_loss
        }, os.path.join(self.output_dir, 'model.pt'))
        log("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--effective_batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=1000)
    args = parser.parse_args()
    
    FATTrainer().train(
        effective_batch_size=args.effective_batch_size,
        micro_batch_size=args.micro_batch_size,
        max_steps=args.max_steps
    )
