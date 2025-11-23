#!/usr/bin/env python3
"""defend_clip_mat_openai.py

Mixed Adversarial Training (MAT) for OpenAI CLIP ViT-L/14.
Uses Gradient Accumulation to fit on GPU.
Verbose logging enabled.

Usage:
    python defend_clip_mat_openai.py --effective_batch_size 16 --micro_batch_size 2 --max_steps 1000
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import os
import sys
import random
import clip
import time

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

class AugmentationDefense:
    """Aggressive data augmentation for robust training."""
    
    @staticmethod
    def gaussian_noise(img, sigma=0.1):
        img_array = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, sigma, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))
    
    @staticmethod
    def color_jitter(img):
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
        return img
    
    @staticmethod
    def random_blur(img):
        radius = random.choice([0, 0.5, 1.0, 1.5, 2.0])
        if radius > 0:
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img
    
    @staticmethod
    def random_crop_resize(img, scale_range=(0.7, 1.0)):
        w, h = img.size
        scale = random.uniform(*scale_range)
        new_w, new_h = int(w * scale), int(h * scale)
        
        left = random.randint(0, w - new_w) if new_w < w else 0
        top = random.randint(0, h - new_h) if new_h < h else 0
        
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.BICUBIC)
    
    @classmethod
    def augment(cls, img):
        if random.random() < 0.5:
            img = cls.color_jitter(img)
        if random.random() < 0.5:
            img = cls.random_blur(img)
        if random.random() < 0.5:
            img = cls.random_crop_resize(img)
        if random.random() < 0.3:
            img = cls.gaussian_noise(img, sigma=random.uniform(0.05, 0.15))
        return img

class MATTrainerOpenAI:
    def __init__(self, model_name="ViT-L/14", output_dir="robust_clip_openai"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.output_dir = output_dir
        
        log(f"Initializing Trainer on {self.device}")
        log(f"Loading OpenAI CLIP model: {model_name}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
        log("Model loaded.")
        
        # Convert to float32 for stability
        self.model.float()
        
        # Model setup
        self.model.train()
        
        # Freeze everything except visual encoder
        log("Freezing non-visual parameters...")
        for name, param in self.model.named_parameters():
            if not name.startswith('visual'):
                param.requires_grad = False
        
        # Optimizer
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
        log("Optimizer initialized.")
        
    def train(self, dataset_name="MSCOCO_i2t", effective_batch_size=16, micro_batch_size=2, max_steps=1000):
        log(f"Loading dataset {dataset_name}...")
        ds = load_dataset("TIGER-Lab/MMEB-eval", dataset_name, split="test")
        log(f"Dataset loaded. Size: {len(ds)}")
        
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
        log(f"Training Configuration:")
        log(f"  Effective Batch Size: {effective_batch_size}")
        log(f"  Micro Batch Size:     {micro_batch_size}")
        log(f"  Accumulation Steps:   {accumulation_steps}")
        log(f"  Max Steps:            {max_steps}")
        
        step = 0
        epoch = 0
        global_step = 0
        
        self.optimizer.zero_grad()
        
        while global_step < max_steps:
            epoch += 1
            log(f"Starting Epoch {epoch}")
            ds = ds.shuffle()
            
            # Manual batching
            for i in range(0, len(ds), micro_batch_size):
                if global_step >= max_steps:
                    break
                
                iter_start = time.time()
                
                batch = ds[i:i+micro_batch_size]
                
                clean_images = []
                aug_images = []
                texts = []
                
                # Load images and texts
                for j in range(len(batch['qry_img_path'])):
                    try:
                        img_path = batch['qry_img_path'][j]
                        text = batch['tgt_text'][j]
                        
                        if isinstance(text, list):
                            text = text[0]
                        
                        img_full_path = os.path.join(base_img_url, img_path)
                        
                        if not os.path.exists(img_full_path):
                            continue
                        
                        image = Image.open(img_full_path).convert("RGB")
                        clean_images.append(image)
                        
                        # Augmented version
                        aug_img = AugmentationDefense.augment(image.copy())
                        aug_images.append(aug_img)
                        
                        texts.append(text)
                    except Exception as e:
                        continue
                
                if len(clean_images) == 0:
                    continue
                
                # Preprocess
                clean_tensors = torch.stack([self.preprocess(img) for img in clean_images]).to(self.device)
                aug_tensors = torch.stack([self.preprocess(img) for img in aug_images]).to(self.device)
                text_tokens = clip.tokenize(texts, truncate=True).to(self.device)
                
                # Forward pass - Compute losses separately to avoid duplicate text issue
                
                logit_scale = self.model.logit_scale.exp()
                
                # --- 1. Clean Loss ---
                image_features_clean = self.model.encode_image(clean_tensors)
                text_features_clean = self.model.encode_text(text_tokens)
                
                image_features_clean = image_features_clean / image_features_clean.norm(dim=-1, keepdim=True)
                text_features_clean = text_features_clean / text_features_clean.norm(dim=-1, keepdim=True)
                
                # Combine batch texts with negative texts for i2t loss
                all_text_features_clean = torch.cat([text_features_clean, negative_text_features], dim=0)
                
                # i2t: Image vs (Batch Texts + Negatives)
                logits_i2t_clean = logit_scale * image_features_clean @ all_text_features_clean.t()
                
                # t2i: Text vs Batch Images (Standard)
                logits_t2i_clean = logit_scale * text_features_clean @ image_features_clean.t()
                
                labels = torch.arange(len(clean_tensors)).to(self.device)
                
                loss_clean = (nn.CrossEntropyLoss()(logits_i2t_clean, labels) + 
                              nn.CrossEntropyLoss()(logits_t2i_clean, labels)) / 2

                # --- 2. Augmented Loss ---
                image_features_aug = self.model.encode_image(aug_tensors)
                # Reuse text features or recompute (recomputing for safety)
                text_features_aug = self.model.encode_text(text_tokens)
                
                image_features_aug = image_features_aug / image_features_aug.norm(dim=-1, keepdim=True)
                text_features_aug = text_features_aug / text_features_aug.norm(dim=-1, keepdim=True)
                
                # Combine batch texts with negative texts for i2t loss
                all_text_features_aug = torch.cat([text_features_aug, negative_text_features], dim=0)
                
                logits_i2t_aug = logit_scale * image_features_aug @ all_text_features_aug.t()
                logits_t2i_aug = logit_scale * text_features_aug @ image_features_aug.t()
                
                loss_aug = (nn.CrossEntropyLoss()(logits_i2t_aug, labels) + 
                            nn.CrossEntropyLoss()(logits_t2i_aug, labels)) / 2
                
                # Total Loss
                loss = (loss_clean + loss_aug) / 2
                
                # Scale loss for accumulation
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                step += 1
                
                # Log micro-step
                if step % 1 == 0:
                    print(f"\rProcessing micro-batch {step}... (Accumulating {step % accumulation_steps}/{accumulation_steps})", end="", flush=True)
                
                # Optimizer step after accumulation
                if step % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.visual.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # Log global step
                    actual_loss = loss.item() * accumulation_steps
                    current_lr = self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - iter_start
                    print(f"\n[{time.strftime('%H:%M:%S')}] Step {global_step}/{max_steps} | Loss: {actual_loss:.4f} | LR: {current_lr:.2e}")
                
                # Cleanup
                del clean_tensors, aug_tensors, text_tokens
                del image_features_clean, text_features_clean, logits_i2t_clean, logits_t2i_clean, loss_clean
                del image_features_aug, text_features_aug, logits_i2t_aug, logits_t2i_aug, loss_aug
                del loss
                torch.cuda.empty_cache()
        
        log(f"Training finished. Total steps: {global_step}")
        self.save_model()
    
    def save_model(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        log(f"Saving model to {self.output_dir}...")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.output_dir, 'model.pt'))
        log("Model saved.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MSCOCO_i2t")
    parser.add_argument("--effective_batch_size", type=int, default=16)
    parser.add_argument("--micro_batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="robust_clip_openai")
    args = parser.parse_args()

    log("STARTING SCRIPT")
    
    trainer = MATTrainerOpenAI(
        model_name="ViT-L/14",
        output_dir=args.output_dir
    )
    
    trainer.train(
        dataset_name=args.dataset,
        effective_batch_size=args.effective_batch_size,
        micro_batch_size=args.micro_batch_size,
        max_steps=args.max_steps
    )
    
    log("SCRIPT COMPLETE")

if __name__ == "__main__":
    main()
