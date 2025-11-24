#!/usr/bin/env python3
"""defend_clip_mat.py

Mixed Adversarial Training (MAT) for CLIP using aggressive data augmentation.
More stable than PGD-based training while still improving robustness.

Usage:
    python defend_clip_mat.py --max_steps 1000 --batch_size 16
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from tqdm import tqdm
import os
import sys
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pipe.pipe import MultimodalRetrievalPipeline, Config

class AugmentationDefense:
    """Aggressive data augmentation for robust training."""
    
    @staticmethod
    def gaussian_noise(img, sigma=0.1):
        """Add Gaussian noise."""
        img_array = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, sigma, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))
    
    @staticmethod
    def color_jitter(img):
        """Apply color jitter."""
        # Random brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
        # Random contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
        # Random saturation
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))
        return img
    
    @staticmethod
    def random_blur(img):
        """Apply random Gaussian blur."""
        radius = random.choice([0, 0.5, 1.0, 1.5, 2.0])
        if radius > 0:
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img
    
    @staticmethod
    def random_crop_resize(img, scale_range=(0.7, 1.0)):
        """Random crop and resize back."""
        w, h = img.size
        scale = random.uniform(*scale_range)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Random crop position
        left = random.randint(0, w - new_w) if new_w < w else 0
        top = random.randint(0, h - new_h) if new_h < h else 0
        
        # Crop and resize back
        cropped = img.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((w, h), Image.BICUBIC)
    
    @classmethod
    def augment(cls, img):
        """Apply random combination of augmentations."""
        # Apply each augmentation with 50% probability
        if random.random() < 0.5:
            img = cls.color_jitter(img)
        if random.random() < 0.5:
            img = cls.random_blur(img)
        if random.random() < 0.5:
            img = cls.random_crop_resize(img)
        if random.random() < 0.3:
            img = cls.gaussian_noise(img, sigma=random.uniform(0.05, 0.15))
        return img

class MATTrainer:
    def __init__(self, model_name=Config.MODEL_NAME, output_dir="robust_clip_mat"):
        """
        Mixed Adversarial Training with data augmentation.
        """
        self.pipeline = MultimodalRetrievalPipeline(model_name)
        self.model = self.pipeline.model
        self.processor = self.pipeline.processor
        self.device = Config.DEVICE
        self.output_dir = output_dir
        
        print(f"{'='*60}")
        print("Mixed Adversarial Training (MAT) with Data Augmentation")
        print("50% clean + 50% augmented (crop/jitter/blur/noise)")
        print(f"{'='*60}\n")
        
        # Model setup
        self.model.train()
        
        # Freeze text encoder
        for param in self.model.text_model.parameters():
            param.requires_grad = False
            
        # Optimizer with conservative learning rate
        self.optimizer = optim.AdamW(
            self.model.vision_model.parameters(), 
            lr=3e-6,  # Very conservative learning rate
            betas=(0.9, 0.98), 
            eps=1e-6, 
            weight_decay=0.1
        )
        
        # Learning rate scheduler for gradual warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000, 
            eta_min=1e-7
        )
        
    def train(self, dataset_name="MSCOCO_i2t", batch_size=16, max_steps=1000):
        """
        Train with MAT strategy.
        """
        print(f"Loading dataset {dataset_name}...")
        ds = load_dataset("TIGER-Lab/MMEB-eval", dataset_name, split="test")
        
        base_img_url = os.path.join(os.getcwd(), 'mmeb_images/')
        
        print(f"Starting MAT training for {max_steps} steps...")
        print(f"Batch size: {batch_size}\n")
        
        step = 0
        epoch = 0
        
        while step < max_steps:
            epoch += 1
            ds = ds.shuffle()
            
            for i in tqdm(range(0, len(ds), batch_size), desc=f"Epoch {epoch}"):
                if step >= max_steps:
                    break
                
                batch = ds[i:i+batch_size]
                
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
                        
                        # Create augmented version
                        aug_img = AugmentationDefense.augment(image.copy())
                        aug_images.append(aug_img)
                        
                        texts.append(text)
                    except Exception as e:
                        continue
                
                if len(clean_images) < 2:
                    continue
                
                # Process clean images
                clean_inputs = self.processor(
                    text=texts,
                    images=clean_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)
                
                # Process augmented images
                aug_inputs = self.processor(
                    text=texts,
                    images=aug_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)
                
                # MAT: Mix 50% clean + 50% augmented
                pixel_values = torch.cat([
                    clean_inputs['pixel_values'], 
                    aug_inputs['pixel_values']
                ], dim=0)
                input_ids = torch.cat([
                    clean_inputs['input_ids'], 
                    aug_inputs['input_ids']
                ], dim=0)
                attention_mask = torch.cat([
                    clean_inputs['attention_mask'], 
                    aug_inputs['attention_mask']
                ], dim=0)
                
                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # CLIP contrastive loss
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                labels = torch.arange(len(pixel_values)).to(self.device)
                
                loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
                loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
                loss = (loss_i + loss_t) / 2
                
                # Backward pass with gradient clipping
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.vision_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                if step % 10 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    tqdm.write(f"Step {step}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
                
                step += 1
        
        print(f"\nTraining finished. Total steps: {step}")
        self.save_model()
        
    def save_model(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print(f"Saving model to {self.output_dir}...")
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        print("Model saved.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MSCOCO_i2t")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="robust_clip_mat")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("MIXED ADVERSARIAL TRAINING (MAT) FOR CLIP")
    print(f"{'='*60}\n")
    
    trainer = MATTrainer(
        model_name=Config.MODEL_NAME,
        output_dir=args.output_dir
    )
    
    trainer.train(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        max_steps=args.max_steps
    )
    
    print(f"\n{'='*60}")
    print(f"Training complete! Model saved to: {args.output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
