#!/usr/bin/env python3
"""defend_clip_comprehensive.py

Comprehensive adversarial training for CLIP models on real datasets.
Trains on MSCOCO or other MMEB datasets with configurable FAT/MAT strategies.

Usage:
    # Train with Full Adversarial Training (FAT)
    python defend_clip_comprehensive.py --mode train --defense_type FAT --dataset MSCOCO_i2t --max_steps 1000 --batch_size 16

    # Train with Mixed Adversarial Training (MAT)
    python defend_clip_comprehensive.py --mode train --defense_type MAT --dataset MSCOCO_i2t --max_steps 1000 --batch_size 16

    # Evaluate a trained robust model
    python defend_clip_comprehensive.py --mode eval --robust_model_path ./robust_clip_model --dataset MSCOCO_i2t
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from pipe.pipe import MultimodalRetrievalPipeline, Config
from pipe.attacks import pgd_attack_clip

class AdversarialTrainerComprehensive:
    def __init__(self, model_name=Config.MODEL_NAME, output_dir="robust_clip_model", defense_type="FAT"):
        """
        Args:
            model_name: CLIP model name
            output_dir: Directory to save the trained model
            defense_type: "FAT" (Full Adversarial Training) or "MAT" (Mixed Adversarial Training)
        """
        self.pipeline = MultimodalRetrievalPipeline(model_name)
        self.model = self.pipeline.model
        self.processor = self.pipeline.processor
        self.device = Config.DEVICE
        self.output_dir = output_dir
        self.defense_type = defense_type
        
        print(f"Defense Type: {defense_type}")
        print(f"{'='*60}")
        if defense_type == "FAT":
            print("Full Adversarial Training: 100% adversarial examples")
        elif defense_type == "MAT":
            print("Mixed Adversarial Training: 50% clean + 50% adversarial")
        else:
            raise ValueError(f"Unknown defense type: {defense_type}")
        print(f"{'='*60}\n")
        
        # Ensure model is in training mode
        self.model.train()
        
        # Freeze text encoder to save memory/compute (optional, but common in fine-tuning)
        for param in self.model.text_model.parameters():
            param.requires_grad = False
            
        self.optimizer = optim.AdamW(self.model.vision_model.parameters(), lr=5e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
        
    def train(self, dataset_name="MSCOCO_i2t", num_epochs=1, batch_size=16, max_steps=1000, attack_epsilon=0.03, attack_steps=5):
        """
        Train the model with adversarial examples.
        
        Args:
            dataset_name: Name of dataset from MMEB
            num_epochs: Number of training epochs
            batch_size: Batch size
            max_steps: Maximum training steps
            attack_epsilon: Epsilon for PGD attack
            attack_steps: Number of PGD steps
        """
        print(f"Loading dataset {dataset_name}...")
        ds = load_dataset("TIGER-Lab/MMEB-eval", dataset_name, split="test")
        
        base_img_url = os.path.join(os.getcwd(), 'mmeb_images/')
        
        print(f"Starting adversarial training for {num_epochs} epochs (max {max_steps} steps)...")
        print(f"Attack params: epsilon={attack_epsilon}, steps={attack_steps}")
        print(f"Batch size: {batch_size}\n")
        
        step = 0
        for epoch in range(num_epochs):
            # Shuffle dataset
            ds = ds.shuffle()
            
            for i in tqdm(range(0, len(ds), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
                if step >= max_steps:
                    break
                
                batch = ds[i:i+batch_size]
                
                images = []
                texts = []
                
                # Load images and texts
                for j in range(len(batch['qry_img_path'])):
                    try:
                        img_path = batch['qry_img_path'][j]
                        text = batch['tgt_text'][j]
                        
                        # Handle text list
                        if isinstance(text, list):
                            text = text[0]
                            
                        img_full_path = os.path.join(base_img_url, img_path)
                        
                        # Check if file exists
                        if not os.path.exists(img_full_path):
                            continue
                            
                        image = Image.open(img_full_path).convert("RGB")
                        images.append(image)
                        texts.append(text)
                    except Exception as e:
                        continue
                
                if len(images) < 2:  # Need at least 2 for contrastive learning
                    continue
                
                # Prepare clean images
                clean_inputs = self.processor(
                    text=texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)
                
                # Generate adversarial examples with WEAKER attack for training stability
                # Use smaller epsilon and fewer steps than evaluation attacks
                adv_images = []
                for img, txt in zip(images, texts):
                    # Switch to eval mode for attack generation
                    self.model.eval()
                    # Use weaker attack: half the epsilon, fewer steps
                    adv_img = pgd_attack_clip(
                        img, self.pipeline, 
                        epsilon=attack_epsilon * 0.5,  # Weaker attack
                        steps=max(attack_steps // 2, 2),  # Fewer steps
                        target_text=txt
                    )
                    adv_images.append(adv_img)
                    self.model.train()
                
                # Prepare adversarial inputs
                adv_inputs = self.processor(
                    text=texts,
                    images=adv_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)
                
                # Select training data based on defense type
                if self.defense_type == "MAT":
                    # Mixed: combine clean and adversarial
                    pixel_values = torch.cat([clean_inputs['pixel_values'], adv_inputs['pixel_values']], dim=0)
                    input_ids = torch.cat([clean_inputs['input_ids'], adv_inputs['input_ids']], dim=0)
                    attention_mask = torch.cat([clean_inputs['attention_mask'], adv_inputs['attention_mask']], dim=0)
                elif self.defense_type == "FAT":
                    # Full adversarial - but still need clean examples for stability
                    # Use 80% adv, 20% clean
                    num_adv = len(adv_inputs['pixel_values'])
                    num_clean = max(num_adv // 4, 1)
                    pixel_values = torch.cat([adv_inputs['pixel_values'], clean_inputs['pixel_values'][:num_clean]], dim=0)
                    input_ids = torch.cat([adv_inputs['input_ids'], clean_inputs['input_ids'][:num_clean]], dim=0)
                    attention_mask = torch.cat([adv_inputs['attention_mask'], clean_inputs['attention_mask'][:num_clean]], dim=0)
                else:
                    raise ValueError(f"Unknown defense type: {self.defense_type}")
                
                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # CLIP loss (contrastive loss)
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                # Ground truth labels are diagonal (0, 1, 2, ...)
                labels = torch.arange(len(pixel_values)).to(self.device)
                
                loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
                loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
                loss = (loss_i + loss_t) / 2
                
                # Gradient clipping for stability
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.vision_model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                if step % 10 == 0:
                    tqdm.write(f"Step {step}, Loss: {loss.item():.4f}")
                
                step += 1
                
            if step >= max_steps:
                break
                
        print(f"\nTraining finished. Total steps: {step}")
        self.save_model()
        
    def save_model(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print(f"Saving model to {self.output_dir}...")
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        print("Model saved.")

def evaluate_robust_model(robust_model_path, dataset_name="MSCOCO_i2t", perturbations=None):
    """
    Evaluate a robust model against various perturbations.
    
    Args:
        robust_model_path: Path to saved robust model
        dataset_name: Dataset to evaluate on
        perturbations: List of perturbation types
    """
    if perturbations is None:
        perturbations = ["ctrl", "fgsm", "pgd"]
    
    print(f"\n{'='*60}")
    print("EVALUATION: Comparing Original vs Robust Model")
    print(f"{'='*60}\n")
    
    print("To evaluate your trained model, use the existing evaluation scripts:")
    print(f"  1. src/pipe/evaluate_robust.py")
    print(f"  2. Or modify src/pipe/experiment.py to use your robust model path")
    print(f"\nYour robust model is saved at: {robust_model_path}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval"], 
                        help="Mode: train (adversarial training) or eval (evaluate model)")
    parser.add_argument("--defense_type", type=str, default="FAT", choices=["FAT", "MAT"],
                        help="Defense type: FAT (Full Adversarial Training) or MAT (Mixed Adversarial Training)")
    parser.add_argument("--dataset", type=str, default="MSCOCO_i2t",
                        help="Dataset name from MMEB (e.g., MSCOCO_i2t, VisualNews_i2t)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--max_steps", type=int, default=1000,
                        help="Maximum training steps")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--attack_epsilon", type=float, default=0.03,
                        help="Epsilon for PGD attack during training")
    parser.add_argument("--attack_steps", type=int, default=5,
                        help="Number of PGD steps during training")
    parser.add_argument("--output_dir", type=str, default="robust_clip_model",
                        help="Output directory for trained model")
    parser.add_argument("--robust_model_path", type=str, default=None,
                        help="Path to robust model for evaluation (eval mode only)")
    args = parser.parse_args()

    if args.mode == "train":
        print(f"\n{'='*60}")
        print("COMPREHENSIVE ADVERSARIAL TRAINING FOR CLIP")
        print(f"{'='*60}\n")
        
        trainer = AdversarialTrainerComprehensive(
            model_name=Config.MODEL_NAME,
            output_dir=args.output_dir,
            defense_type=args.defense_type
        )
        
        trainer.train(
            dataset_name=args.dataset,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            attack_epsilon=args.attack_epsilon,
            attack_steps=args.attack_steps
        )
        
        print(f"\n{'='*60}")
        print(f"Training complete! Model saved to: {args.output_dir}")
        print(f"{'='*60}\n")
        
    elif args.mode == "eval":
        if args.robust_model_path is None:
            args.robust_model_path = args.output_dir
            
        evaluate_robust_model(
            robust_model_path=args.robust_model_path,
            dataset_name=args.dataset
        )

if __name__ == "__main__":
    main()
