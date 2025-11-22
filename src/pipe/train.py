import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import os
from io import BytesIO
import requests

from pipe import MultimodalRetrievalPipeline, Config
from attacks import pgd_attack_clip

class AdversarialTrainer:
    def __init__(self, model_name=Config.MODEL_NAME, output_dir="robust_clip_model"):
        self.pipeline = MultimodalRetrievalPipeline(model_name)
        self.model = self.pipeline.model
        self.processor = self.pipeline.processor
        self.device = Config.DEVICE
        self.output_dir = output_dir
        
        # Ensure model is in training mode
        self.model.train()
        
        # Freeze text encoder to save memory/compute (optional, but common in fine-tuning)
        for param in self.model.text_model.parameters():
            param.requires_grad = False
            
        self.optimizer = optim.AdamW(self.model.vision_model.parameters(), lr=1e-5)
        
    def train(self, dataset_name="MSCOCO_i2t", num_epochs=1, batch_size=16, max_steps=100):
        print(f"Loading dataset {dataset_name}...")
        ds = load_dataset("TIGER-Lab/MMEB-eval", dataset_name, split="test") # Using test split as it's usually smaller/available in this specific eval dataset structure
        
        # Create a simple dataloader
        # Note: The MMEB dataset structure might vary, adapting for MSCOCO_i2t
        # It usually has 'qry_img_path' and 'tgt_text'
        
        base_img_url = '/work/aho13/work/aho13/'
        
        print(f"Starting adversarial training for {num_epochs} epochs...")
        
        step = 0
        for epoch in range(num_epochs):
            # Shuffle dataset
            ds = ds.shuffle()
            
            for i in tqdm(range(0, len(ds), batch_size)):
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
                        print(f"Error loading data: {e}")
                        continue
                
                if not images:
                    continue
                    
                # Generate adversarial examples
                adv_images = []
                for img, txt in zip(images, texts):
                    # Generate PGD attack
                    # We use the pipeline in eval mode for attack generation to get accurate gradients relative to current state
                    self.model.eval()
                    adv_img = pgd_attack_clip(img, self.pipeline, epsilon=0.03, steps=5, target_text=txt)
                    adv_images.append(adv_img)
                    self.model.train()
                
                # Prepare inputs for CLIP
                inputs = self.processor(
                    text=texts,
                    images=adv_images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # CLIP loss (contrastive loss)
                logits_per_image = outputs.logits_per_image
                logits_per_text = outputs.logits_per_text
                
                # Ground truth labels are diagonal (0, 1, 2, ...)
                labels = torch.arange(len(images)).to(self.device)
                
                loss_i = nn.CrossEntropyLoss()(logits_per_image, labels)
                loss_t = nn.CrossEntropyLoss()(logits_per_text, labels)
                loss = (loss_i + loss_t) / 2
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if step % 10 == 0:
                    print(f"Step {step}, Loss: {loss.item():.4f}")
                
                step += 1
                
            if step >= max_steps:
                break
                
        print("Training finished.")
        self.save_model()
        
    def save_model(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        print(f"Saving model to {self.output_dir}...")
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        print("Model saved.")

if __name__ == "__main__":
    trainer = AdversarialTrainer()
    trainer.train(max_steps=50) # Run for 50 steps for demonstration
