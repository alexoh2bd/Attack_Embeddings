from huggingface_hub import hf_hub_download
from datasets import load_dataset
import zipfile, os
from PIL import Image, ImageFilter, ImageEnhance
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import logging
import gc
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Union
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import ndcg_score
import cv2
from pipe import MultimodalRetrievalPipeline
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel, AutoModelForCausalLM
# from attacks import pgd_attack

# model = AutoModelForCausalLM.from_pretrained("TIGER-Lab/VLM2Vec-Full", trust_remote_code=True, torch_dtype="auto")
torch.cuda.is_available()
def setup_ndcg_logger(log_file: str = None):
    """
    Setup a simple logger for NDCG results.
    
    Args:
        log_file: Path to log file (default: logs/ndcg_TIMESTAMP.log)
    
    Returns:
        logger instance
    """
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"ndcg_{timestamp}.log"
    
    logger = logging.getLogger('NDCG_Eval')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging to: {log_file}")
    return logger

def pgd_attack(model, image_tensor_01, text_tokens, normalize_fn, epsilon=0.0314, alpha=0.0078, steps=20, device='cuda'):
    """
    PGD attack - exact copy from attack_clip.py
    """
    x0 = image_tensor_01.clone().detach()
    
    # Random initialization
    x_adv = x0 + torch.empty_like(x0).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(0, 1)
    
    for _ in range(steps):
        x_adv.requires_grad_(True)
        
        # Normalize for CLIP
        x_input = normalize_fn(x_adv)
        
        # Get embeddings
        img_emb = model.encode_image(x_input)
        img_emb = F.normalize(img_emb, dim=-1)
        
        text_emb = model.encode_text(text_tokens)
        text_emb = F.normalize(text_emb, dim=-1)
        
        # Similarity
        sim = img_emb @ text_emb.T  # (B, L)
        
        # Loss: NEGATIVE similarity (we want to minimize it)
        loss = -sim.mean()
        
        # Backward
        model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            grad = x_adv.grad
            # PGD step
            x_adv = x_adv + alpha * grad.sign()
            # Project to epsilon ball
            x_adv = torch.max(torch.min(x_adv, x0 + epsilon), x0 - epsilon)
            x_adv = x_adv.clamp(0, 1)
        
        x_adv.requires_grad_(True)
    
    return x_adv.detach()
# This attack is based on the FGSM attack on CLIP to reduce image-text similarity.
def fgsm_attack_clip(image, pipeline, epsilon=0.03, target_text=None):
    """
    FGSM attack on CLIP to reduce image-text similarity.
    
    Args:
        image: PIL Image
        pipeline: MultimodalRetrievalPipeline with CLIP model
        epsilon: Perturbation magnitude (default 0.03)
        target_text: Optional specific text to attack against. If None, uses generic text.
    
    Returns:
        Perturbed PIL Image
    """
    # Use a generic text if none provided
    # This genereci text is aimed to reduce image-text similarity
    if target_text is None:
        target_text = "A photo of an object"
    
    # Convert image to tensor with CLIP preprocessing
    img_rgb = image.convert("RGB")
    transform = TF.Compose([
        TF.Resize((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                     std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    img_tensor = transform(img_rgb).unsqueeze(0).to(pipeline.model.device)
    img_tensor.requires_grad = True
    
    # Encode text (target to reduce similarity with)
    text_inputs = pipeline.processor(text=[target_text], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(pipeline.model.device) for k, v in text_inputs.items()}
    
    with torch.enable_grad():
        # Get embeddings
        text_embeds = pipeline.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        image_embeds = pipeline.model.get_image_features(pixel_values=img_tensor)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # Loss: maximize distance (minimize similarity)
        # We want to REDUCE relevance, so we minimize the negative similarity
        similarity = F.cosine_similarity(image_embeds, text_embeds)
        loss = similarity.mean()  # We'll add gradient to reduce this
        
        # Backprop
        pipeline.model.zero_grad()
        loss.backward()
        
        # FGSM: perturb in direction that INCREASES loss (reduces similarity)
        # Since we want to reduce similarity, we add epsilon * sign(gradient)
        sign_grad = img_tensor.grad.sign()
        perturbed_tensor = img_tensor + epsilon * sign_grad
        
        # Denormalize back to [0, 1] range
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(pipeline.model.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(pipeline.model.device)
        
        perturbed_tensor = perturbed_tensor * std + mean
        perturbed_tensor = torch.clamp(perturbed_tensor, 0, 1)
    
    # Convert back to PIL Image
    perturbed_np = perturbed_tensor.squeeze(0).detach().cpu().numpy()
    perturbed_np = (perturbed_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    perturbed_image = Image.fromarray(perturbed_np)
    
    # Resize back to original size if needed
    if perturbed_image.size != image.size:
        perturbed_image = perturbed_image.resize(image.size, Image.LANCZOS)
    
    return perturbed_image

# Perturb images function
def perturb(img, ptype, pipeline=None):
    if ptype == "gauss1":
        return img.filter(ImageFilter.GaussianBlur(radius=1))
    if ptype == "gauss2":
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    elif ptype== "grayscale":
        return img.convert('L')
    # increase brightness
    elif ptype=="bright":
        enhancer = ImageEnhance.Brightness(img)
        brightness_factor = 1.5
        brightened_image = enhancer.enhance(brightness_factor)
        return brightened_image
    elif ptype=="flip":
        flip_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return flip_img
    elif ptype=="compress":
        new_width, new_height = img.size[0] // 5 * 4, img.size[1] //5 *4
        return img.resize((new_width, new_height))
    elif ptype == "fgsm":
        assert pipeline is not None, "Pipeline required for FGSM attack"
        return fgsm_attack_clip(img, pipeline, epsilon=0.03)
    # control
    elif ptype == "pgd":
        assert pipeline is not None, "pipeline required for pgd"
        inputs = pipeline.processor(images=img, return_tensors="pt", padding=True)
        pixel_values = inputs['pixel_values'].to(pipeline.model.device)
        return pgd_attack(pipeline.model, pixel_values)
    else:
        return img

def i2t(test, pipeline, base_img_url, p):
    retrieved_count = 0
    perfect_count = 0
    scores = []
    for row in tqdm(test):
        # Load example image
        imgURL = base_img_url+ row['qry_img_path']
        image = Image.open(imgURL)
        
        p_img = perturb(image, p, pipeline)
            
        truth = [1] + [0]*999
        results = pipeline.retrieve_i2t(
            query_img = [p_img],
            corpus_text= row['tgt_text'],
        )
        
        # Clean up image after encoding
        if hasattr(p_img, 'close'):
            p_img.close()
        if hasattr(image, 'close'):
            image.close()
        
        # Calculate nDCG
        t = [truth[i] for i in results['indices'][0][:10]]
        score = ndcg_score([t],[results['scores'][0][:10].tolist()], k=10)
        if score > 0.0: 
            retrieved_count +=1
        if score == 1.0:
            perfect_count +=1

        scores.append(score)
    return scores, perfect_count, retrieved_count
def t2i(test, pipeline, base_img_url, p):
    retrieved_count = 0
    perfect_count = 0
    scores = []
    img_embed_dict = {}
    for row in tqdm(test):
        # Load example image
        # textURL = base_img_url+ row['qry_text']
        idx = 0 
        imgbatch_len = 64  # Match GPU batch size for optimal performance
        img_embeddings=[]
        hit_count = 0
        # encode images in batches
        num_batches = (len(row['tgt_img_path']) + imgbatch_len - 1) // imgbatch_len  # Ceiling division
        for batch_idx in range(num_batches):
            # Separate cached and non-cached images
            cached_embeddings = []
            images_to_encode = []
            urls_to_cache = []
            
            # Handle last batch which may be smaller
            batch_size = min(imgbatch_len, len(row['tgt_img_path']) - batch_idx * imgbatch_len)
            
            for j in range(batch_size):
                idx = batch_idx * imgbatch_len + j
                imgURL = base_img_url+ row['tgt_img_path'][idx] 

                if imgURL in img_embed_dict:
                    cached_embeddings.append((j, img_embed_dict[imgURL]))
                    hit_count += 1
                else:
                    # Load and perturb image (don't use context manager to avoid closing before encoding)
                    image = Image.open(imgURL)
                    p_img = perturb(image, p, pipeline)
                    images_to_encode.append((j, p_img))
                    urls_to_cache.append(imgURL)
            
            # Batch encode all non-cached images at once (single GPU transfer)
            if images_to_encode:
                imgs_only = [img for _, img in images_to_encode]
                batch_embeds = pipeline.encode_images(imgs_only)
                
                # Cache the new embeddings
                for i, imgURL in enumerate(urls_to_cache):
                    img_embed_dict[imgURL] = batch_embeds[i]
                
                # Combine cached and newly encoded embeddings in correct order
                newly_encoded = [(images_to_encode[i][0], batch_embeds[i]) for i in range(len(images_to_encode))]
                all_embeddings = cached_embeddings + newly_encoded
                all_embeddings.sort(key=lambda x: x[0])  # Sort by original position
                
                img_embeddings.extend([emb for _, emb in all_embeddings])
                
                # Clean up images and temporary data
                for img in imgs_only:
                    if hasattr(img, 'close'):
                        img.close()
                del batch_embeds, imgs_only, newly_encoded, all_embeddings, images_to_encode
            else:
                # All cached - just add them in order
                img_embeddings.extend([emb for _, emb in cached_embeddings])
                del images_to_encode
            
            # Clear lists for next batch
            del cached_embeddings, urls_to_cache
            
        # tqdm.write(f"Embedding Dict Hit count {hit_count}/{len(row['tgt_img_path'])}")
        
        # Stack embeddings efficiently - embeddings are already numpy arrays
        corpus_embeds = np.vstack(img_embeddings) if img_embeddings else np.array([])
        
        truth = [1] + [0]*999
        results = pipeline.retrieve_t2i(
            query_text =[row['qry_text']],
            corpus_img_embeds= corpus_embeds,
        )

        # Calculate nDCG
        t = [truth[i] for i in results['indices'][0][:10]]
        #print(results['scores'].shape, len(t), results['scores'])
        score = ndcg_score([t],[results['scores'][0][:10].tolist()], k=10)
        if score > 0.0: 
            retrieved_count +=1
        if score == 1.0:
            perfect_count +=1

        scores.append(score)
    return scores, perfect_count, retrieved_count

def deliverthegoods(datasets, perturbations, model_name):
    pipeline = MultimodalRetrievalPipeline(model_name)
    ndcg_scores = {}
    avg_ndcg_scores = {}
    base_img_url = '/work/aho13/MMEB-eval/'
    k=10

    i2tds = set(["MSCOCO_i2t","VisualNews_i2t"])
    t2ids = set(["MSCOCO_t2i","VisualNews_t2i", "VisDial", "Wiki-SS-NQ"])


    with torch.autocast(device_type='cuda', dtype=torch.float16):
        # Loops for each dataset
        for ds_name in datasets:
            #df = pd.read_csv(base_url+f"{ds_name}/test-00000-of-00001.parquet", encoding="utf-16")
            ds = load_dataset("TIGER-Lab/MMEB-eval", ds_name)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{os.getcwd()}/logs/{ds_name}_{timestamp}.log"
    
            logger = setup_ndcg_logger(log_file)

            for p in perturbations:
                scores=[]
                perfect_count = 0
                retrieved_count = 0
                if ds_name in i2tds:
                    scores, perfect_count, retrieved_count = i2t(list(ds['test']), pipeline, base_img_url, p)
                
                elif ds_name in t2ids:
                    scores, perfect_count, retrieved_count = t2i(list(ds['test']), pipeline, base_img_url, p)
                
                # Calculate average nDCG
                avg_ndcg = np.mean(scores)
                # Log summary
                if logger:
                    logger.info("="*60)
                    logger.info(f"NDCG@{k} {ds_name} {p} Results:")
                    logger.info(f"  Mean NDCG: {avg_ndcg:.4f}")
                    logger.info(f"  Median NDCG: {np.median(scores):.4f}")
                    logger.info(f"  Std Dev: {np.std(scores):.4f}")
                    logger.info(f"  Perfect retrievals (rank 1): {perfect_count}/{len(scores)} ({100*perfect_count/len(scores):.1f}%)")
                    logger.info(f"  Relevant in top-{k}: {retrieved_count}/{len(scores)} ({100*retrieved_count/len(scores):.1f}%)")
                    logger.info("="*60)
    pipeline.cleanup()
    return ndcg_scores


def main():

    # Visual News_i2t dataset and pipeline
    datasets= ["VisualNews_i2t", "MSCOCO_i2t"]
    # datasets= [ "Wiki-SS-NQ"]
    # datasets=["VisualNews_t2i","MSCOCO_t2i","VisDial","WIKI-SS-NQ"]
    # perturbations = ["flip", "gauss1","bright","grayscale"]
    # perturbations = ['gauss1','compress', 'flip']
    perturbations = ['pgd','ctrl']
    # model_name = "openai/clip-vit-base-patch32"
    ndcg_scores = deliverthegoods(datasets, perturbations,"openai/clip-vit-large-patch14")


if __name__ == "__main__":
    main()



