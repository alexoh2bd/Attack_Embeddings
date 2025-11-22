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


from attacks import fgsm_attack_clip, pgd_attack_clip

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
    elif ptype == "pgd":
        assert pipeline is not None, "Pipeline required for PGD attack"
        return pgd_attack_clip(img, pipeline, epsilon=0.03, steps=10)
    # control
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
    base_img_url = '/work/aho13/work/aho13/'
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
    # datasets= ["VisualNews_i2t"]
    # datasets= [ "Wiki-SS-NQ"]
    datasets=["VisualNews_t2i","MSCOCO_t2i","VisDial","WIKI-SS-NQ"]
    # perturbations = ["flip", "gauss1","bright","grayscale"]
    # perturbations = ['gauss1','compress', 'flip']
    perturbations = ["ctrl",'fgsm']
    model_name = "openai/clip-vit-base-patch32"
    ndcg_scores = deliverthegoods(datasets, perturbations, model_name)


if __name__ == "__main__":
    main()



