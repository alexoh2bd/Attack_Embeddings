#!/usr/bin/env python3
"""download_mscoco.py

Download MSCOCO images for MMEB dataset training.
This script downloads a subset of MSCOCO images to a local directory.

Usage:
    python download_mscoco.py --num_images 1000
"""

import argparse
import os
import requests
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from io import BytesIO

def download_mscoco_images(num_images=1000, output_dir="mmeb_images"):
    """
    Download MSCOCO images from the MMEB dataset.
    
    Args:
        num_images: Number of images to download
        output_dir: Output directory for images
    """
    print(f"Loading MSCOCO_i2t dataset metadata...")
    ds = load_dataset("TIGER-Lab/MMEB-eval", "MSCOCO_i2t", split="test")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {min(num_images, len(ds))} images to {output_dir}...")
    
    downloaded = 0
    skipped = 0
    
    for i, row in enumerate(tqdm(ds)):
        if downloaded >= num_images:
            break
            
        # Get image path from dataset
        img_path = row['qry_img_path']
        
        # Create full output path
        full_output_path = os.path.join(output_dir, img_path)
        
        # Skip if already exists
        if os.path.exists(full_output_path):
            skipped += 1
            continue
        
        # Create directory structure
        os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
        
        # Try to download image (MSCOCO images are publicly available)
        # Extract image ID from path (e.g., "MSCOCO_i2t/COCO_train2014_000000123456.jpg")
        img_filename = os.path.basename(img_path)
        
        # MSCOCO images are available at http://images.cocodataset.org/
        # Format: http://images.cocodataset.org/train2014/COCO_train2014_000000123456.jpg
        if 'train2014' in img_filename:
            url = f"http://images.cocodataset.org/train2014/{img_filename}"
        elif 'val2014' in img_filename:
            url = f"http://images.cocodataset.org/val2014/{img_filename}"
        else:
            print(f"Unknown image type: {img_filename}")
            continue
        
        try:
            response = requests.get(url, timeout=10, stream=True)
            response.raise_for_status()
            
            # Save image
            img = Image.open(BytesIO(response.content))
            img.save(full_output_path)
            downloaded += 1
            
        except Exception as e:
            print(f"Failed to download {img_filename}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Download complete!")
    print(f"  Downloaded: {downloaded} images")
    print(f"  Skipped (already exists): {skipped} images")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images", type=int, default=1000,
                        help="Number of images to download (default: 1000)")
    parser.add_argument("--output_dir", type=str, default="mmeb_images",
                        help="Output directory for images (default: mmeb_images)")
    args = parser.parse_args()
    
    download_mscoco_images(args.num_images, args.output_dir)

if __name__ == "__main__":
    main()
