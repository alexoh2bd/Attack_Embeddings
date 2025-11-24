#!/usr/bin/env python3
"""apply_jpeg_defense.py

Generation augmented by the help of Google Antigravity - model Gemini 3 Pro and Claude Sonnet 4.5 Thinking
CHAT HISTORY ATTACHED as Antigravity-Agentic-Coding-Chat-History.md

Apply JPEG compression defense to all adversarial images.

Usage:
    python apply_jpeg_defense.py --attacked_dir mmeb_eval_attacked --quality 75
"""

import os
import argparse
import json
from PIL import Image
from tqdm import tqdm

def apply_jpeg(img_path, output_path, quality=75):
    """Apply JPEG compression to an image."""
    img = Image.open(img_path)
    img.save(output_path, 'JPEG', quality=quality)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacked_dir", type=str, default="mmeb_eval_attacked")
    parser.add_argument("--qualities", type=int, nargs="+", default=[75, 50, 30], help="JPEG qualities (e.g. 75 50 30)")
    args = parser.parse_args()
    
    # Load metadata
    metadata_path = os.path.join(args.attacked_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"Applying JPEG defense for qualities: {args.qualities}")
    
    for quality in args.qualities:
        # Create JPEG subdirectory for this quality
        jpeg_dir = os.path.join(args.attacked_dir, f"jpeg_q{quality}")
        os.makedirs(jpeg_dir, exist_ok=True)
        
        print(f"Processing quality={quality}...")
        
        for item in tqdm(metadata):
            adv_path = os.path.join(args.attacked_dir, item['adv_path'])
            jpeg_filename = item['adv_path'].replace('.png', f'_q{quality}.png')
            jpeg_path = os.path.join(jpeg_dir, jpeg_filename)
            
            apply_jpeg(adv_path, jpeg_path, quality=quality)
            
            # Update metadata
            if 'jpeg_paths' not in item:
                item['jpeg_paths'] = {}
            item['jpeg_paths'][str(quality)] = os.path.join(f"jpeg_q{quality}", jpeg_filename)
    
    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nJPEG defense applied for qualities {args.qualities}!")
    print(f"Updated metadata: {metadata_path}")

if __name__ == "__main__":
    main()
