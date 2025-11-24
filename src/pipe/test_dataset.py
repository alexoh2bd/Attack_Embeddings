from datasets import load_dataset
import os

try:
    print("Loading dataset...")
    ds = load_dataset("TIGER-Lab/MMEB-eval", "MSCOCO_i2t", split="test[:5]")
    print("Dataset loaded.")
    print(f"Keys: {ds[0].keys()}")
    print(f"Example item: {ds[0]}")
    
    if 'image' in ds[0]:
        print(f"Image type: {type(ds[0]['image'])}")
except Exception as e:
    print(f"Error: {e}")
