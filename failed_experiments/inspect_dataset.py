from datasets import load_dataset
import os

try:
    print("Loading MMEB-train dataset...")
    ds = load_dataset("TIGER-Lab/MMEB-train", "MSCOCO_i2t", split="original", streaming=True)
    print("Dataset loaded (streaming).")
    item = next(iter(ds))
    print("First item keys:", item.keys())
    print("First item sample:", item)
except Exception as e:
    print(f"Error: {e}")
