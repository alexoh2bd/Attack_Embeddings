#!/usr/bin/env python3
"""Apply JPEG compression defense to an image."""
import sys
from PIL import Image
import io

if len(sys.argv) != 3:
    print("Usage: python apply_jpeg.py input.png output.png")
    sys.exit(1)

input_path = sys.argv[1]
output_path = sys.argv[2]

# Load image
img = Image.open(input_path)

# Apply JPEG compression (quality=75)
buffer = io.BytesIO()
img.save(buffer, format='JPEG', quality=75)
buffer.seek(0)
img_jpeg = Image.open(buffer)

# Save back as PNG
img_jpeg.save(output_path)
print(f"✅ Applied JPEG compression (quality=75): {output_path}")
