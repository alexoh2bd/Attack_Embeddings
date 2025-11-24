import json

# Load notebook
with open('adversarial_demo.ipynb', 'r') as f:
    notebook = json.load(f)

# Find and fix the cell with the image size mismatch
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'source' in cell:
        source_text = ''.join(cell['source'])
        if 'clean_img = Image.open(clean_path).convert("RGB")' in source_text:
            # Replace the source
            cell['source'] = [
                "# Load and display images\n",
                "clean_path = os.path.join(\"mmeb_images\", example['original_path'])\n",
                "adv_path = os.path.join(\"mmeb_eval_attacked\", example['adv_path'])\n",
                "\n",
                "clean_img = Image.open(clean_path).convert(\"RGB\")\n",
                "adv_img = Image.open(adv_path).convert(\"RGB\")\n",
                "\n",
                "# Resize clean image to match adversarial image size (224x224)\n",
                "clean_img_resized = clean_img.resize((224, 224), Image.BICUBIC)\n",
                "\n",
                "# Compute perturbation (amplified for visualization)\n",
                "clean_arr = np.array(clean_img_resized).astype(np.float32) / 255.0\n",
                "adv_arr = np.array(adv_img).astype(np.float32) / 255.0\n",
                "perturbation = np.abs(adv_arr - clean_arr)\n",
                "perturbation_viz = (perturbation * 10).clip(0, 1)  # Amplify 10x for visibility\n",
                "\n",
                "# Visualize (use original size clean image for display)\n",
                "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
                "axes[0].imshow(clean_img)  # Show original size\n",
                "axes[0].set_title(\"Clean Image\")\n",
                "axes[0].axis('off')\n",
                "\n",
                "axes[1].imshow(adv_img)\n",
                "axes[1].set_title(\"Adversarial Image\\n(Looks identical!)\")\n",
                "axes[1].axis('off')\n",
                "\n",
                "axes[2].imshow(perturbation_viz)\n",
                "axes[2].set_title(\"Perturbation (10x amplified)\")\n",
                "axes[2].axis('off')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
            # Clear the outputs to force re-run
            cell['outputs'] = []
            cell['execution_count'] = None
            break

# Save notebook
with open('adversarial_demo.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✓ Fixed notebook - cell 3 updated to resize images before perturbation calculation")
