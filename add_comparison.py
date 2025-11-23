import json

# Load notebook
with open('adversarial_demo.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the defense section and add a new cell after the JPEG visualization
new_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Visual Comparison: All Defenses\n",
        "\n",
        "Let's visualize how each defense method affects the adversarial image:"
    ]
}

new_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Load all defense versions of the adversarial image\n",
        "fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n",
        "axes = axes.flatten()\n",
        "\n",
        "# Prepare images and similarity scores\n",
        "images = []\n",
        "titles = []\n",
        "similarities = []\n",
        "\n",
        "# 1. Original adversarial image\n",
        "adv_img = Image.open(adv_path).convert(\"RGB\")\n",
        "images.append(adv_img)\n",
        "titles.append(\"Adversarial (No Defense)\")\n",
        "similarities.append(adv_sim)\n",
        "\n",
        "# 2-4. JPEG defenses (q=75, 50, 30)\n",
        "for quality in ['75', '50', '30']:\n",
        "    if 'jpeg_paths' in example and quality in example['jpeg_paths']:\n",
        "        jpeg_path = os.path.join(\"mmeb_eval_attacked\", example['jpeg_paths'][quality])\n",
        "        if os.path.exists(jpeg_path):\n",
        "            jpeg_img = Image.open(jpeg_path).convert(\"RGB\")\n",
        "            jpeg_sim = evaluate_similarity(model, preprocess, jpeg_path, example['text'], device)\n",
        "            images.append(jpeg_img)\n",
        "            titles.append(f\"JPEG (q={quality})\")\n",
        "            similarities.append(jpeg_sim)\n",
        "\n",
        "# 5. MAT defense (if available)\n",
        "if mat_model is not None:\n",
        "    # MAT applies defense during inference, not preprocessing\n",
        "    # So we show the adversarial image but evaluated with MAT model\n",
        "    images.append(adv_img)  # Same image\n",
        "    titles.append(\"MAT Model Defense\")\n",
        "    mat_adv_sim = evaluate_similarity(mat_model, mat_preprocess, adv_path, example['text'], device)\n",
        "    similarities.append(mat_adv_sim)\n",
        "\n",
        "# 6. PGD defense (if available)\n",
        "if pgd_model is not None:\n",
        "    # PGD applies defense during inference, not preprocessing\n",
        "    images.append(adv_img)  # Same image\n",
        "    titles.append(\"PGD Model Defense\")\n",
        "    pgd_adv_sim = evaluate_similarity(pgd_model, pgd_preprocess, adv_path, example['text'], device)\n",
        "    similarities.append(pgd_adv_sim)\n",
        "\n",
        "# Plot all images\n",
        "for i, (img, title, sim) in enumerate(zip(images, titles, similarities)):\n",
        "    if i < len(axes):\n",
        "        axes[i].imshow(img)\n",
        "        status = \"✓\" if sim > 0 else \"✗\"\n",
        "        color = 'green' if sim > 0 else 'red'\n",
        "        axes[i].set_title(f\"{title}\\nSimilarity: {sim:+.4f} {status}\", \n",
        "                         fontsize=12, color=color, weight='bold')\n",
        "        axes[i].axis('off')\n",
        "\n",
        "# Hide empty subplots\n",
        "for i in range(len(images), len(axes)):\n",
        "    axes[i].axis('off')\n",
        "\n",
        "plt.suptitle(f'Defense Comparison: \"{example[\"text\"]}\"', \n",
        "             fontsize=14, weight='bold', y=0.98)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Print summary\n",
        "print(\"\\n\" + \"=\"*70)\n",
        "print(\"Image Comparison Summary\")\n",
        "print(\"=\"*70)\n",
        "for title, sim in zip(titles, similarities):\n",
        "    status = \"✓ DEFENDED\" if sim > 0 else \"✗ BROKEN\"\n",
        "    improvement = ((adv_sim - sim) / abs(adv_sim)) * 100 if adv_sim != 0 else 0\n",
        "    print(f\"{title:<30} Sim: {sim:+.4f}  {status:<12} Recovery: {improvement:+.1f}%\")\n",
        "print(\"=\"*70)"
    ]
}

# Find where to insert (after the bar chart visualization)
insert_index = None
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source_text = ''.join(cell['source'])
        if 'plt.show()' in source_text and 'bars1' in source_text:
            insert_index = i + 1
            break

if insert_index:
    notebook['cells'].insert(insert_index, new_cell)
    notebook['cells'].insert(insert_index + 1, new_code_cell)
    print(f"✓ Added visual comparison cells at position {insert_index}")
else:
    # Add at the end of defenses section
    notebook['cells'].insert(-1, new_cell)
    notebook['cells'].insert(-1, new_code_cell)
    print("✓ Added visual comparison cells at end of defenses section")

# Save notebook
with open('adversarial_demo.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✓ Updated notebook with image comparison visualization")
