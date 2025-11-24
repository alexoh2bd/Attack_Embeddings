import json

# Load notebook
with open('adversarial_demo.ipynb', 'r') as f:
    notebook = json.load(f)

# Update the visual grid cell to show each example separately
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source_text = ''.join(cell['source'])
        
        # Update the visual grid cell
        if 'Visual grid of multiple examples with clean images and captions' in source_text:
            cell['source'] = [
                "# Display each example as a separate figure for better clarity\n",
                "num_display = min(3, len(all_results))  # Display 3 examples\n",
                "\n",
                "defense_names = ['Clean', 'Adversarial', 'JPEG q=75', 'JPEG q=50', 'JPEG q=30', 'MAT', 'PGD']\n",
                "\n",
                "for example_idx in range(num_display):\n",
                "    example = metadata[example_idx]\n",
                "    c_path = os.path.join(\"mmeb_images\", example['original_path'])\n",
                "    a_path = os.path.join(\"mmeb_eval_attacked\", example['adv_path'])\n",
                "    \n",
                "    # Create figure for this example\n",
                "    fig, axes = plt.subplots(1, 7, figsize=(24, 4))\n",
                "    \n",
                "    # Load images\n",
                "    clean_img = Image.open(c_path).convert(\"RGB\")\n",
                "    adv_img = Image.open(a_path).convert(\"RGB\")\n",
                "    \n",
                "    # Column 0: Clean image\n",
                "    axes[0].imshow(clean_img)\n",
                "    clean_sim = example['clean_similarity']\n",
                "    axes[0].set_title(f\"Clean\\nSim: {clean_sim:+.3f}\", \n",
                "                      fontsize=11, color='blue', weight='bold')\n",
                "    axes[0].axis('off')\n",
                "    \n",
                "    # Column 1: Adversarial\n",
                "    axes[1].imshow(adv_img)\n",
                "    adv_sim = example['adv_similarity']\n",
                "    axes[1].set_title(f\"Adversarial\\nSim: {adv_sim:+.3f}\", \n",
                "                      fontsize=11, color='red', weight='bold')\n",
                "    axes[1].axis('off')\n",
                "    \n",
                "    # Columns 2-4: JPEG defenses\n",
                "    baseline_drop = clean_sim - adv_sim\n",
                "    for col_idx, q in enumerate(['75', '50', '30'], start=2):\n",
                "        if 'jpeg_paths' in example and q in example['jpeg_paths']:\n",
                "            jpeg_path = os.path.join(\"mmeb_eval_attacked\", example['jpeg_paths'][q])\n",
                "            if os.path.exists(jpeg_path):\n",
                "                jpeg_img = Image.open(jpeg_path).convert(\"RGB\")\n",
                "                jpeg_sim = all_results[example_idx].get(f'jpeg_{q}', 0)\n",
                "                \n",
                "                # Calculate improvement\n",
                "                defense_drop = clean_sim - jpeg_sim\n",
                "                improvement = ((baseline_drop - defense_drop) / baseline_drop * 100) if baseline_drop > 0 else 0\n",
                "                \n",
                "                color = 'green' if jpeg_sim > 0 else 'red'\n",
                "                axes[col_idx].imshow(jpeg_img)\n",
                "                axes[col_idx].set_title(f\"JPEG q={q}\\nSim: {jpeg_sim:+.3f}\\n({improvement:+.0f}%)\", \n",
                "                                        fontsize=11, color=color, weight='bold')\n",
                "        axes[col_idx].axis('off')\n",
                "    \n",
                "    # Column 5: MAT\n",
                "    if mat_model is not None:\n",
                "        mat_sim = all_results[example_idx].get('mat', 0)\n",
                "        defense_drop = clean_sim - mat_sim\n",
                "        improvement = ((baseline_drop - defense_drop) / baseline_drop * 100) if baseline_drop > 0 else 0\n",
                "        color = 'green' if mat_sim > 0 else 'red'\n",
                "        axes[5].imshow(adv_img)  # Same image, different model\n",
                "        axes[5].set_title(f\"MAT\\nSim: {mat_sim:+.3f}\\n({improvement:+.0f}%)\", \n",
                "                          fontsize=11, color=color, weight='bold')\n",
                "    axes[5].axis('off')\n",
                "    \n",
                "    # Column 6: PGD\n",
                "    if pgd_model is not None:\n",
                "        pgd_sim = all_results[example_idx].get('pgd', 0)\n",
                "        defense_drop = clean_sim - pgd_sim\n",
                "        improvement = ((baseline_drop - defense_drop) / baseline_drop * 100) if baseline_drop > 0 else 0\n",
                "        color = 'green' if pgd_sim > 0 else 'red'\n",
                "        axes[6].imshow(adv_img)  # Same image, different model\n",
                "        axes[6].set_title(f\"PGD\\nSim: {pgd_sim:+.3f}\\n({improvement:+.0f}%)\", \n",
                "                          fontsize=11, color=color, weight='bold')\n",
                "    axes[6].axis('off')\n",
                "    \n",
                "    # Add caption as suptitle\n",
                "    caption = example['text']\n",
                "    plt.suptitle(f\"Example #{example_idx}: {caption}\", \n",
                "                 fontsize=13, weight='bold', y=1.02)\n",
                "    \n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "    print()\n"
            ]
            cell['outputs'] = []
            cell['execution_count'] = None
            print("✓ Updated to show each example as a separate figure")
            break

# Save notebook
with open('adversarial_demo.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✓ Notebook updated successfully!")
print("  - Each example now displayed in its own figure")
print("  - 1 row × 7 columns per example")
print("  - Caption as title above each figure")
print("  - Improvement percentages shown for each defense")
