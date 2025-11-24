import json

# Load notebook
with open('adversarial_demo.ipynb', 'r') as f:
    notebook = json.load(f)

# Update the visual grid cell to include clean images and captions
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source_text = ''.join(cell['source'])
        
        # Update the visual grid cell
        if 'Visual grid of multiple examples' in source_text and 'num_display' in source_text:
            cell['source'] = [
                "# Visual grid of multiple examples with clean images and captions\n",
                "num_display = min(3, len(all_results))  # Display 3 examples visually\n",
                "\n",
                "fig, axes = plt.subplots(num_display, 7, figsize=(24, num_display * 3.5))\n",
                "if num_display == 1:\n",
                "    axes = axes.reshape(1, -1)\n",
                "\n",
                "defense_names = ['Clean', 'Adversarial', 'JPEG q=75', 'JPEG q=50', 'JPEG q=30', 'MAT', 'PGD']\n",
                "\n",
                "for row_idx in range(num_display):\n",
                "    example = metadata[row_idx]\n",
                "    c_path = os.path.join(\"mmeb_images\", example['original_path'])\n",
                "    a_path = os.path.join(\"mmeb_eval_attacked\", example['adv_path'])\n",
                "    \n",
                "    # Load images\n",
                "    clean_img = Image.open(c_path).convert(\"RGB\")\n",
                "    adv_img = Image.open(a_path).convert(\"RGB\")\n",
                "    \n",
                "    # Column 0: Clean image\n",
                "    axes[row_idx, 0].imshow(clean_img)\n",
                "    clean_sim = example['clean_similarity']\n",
                "    axes[row_idx, 0].set_title(f\"Clean\\n{clean_sim:+.3f}\", \n",
                "                                fontsize=9, color='blue', weight='bold')\n",
                "    axes[row_idx, 0].axis('off')\n",
                "    \n",
                "    # Column 1: Adversarial\n",
                "    axes[row_idx, 1].imshow(adv_img)\n",
                "    axes[row_idx, 1].set_title(f\"Adversarial\\n{example['adv_similarity']:+.3f}\", \n",
                "                                fontsize=9, color='red', weight='bold')\n",
                "    axes[row_idx, 1].axis('off')\n",
                "    \n",
                "    # Columns 2-4: JPEG defenses\n",
                "    for col_idx, q in enumerate(['75', '50', '30'], start=2):\n",
                "        if 'jpeg_paths' in example and q in example['jpeg_paths']:\n",
                "            jpeg_path = os.path.join(\"mmeb_eval_attacked\", example['jpeg_paths'][q])\n",
                "            if os.path.exists(jpeg_path):\n",
                "                jpeg_img = Image.open(jpeg_path).convert(\"RGB\")\n",
                "                jpeg_sim = all_results[row_idx].get(f'jpeg_{q}', 0)\n",
                "                color = 'green' if jpeg_sim > 0 else 'red'\n",
                "                axes[row_idx, col_idx].imshow(jpeg_img)\n",
                "                axes[row_idx, col_idx].set_title(f\"JPEG q={q}\\n{jpeg_sim:+.3f}\", \n",
                "                                                  fontsize=9, color=color, weight='bold')\n",
                "        axes[row_idx, col_idx].axis('off')\n",
                "    \n",
                "    # Column 5: MAT\n",
                "    if mat_model is not None:\n",
                "        mat_sim = all_results[row_idx].get('mat', 0)\n",
                "        color = 'green' if mat_sim > 0 else 'red'\n",
                "        axes[row_idx, 5].imshow(adv_img)  # Same image, different model\n",
                "        axes[row_idx, 5].set_title(f\"MAT\\n{mat_sim:+.3f}\", \n",
                "                                    fontsize=9, color=color, weight='bold')\n",
                "    axes[row_idx, 5].axis('off')\n",
                "    \n",
                "    # Column 6: PGD\n",
                "    if pgd_model is not None:\n",
                "        pgd_sim = all_results[row_idx].get('pgd', 0)\n",
                "        color = 'green' if pgd_sim > 0 else 'red'\n",
                "        axes[row_idx, 6].imshow(adv_img)  # Same image, different model\n",
                "        axes[row_idx, 6].set_title(f\"PGD\\n{pgd_sim:+.3f}\", \n",
                "                                    fontsize=9, color=color, weight='bold')\n",
                "    axes[row_idx, 6].axis('off')\n",
                "    \n",
                "    # Add caption below the row\n",
                "    caption = example['text']\n",
                "    fig.text(0.1, 0.98 - (row_idx + 1) / num_display, \n",
                "             f\"Example #{row_idx}: {caption}\",\n",
                "             fontsize=10, ha='left', va='top', weight='bold',\n",
                "             transform=fig.transFigure, wrap=True)\n",
                "\n",
                "plt.suptitle('Defense Performance Across Multiple Examples (with Clean Images)', \n",
                "             fontsize=14, weight='bold', y=0.995)\n",
                "plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for captions\n",
                "plt.show()"
            ]
            cell['outputs'] = []
            cell['execution_count'] = None
            print("✓ Updated visual grid to include clean images and captions")
            break

# Save notebook
with open('adversarial_demo.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("✓ Notebook updated successfully!")
print("  - Grid now shows: Clean | Adversarial | JPEG (3) | MAT | PGD")
print("  - Captions displayed above each row")
print("  - Clean image similarity score shown")
