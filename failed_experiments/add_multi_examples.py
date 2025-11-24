import json

# Load notebook
with open('adversarial_demo.ipynb', 'r') as f:
    notebook = json.load(f)

# Add a new section for multiple examples
new_section_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Multiple Example Comparison\n",
        "\n",
        "Let's evaluate all defenses across multiple examples to see consistency:"
    ]
}

new_multi_example_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Evaluate multiple examples\n",
        "num_examples = min(5, len(metadata))  # Show up to 5 examples\n",
        "\n",
        "# Collect results for all examples\n",
        "all_results = []\n",
        "\n",
        "for idx in range(num_examples):\n",
        "    example = metadata[idx]\n",
        "    c_path = os.path.join(\"mmeb_images\", example['original_path'])\n",
        "    a_path = os.path.join(\"mmeb_eval_attacked\", example['adv_path'])\n",
        "    \n",
        "    if not os.path.exists(c_path):\n",
        "        continue\n",
        "    \n",
        "    ex_results = {\n",
        "        'idx': idx,\n",
        "        'text': example['text'][:50] + '...' if len(example['text']) > 50 else example['text'],\n",
        "        'clean_sim': example['clean_similarity']\n",
        "    }\n",
        "    \n",
        "    # Original (no defense)\n",
        "    ex_results['original'] = example['adv_similarity']\n",
        "    \n",
        "    # JPEG defenses\n",
        "    for q in ['75', '50', '30']:\n",
        "        if 'jpeg_paths' in example and q in example['jpeg_paths']:\n",
        "            jpeg_path = os.path.join(\"mmeb_eval_attacked\", example['jpeg_paths'][q])\n",
        "            if os.path.exists(jpeg_path):\n",
        "                ex_results[f'jpeg_{q}'] = evaluate_similarity(model, preprocess, jpeg_path, example['text'], device)\n",
        "    \n",
        "    # MAT defense\n",
        "    if mat_model is not None:\n",
        "        ex_results['mat'] = evaluate_similarity(mat_model, mat_preprocess, a_path, example['text'], device)\n",
        "    \n",
        "    # PGD defense\n",
        "    if pgd_model is not None:\n",
        "        ex_results['pgd'] = evaluate_similarity(pgd_model, pgd_preprocess, a_path, example['text'], device)\n",
        "    \n",
        "    all_results.append(ex_results)\n",
        "\n",
        "# Create comparison table\n",
        "print(f\"\\n{'='*120}\")\n",
        "print(f\"Multi-Example Defense Comparison ({num_examples} samples)\")\n",
        "print(f\"{'='*120}\")\n",
        "print(f\"{'#':<3} {'Text':<50} {'Clean':<8} {'Adv':<8} {'JPEG75':<8} {'JPEG50':<8} {'JPEG30':<8} {'MAT':<8} {'PGD':<8}\")\n",
        "print(f\"{'-'*120}\")\n",
        "\n",
        "for res in all_results:\n",
        "    print(f\"{res['idx']:<3} {res['text']:<50} \"\n",
        "          f\"{res['clean_sim']:>+.4f}  \"\n",
        "          f\"{res.get('original', 0):>+.4f}  \"\n",
        "          f\"{res.get('jpeg_75', 0):>+.4f}  \"\n",
        "          f\"{res.get('jpeg_50', 0):>+.4f}  \"\n",
        "          f\"{res.get('jpeg_30', 0):>+.4f}  \"\n",
        "          f\"{res.get('mat', 0):>+.4f}  \"\n",
        "          f\"{res.get('pgd', 0):>+.4f}\")\n",
        "\n",
        "print(f\"{'-'*120}\")\n",
        "\n",
        "# Calculate average performance\n",
        "avg_clean = np.mean([r['clean_sim'] for r in all_results])\n",
        "avg_original = np.mean([r.get('original', 0) for r in all_results])\n",
        "avg_jpeg75 = np.mean([r.get('jpeg_75', 0) for r in all_results if 'jpeg_75' in r])\n",
        "avg_jpeg50 = np.mean([r.get('jpeg_50', 0) for r in all_results if 'jpeg_50' in r])\n",
        "avg_jpeg30 = np.mean([r.get('jpeg_30', 0) for r in all_results if 'jpeg_30' in r])\n",
        "avg_mat = np.mean([r.get('mat', 0) for r in all_results if 'mat' in r])\n",
        "avg_pgd = np.mean([r.get('pgd', 0) for r in all_results if 'pgd' in r])\n",
        "\n",
        "print(f\"{'AVG':<3} {'':<50} \"\n",
        "      f\"{avg_clean:>+.4f}  \"\n",
        "      f\"{avg_original:>+.4f}  \"\n",
        "      f\"{avg_jpeg75:>+.4f}  \"\n",
        "      f\"{avg_jpeg50:>+.4f}  \"\n",
        "      f\"{avg_jpeg30:>+.4f}  \"\n",
        "      f\"{avg_mat:>+.4f}  \"\n",
        "      f\"{avg_pgd:>+.4f}\")\n",
        "\n",
        "print(f\"{'='*120}\")\n",
        "\n",
        "# Calculate improvement percentages\n",
        "baseline_drop = avg_clean - avg_original\n",
        "print(f\"\\nAverage Improvement vs Baseline:\")\n",
        "for name, avg_sim in [('JPEG q=75', avg_jpeg75), ('JPEG q=50', avg_jpeg50), \n",
        "                       ('JPEG q=30', avg_jpeg30), ('MAT', avg_mat), ('PGD', avg_pgd)]:\n",
        "    defense_drop = avg_clean - avg_sim\n",
        "    improvement = ((baseline_drop - defense_drop) / baseline_drop * 100) if baseline_drop > 0 else 0\n",
        "    status = \"✓\" if avg_sim > 0 else \"✗\"\n",
        "    print(f\"  {name:<15} Avg Sim: {avg_sim:>+.4f} {status}  Improvement: {improvement:>+.1f}%\")\n",
        "print(f\"{'='*120}\")"
    ]
}

new_visual_grid_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Visual grid of multiple examples\n",
        "num_display = min(3, len(all_results))  # Display 3 examples visually\n",
        "\n",
        "fig, axes = plt.subplots(num_display, 6, figsize=(20, num_display * 3.5))\n",
        "if num_display == 1:\n",
        "    axes = axes.reshape(1, -1)\n",
        "\n",
        "defense_names = ['Adversarial', 'JPEG q=75', 'JPEG q=50', 'JPEG q=30', 'MAT', 'PGD']\n",
        "\n",
        "for row_idx in range(num_display):\n",
        "    example = metadata[row_idx]\n",
        "    a_path = os.path.join(\"mmeb_eval_attacked\", example['adv_path'])\n",
        "    adv_img = Image.open(a_path).convert(\"RGB\")\n",
        "    \n",
        "    # Column 0: Adversarial\n",
        "    axes[row_idx, 0].imshow(adv_img)\n",
        "    axes[row_idx, 0].set_title(f\"Adversarial\\n{example['adv_similarity']:+.3f}\", \n",
        "                                fontsize=9, color='red', weight='bold')\n",
        "    axes[row_idx, 0].axis('off')\n",
        "    \n",
        "    # Columns 1-3: JPEG defenses\n",
        "    for col_idx, q in enumerate(['75', '50', '30'], start=1):\n",
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
        "    # Column 4: MAT\n",
        "    if mat_model is not None:\n",
        "        mat_sim = all_results[row_idx].get('mat', 0)\n",
        "        color = 'green' if mat_sim > 0 else 'red'\n",
        "        axes[row_idx, 4].imshow(adv_img)  # Same image, different model\n",
        "        axes[row_idx, 4].set_title(f\"MAT\\n{mat_sim:+.3f}\", \n",
        "                                    fontsize=9, color=color, weight='bold')\n",
        "    axes[row_idx, 4].axis('off')\n",
        "    \n",
        "    # Column 5: PGD\n",
        "    if pgd_model is not None:\n",
        "        pgd_sim = all_results[row_idx].get('pgd', 0)\n",
        "        color = 'green' if pgd_sim > 0 else 'red'\n",
        "        axes[row_idx, 5].imshow(adv_img)  # Same image, different model\n",
        "        axes[row_idx, 5].set_title(f\"PGD\\n{pgd_sim:+.3f}\", \n",
        "                                    fontsize=9, color=color, weight='bold')\n",
        "    axes[row_idx, 5].axis('off')\n",
        "    \n",
        "    # Add text description on the left\n",
        "    axes[row_idx, 0].text(-0.5, 0.5, f'#{row_idx}', \n",
        "                          transform=axes[row_idx, 0].transAxes,\n",
        "                          fontsize=12, weight='bold', ha='right', va='center')\n",
        "\n",
        "plt.suptitle('Defense Performance Across Multiple Examples', fontsize=14, weight='bold', y=0.99)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}

# Find where to insert (at the end, before final summary)
insert_position = len(notebook['cells']) - 1  # Before the last markdown cell

notebook['cells'].insert(insert_position, new_section_markdown)
notebook['cells'].insert(insert_position + 1, new_multi_example_cell)
notebook['cells'].insert(insert_position + 2, new_visual_grid_cell)

# Save notebook
with open('adversarial_demo.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"✓ Added multi-example comparison section with 2 cells")
print(f"  - Table comparing 5 examples across all defenses")
print(f"  - Visual grid showing 3 examples side-by-side")
print("✓ Notebook updated successfully!")
