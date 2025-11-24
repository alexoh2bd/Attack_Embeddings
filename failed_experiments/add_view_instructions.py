import json, os

NOTEBOOK_PATH = '/home/gss/duke/Attack_Embeddings/adversarial_demo.ipynb'

def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def insert_instruction_cell(nb):
    instruction_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# How to view the images\n",
            "\n",
            "This notebook contains several visualizations using **matplotlib**. To see the images correctly, make sure you are running the notebook in a Jupyter environment (e.g., Jupyter Lab, Jupyter Notebook, VS Code notebook).\n",
            "\n",
            "* The first code cell now includes `%matplotlib inline` so plots are rendered inline.\n",
            "* Run **all cells** (Kernel → Restart & Run All) to generate the figures.\n",
            "* If you are opening the notebook in a static viewer (e.g., GitHub), the images will not render because they require execution.\n",
            "\n",
            "Once you execute the notebook, you should see the clean, adversarial, JPEG‑compressed, MAT‑defended, and PGD‑defended images for each example.\n"
        ]
    }
    # Insert after any initial markdown cells (keep existing title)
    insert_idx = 0
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'markdown':
            insert_idx = i + 1
        else:
            break
    nb['cells'].insert(insert_idx, instruction_md)
    print(f"Inserted instruction markdown cell at index {insert_idx}")
    return nb

if __name__ == '__main__':
    if not os.path.exists(NOTEBOOK_PATH):
        raise FileNotFoundError(f'Notebook not found: {NOTEBOOK_PATH}')
    nb = load_notebook(NOTEBOOK_PATH)
    nb = insert_instruction_cell(nb)
    save_notebook(nb, NOTEBOOK_PATH)
    print('Notebook updated with viewing instructions')
