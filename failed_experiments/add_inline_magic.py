import json, os

NOTEBOOK_PATH = '/home/gss/duke/Attack_Embeddings/adversarial_demo.ipynb'

def load_notebook(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

def ensure_inline_magic(nb):
    # Find first code cell (skip any markdown at top)
    for idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            # Insert %matplotlib inline at the very start of this cell if not present
            source = cell.get('source', [])
            if not any('%matplotlib inline' in line for line in source):
                new_source = ['%matplotlib inline\n'] + source
                cell['source'] = new_source
                nb['cells'][idx] = cell
                print(f"Inserted %matplotlib inline into cell {idx}")
            else:
                print('Inline magic already present')
            break
    return nb

if __name__ == '__main__':
    if not os.path.exists(NOTEBOOK_PATH):
        raise FileNotFoundError(f'Notebook not found: {NOTEBOOK_PATH}')
    nb = load_notebook(NOTEBOOK_PATH)
    nb = ensure_inline_magic(nb)
    save_notebook(nb, NOTEBOOK_PATH)
    print('Notebook updated with %matplotlib inline')
