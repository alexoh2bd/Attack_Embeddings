# ===== CLIP: prompt-ensemble + image TTA to boost similarity =====
import torch, torch.nn.functional as F
from PIL import Image
import clip
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "ViT-L/14"
model, preprocess = clip.load(model_name, device=device)
model.eval()

img_path = "/home/gss/duke/Attack_Embeddings/sample_image.jpg"
pil_image = Image.open(img_path).convert("RGB")

# labels you care about
labels = ["white dog", "fluffy white dog", "dog sitting on grass", "cat", "car", "person", "fox"]

# prompt templates (expandable)
templates = [
    "a photo of a {}",
    "a photograph of a {}",
    "a close-up of a {}",
    "a picture of a {}",
    "a cropped photo of a {}",
    "a professional photo of a {}",
    "a high quality photo of a {}",
    "an image of a {}"
]

# build prompt list and map index->label
prompts = []
idx_map = []  # which label each prompt belongs to
for i, lbl in enumerate(labels):
    for t in templates:
        prompts.append(t.format(lbl))
        idx_map.append(i)

# tokenize and get text embeddings in batch
text_tokens = clip.tokenize(prompts).to(device)
with torch.no_grad():
    txt_feats = model.encode_text(text_tokens)   # (P, D)
    txt_feats = F.normalize(txt_feats, dim=-1)

# average embeddings per label
label_embs = []
for i in range(len(labels)):
    inds = [j for j,k in enumerate(idx_map) if k==i]
    avg = txt_feats[inds].mean(dim=0, keepdim=True)
    avg = F.normalize(avg, dim=-1)
    label_embs.append(avg)
label_embs = torch.cat(label_embs, dim=0)  # (L, D)

# Image TTA: get many small transforms and average image embeddings
tta_transforms = [
    lambda im: preprocess(im),
    lambda im: preprocess(im.resize((int(im.width*0.95), int(im.height*0.95)))),
    lambda im: preprocess(im.resize((int(im.width*1.05), int(im.height*1.05)))),
    lambda im: preprocess(im.crop((0,0,im.width-10,im.height-10)).resize((im.width,im.height))),
    lambda im: preprocess(im.crop((10,10,im.width,im.height)).resize((im.width,im.height))),
    lambda im: preprocess(im).flip(-1) if hasattr(torch.Tensor, 'flip') else preprocess(im)  # fallback
]

img_inputs = torch.stack([t(pil_image) for t in tta_transforms]).to(device)  # (T,3,H,W)
with torch.no_grad():
    img_feats = model.encode_image(img_inputs)   # (T, D)
    img_feats = F.normalize(img_feats, dim=-1)
    img_emb = img_feats.mean(dim=0, keepdim=True)   # (1, D)
    img_emb = F.normalize(img_emb, dim=-1)

# compute cosines, logits (with logit_scale), probs
cosines = (img_emb @ label_embs.T).squeeze(0)   # (L,)
logits = cosines * model.logit_scale.exp()
probs = logits.softmax(dim=-1)

# print results sorted
vals = list(zip(labels, cosines.detach().cpu().numpy(), probs.detach().cpu().numpy()))
vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)
print("Label\t\tcosine\tprob")
for lbl, c, p in vals_sorted:
    print(f"{lbl:25s}\t{c:6.3f}\t{p:6.3f}")

# top-k
topk = torch.topk(cosines, k=min(5, len(labels)))
print("\nTop-k (cosine):")
for idx, v in zip(topk.indices.cpu().numpy(), topk.values.cpu().numpy()):
    print(f"{labels[idx]:25s}\t{v:6.3f}")
