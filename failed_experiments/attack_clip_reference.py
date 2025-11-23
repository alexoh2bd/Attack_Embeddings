#!/usr/bin/env python3
"""attack_clip.py

Run an untargeted PGD (L-inf) attack against a CLIP model on a single image.
Saves adversarial image and prints before/after top-k captions and cosine/prob metrics.

Usage example:
    python attack_clip.py --img sample_image.jpg --labels "white dog" "cat" "car" --eps 8/255 --iters 20 --restarts 5

Requirements:
    pip install git+https://github.com/openai/CLIP.git torch pillow numpy
"""

import argparse, os, io, sys
import torch, torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, Normalize

# CLIP Normalization constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def save_tensor_image(x, out_path):
    # x: (1,C,H,W) in [0,1] or (C,H,W)
    if x.dim() == 4:
        x = x[0]
    arr = (x.detach().cpu().permute(1,2,0).numpy() * 255.0).clip(0,255).astype(np.uint8)
    img = Image.fromarray(arr)
    img.save(out_path)

def load_clip_model(model_name, device):
    import clip
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess

def image_tta_emb(model, preprocess, pil_img, device, dtype=None, tta_transforms=None):
    # returns normalized averaged image embedding (1,D)
    if tta_transforms is None:
        tta_transforms = [
            lambda im: preprocess(im),
            lambda im: preprocess(im.resize((int(im.width*0.95), int(im.height*0.95)))),
            lambda im: preprocess(im.resize((int(im.width*1.05), int(im.height*1.05)))),
            lambda im: preprocess(im.crop((0,0,im.width-10,im.height-10)).resize((im.width,im.height))),
            lambda im: preprocess(im.crop((10,10,im.width,im.height)).resize((im.width,im.height))),
            lambda im: preprocess(im).flip(-1) if hasattr(torch.Tensor, 'flip') else preprocess(im)
        ]
    inputs = torch.stack([t(pil_img) for t in tta_transforms]).to(device)
    if dtype is not None:
        inputs = inputs.to(dtype=dtype)
    with torch.no_grad():
        feats = model.encode_image(inputs)
        if dtype is not None:
            feats = feats.to(dtype=dtype)
        feats = F.normalize(feats, dim=-1)
        avg = feats.mean(dim=0, keepdim=True)
        avg = F.normalize(avg, dim=-1)
    return avg

def topk_for_labels(model, img_tensor, label_embs, labels, topk=5):
    with torch.no_grad():
        img_emb = model.encode_image(img_tensor)
        img_emb = F.normalize(img_emb, dim=-1)
        label_embs = label_embs.to(dtype=img_emb.dtype)
        cos = (img_emb @ label_embs.T).squeeze(0)   # (L,)
        logits = cos * model.logit_scale.exp()
        probs = logits.softmax(dim=-1)
    vals = list(zip(labels, cos.detach().cpu().numpy(), probs.detach().cpu().numpy()))
    vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)
    return vals_sorted, cos, probs

def pgd_linf_attack(pipeline_model, x0_tensor, normalize_fn, text_embs, device,
                    eps=8/255.0, alpha=None, iters=20, restarts=5, rand_start=True, eot=1):
    
    # dtype & device consistent with model
    dtype = next(pipeline_model.parameters()).dtype
    device = next(pipeline_model.parameters()).device

    # x0_tensor is (1,C,H,W) in [0,1]
    x0 = x0_tensor.to(device=device, dtype=dtype)
    x0 = x0.clamp(0,1)

    # ensure text_embs on same device
    text_embs = text_embs.to(device=device)
    text_embs = F.normalize(text_embs, dim=-1)

    if alpha is None:
        alpha = max(eps / 4.0, 1.0 / 255.0)

    worst_adv = x0.clone().detach()
    worst_loss = torch.full((x0.size(0),), -1e9, device=device, dtype=dtype)

    for r in range(restarts):
        if rand_start:
            x_adv = x0 + torch.empty_like(x0).uniform_(-eps, eps).to(device=device, dtype=dtype)
            x_adv = x_adv.clamp(0, 1)
        else:
            x_adv = x0.clone().detach()

        x_adv.requires_grad_(True)
        for _ in range(iters):
            total_loss = 0.0
            for _ in range(eot):
                # Apply normalization
                x_input = normalize_fn(x_adv)
                
                img_emb = pipeline_model.encode_image(x_input)
                img_emb = F.normalize(img_emb, dim=-1)   # (B, D)
                
                # Cast text_embs to match img_emb dtype
                text_embs = text_embs.to(dtype=img_emb.dtype)
                
                sim = img_emb @ text_embs.T              # (B, L)
                # scalar loss per example: mean similarity across candidate labels, negated
                loss_per_example = -sim.mean(dim=1)      # (B,)
                loss = loss_per_example.mean()           # scalar (dtype matches)
                total_loss = total_loss + loss
            total_loss = total_loss / float(eot)

            pipeline_model.zero_grad()
            total_loss.backward()
            with torch.no_grad():
                grad = x_adv.grad
                # grad may be float32 even in fp16 model; cast step to model dtype
                step = (alpha * grad.sign()).to(device=device, dtype=dtype)
                x_adv = x_adv + step
                x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)
                x_adv = x_adv.clamp(0, 1)
            x_adv.requires_grad_(True)

        # evaluate final loss per-example and update worst-case
        with torch.no_grad():
            x_input = normalize_fn(x_adv)
            final_img_emb = pipeline_model.encode_image(x_input)
            final_img_emb = F.normalize(final_img_emb, dim=-1)
            
            text_embs = text_embs.to(dtype=final_img_emb.dtype)
            
            final_sim = final_img_emb @ text_embs.T   # (B, L)
            final_loss_per = -final_sim.mean(dim=1)   # (B,)
            final_loss_per = final_loss_per.to(dtype=dtype)
            mask = final_loss_per > worst_loss
            if mask.any():
                worst_loss[mask] = final_loss_per[mask]
                worst_adv[mask] = x_adv[mask].detach()

    return worst_adv, x0

def build_label_embs(model, prompts, idx_map, device, dtype):
    # tokenize and encode prompts, cast to model dtype and average per label index in idx_map
    text_tokens = __import__("clip").tokenize(prompts).to(device)
    with torch.no_grad():
        txt_feats = model.encode_text(text_tokens)    # (P, D)
        txt_feats = txt_feats.to(device=device, dtype=dtype)
        txt_feats = F.normalize(txt_feats, dim=-1)
    # average per label
    num_labels = max(idx_map) + 1 if len(idx_map)>0 else 0
    label_embs = []
    for i in range(num_labels):
        inds = [j for j,k in enumerate(idx_map) if k==i]
        if len(inds) == 0:
            # fallback zero vector (shouldn't happen)
            vec = torch.zeros((1, txt_feats.size(1)), device=device, dtype=dtype)
        else:
            vec = txt_feats[inds].mean(dim=0, keepdim=True)
            vec = F.normalize(vec, dim=-1)
        label_embs.append(vec)
    if len(label_embs)==0:
        return torch.empty((0, txt_feats.size(1)), device=device, dtype=dtype)
    label_embs = torch.cat(label_embs, dim=0)
    return label_embs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="input image path")
    parser.add_argument("--model", type=str, default="ViT-L/14", help="CLIP model name")
    parser.add_argument("--eps", type=float, default=8.0/255.0, help="L-inf epsilon (decimal)")
    parser.add_argument("--alpha", type=float, default=None, help="PGD step size (decimal)")
    parser.add_argument("--iters", type=int, default=20, help="PGD iterations")
    parser.add_argument("--restarts", type=int, default=5, help="random restarts")
    parser.add_argument("--out", type=str, default="adv.png", help="output adversarial image path")
    parser.add_argument("--labels", nargs="+", default=["white dog"], help="list of ground-truth labels to reduce similarity to")
    parser.add_argument("--candidates", nargs="+", default=None, help="candidate labels for ranking (if omitted, uses --labels)")
    parser.add_argument("--eot", type=int, default=1, help="EOT samples per gradient step (for randomized defenses)")
    args = parser.parse_args()

    # device & load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Loading CLIP model {args.model} on {device}...')
    model, preprocess = load_clip_model(args.model, device)
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    # load image
    pil_img = Image.open(args.img).convert("RGB")

    # Prepare preprocessing without normalization
    if isinstance(preprocess.transforms[-1], Normalize):
        preprocess_no_norm = Compose(preprocess.transforms[:-1])
    else:
        print("Warning: Could not detect Normalize in preprocess. Using standard CLIP normalization assumptions.")
        preprocess_no_norm = Compose(preprocess.transforms[:-1])

    # Prepare normalization function
    mean = torch.tensor(CLIP_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    normalize_fn = lambda x: (x - mean) / std

    # build candidates list
    if args.candidates is None:
        candidates = args.labels
    else:
        candidates = args.candidates

    # build prompt templates and mapping
    templates = [
        "a photo of a {}",
        "a photograph of a {}",
        "a picture of a {}",
        "a close-up of a {}",
        "a cropped photo of a {}",
        "a professional photo of a {}",
        "a high quality photo of a {}",
        "an image of a {}"
    ]
    prompts = []
    idx_map = []
    for i, lbl in enumerate(candidates):
        for t in templates:
            prompts.append(t.format(lbl))
            idx_map.append(i)

    # create label embeddings (dtype-safe)
    label_embs = build_label_embs(model, prompts, idx_map, device, dtype)

    # baseline scores (single-image TTA average)
    print("\nBaseline top-k on clean image (TTA averaged):")
    tta_emb = image_tta_emb(model, preprocess, pil_img, device, dtype=dtype)
    
    # Manually compute topk for tta_emb
    with torch.no_grad():
        # tta_emb is (1, D) and normalized
        # label_embs is (L, D) and normalized
        label_embs = label_embs.to(dtype=tta_emb.dtype)
        cos = (tta_emb @ label_embs.T).squeeze(0)
        logits = cos * model.logit_scale.exp()
        probs = logits.softmax(dim=-1)
        
    vals = list(zip(candidates, cos.detach().cpu().numpy(), probs.detach().cpu().numpy()))
    vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)
    
    for lbl, c, p in vals_sorted:
        print(f"{lbl:25s}\t{c:6.3f}\t{p:6.3f}")

    # run PGD attack
    print(f"Running PGD L-inf attack eps={args.eps}, iters={args.iters}, restarts={args.restarts} ...")
    
    x0_tensor = preprocess_no_norm(pil_img).unsqueeze(0)
    
    adv, clean = pgd_linf_attack(model, x0_tensor, normalize_fn, label_embs, device,
                                 eps=args.eps, alpha=args.alpha, iters=args.iters, restarts=args.restarts, rand_start=True, eot=args.eot)
    # save adv image
    out_path = args.out
    save_tensor_image(adv, out_path)
    print(f"Saved adversarial image to {out_path}")

    # compute perturbation stats
    linf = float(torch.max(torch.abs(adv - clean)).item())
    l2 = torch.norm((adv - clean).view(adv.size(0), -1), dim=1).cpu().numpy().tolist()
    print(f"Perturbation stats: L-inf={linf:.6f}, L2={l2}")

    # post-attack top-k
    print("\nPost-attack top-k on adversarial image:")
    vals_sorted_post, cos_post, probs_post = topk_for_labels(model, normalize_fn(adv), label_embs, candidates)
    for lbl, c, p in vals_sorted_post:
        print(f"{lbl:25s}\t{c:6.3f}\t{p:6.3f}")

if __name__ == '__main__':
    main()
