import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image
import numpy as np

def fgsm_attack_clip(image, pipeline, epsilon=0.03, target_text=None):
    """
    FGSM attack on CLIP to reduce image-text similarity.
    
    Args:
        image: PIL Image
        pipeline: MultimodalRetrievalPipeline with CLIP model
        epsilon: Perturbation magnitude (default 0.03)
        target_text: Optional specific text to attack against. If None, uses generic text.
    
    Returns:
        Perturbed PIL Image
    """
    # Use a generic text if none provided
    if target_text is None:
        target_text = "A photo of an object"
    
    # Convert image to tensor with CLIP preprocessing
    img_rgb = image.convert("RGB")
    transform = TF.Compose([
        TF.Resize((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                     std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    img_tensor = transform(img_rgb).unsqueeze(0).to(pipeline.model.device)
    img_tensor.requires_grad = True
    
    # Encode text
    text_inputs = pipeline.processor(text=[target_text], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(pipeline.model.device) for k, v in text_inputs.items()}
    
    with torch.enable_grad():
        # Get embeddings
        text_embeds = pipeline.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        image_embeds = pipeline.model.get_image_features(pixel_values=img_tensor)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # Loss: maximize distance (minimize similarity)
        similarity = F.cosine_similarity(image_embeds, text_embeds)
        loss = similarity.mean()
        
        # Backprop
        pipeline.model.zero_grad()
        loss.backward()
        
        # FGSM: perturb in direction that INCREASES loss (reduces similarity)
        sign_grad = img_tensor.grad.sign()
        perturbed_tensor = img_tensor + epsilon * sign_grad
        
        # Denormalize back to [0, 1] range
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(pipeline.model.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(pipeline.model.device)
        
        perturbed_tensor = perturbed_tensor * std + mean
        perturbed_tensor = torch.clamp(perturbed_tensor, 0, 1)
    
    # Convert back to PIL Image
    perturbed_np = perturbed_tensor.squeeze(0).detach().cpu().numpy()
    perturbed_np = (perturbed_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    perturbed_image = Image.fromarray(perturbed_np)
    
    # Resize back to original size if needed
    if perturbed_image.size != image.size:
        perturbed_image = perturbed_image.resize(image.size, Image.LANCZOS)
    
    return perturbed_image

def pgd_attack_clip(image, pipeline, epsilon=0.03, alpha=0.01, steps=10, target_text=None):
    """
    PGD attack on CLIP to reduce image-text similarity.
    
    Args:
        image: PIL Image
        pipeline: MultimodalRetrievalPipeline with CLIP model
        epsilon: Maximum perturbation magnitude
        alpha: Step size
        steps: Number of steps
        target_text: Optional specific text to attack against.
    
    Returns:
        Perturbed PIL Image
    """
    if target_text is None:
        target_text = "A photo of an object"
        
    img_rgb = image.convert("RGB")
    transform = TF.Compose([
        TF.Resize((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                     std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    # Initial image tensor
    img_tensor = transform(img_rgb).unsqueeze(0).to(pipeline.model.device)
    original_img_tensor = img_tensor.clone().detach()
    
    # Random initialization within epsilon ball
    delta = torch.zeros_like(img_tensor).uniform_(-epsilon, epsilon)
    delta = torch.clamp(delta, -epsilon, epsilon)
    
    # Encode text
    text_inputs = pipeline.processor(text=[target_text], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(pipeline.model.device) for k, v in text_inputs.items()}
    
    with torch.no_grad():
        text_embeds = pipeline.model.get_text_features(**text_inputs)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    for _ in range(steps):
        delta.requires_grad = True
        
        # Apply perturbation
        adv_img_tensor = original_img_tensor + delta
        
        # Forward pass
        image_embeds = pipeline.model.get_image_features(pixel_values=adv_img_tensor)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # Loss: maximize distance (minimize similarity)
        similarity = F.cosine_similarity(image_embeds, text_embeds)
        loss = similarity.mean()
        
        # Backward pass
        pipeline.model.zero_grad()
        loss.backward()
        
        # Update delta
        grad = delta.grad.detach()
        delta.data = delta.data + alpha * grad.sign()
        
        # Project delta
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.grad.zero_()
        
    # Apply final perturbation
    adv_img_tensor = original_img_tensor + delta.detach()
    
    # Denormalize
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(pipeline.model.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(pipeline.model.device)
    
    adv_img_tensor = adv_img_tensor * std + mean
    adv_img_tensor = torch.clamp(adv_img_tensor, 0, 1)
    
    # Convert to PIL
    perturbed_np = adv_img_tensor.squeeze(0).cpu().numpy()
    perturbed_np = (perturbed_np.transpose(1, 2, 0) * 255).astype(np.uint8)
    perturbed_image = Image.fromarray(perturbed_np)
    
    if perturbed_image.size != image.size:
        perturbed_image = perturbed_image.resize(image.size, Image.LANCZOS)
        
    return perturbed_image
