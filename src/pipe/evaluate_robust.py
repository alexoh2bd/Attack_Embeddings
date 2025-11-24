from experiment import deliverthegoods
import os

def main():
    # Define datasets and perturbations
    # Using a subset for faster evaluation
    datasets = ["VisualNews_i2t"] 
    perturbations = ["ctrl", "fgsm", "pgd"]
    
    # Evaluate original model
    print("Evaluating original model...")
    model_name_original = "openai/clip-vit-base-patch32"
    deliverthegoods(datasets, perturbations, model_name_original)
    
    # Evaluate robust model
    robust_model_path = "robust_clip_model"
    if os.path.exists(robust_model_path):
        print("\nEvaluating robust model...")
        deliverthegoods(datasets, perturbations, robust_model_path)
    else:
        print(f"\nRobust model not found at {robust_model_path}. Please run train.py first.")

if __name__ == "__main__":
    main()
