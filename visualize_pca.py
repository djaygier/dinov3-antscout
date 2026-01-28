
import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from dinov3.hub.backbones import dinov3_vitl16

def get_args():
    parser = argparse.ArgumentParser(description="Visualize DINOv3 PCA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", type=str, default="pca_image.png", help="Path to save the output image")
    parser.add_argument("--resolution", type=int, default=1024, help="Input resolution")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--img_size", type=int, default=1024, help="Model image size (for interpolation)")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for background removal (optional)")
    return parser.parse_args()

def load_model(checkpoint_path, img_size):
    # Initialize the model structure (ViT-Large)
    # We set img_size to the target resolution to avoid excessive interpolation during rope if possible,
    # though dinov3 handles variable sizes via ROPE.
    model = dinov3_vitl16(
        pretrained=False,
        img_size=img_size, # This hints the ROPE grid size
    )
    model.eval()
    model.cuda()

    # Load weights
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle different checkpoint keys
    if "teacher" in state_dict:
        print("Found 'teacher' key in checkpoint.")
        state_dict = state_dict["teacher"]
    elif "student" in state_dict:
        print("Found 'student' key in checkpoint.")
        state_dict = state_dict["student"]
    elif "model" in state_dict:
        print("Found 'model' key in checkpoint.")
        state_dict = state_dict["model"]
    
    # Remove prefix "backbone." or "module." if present
    clean_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("backbone.", "").replace("module.", "")
        clean_state_dict[k] = v
        
    msg = model.load_state_dict(clean_state_dict, strict=False)
    print(f"Model loaded with msg: {msg}")
    return model

def main():
    args = get_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Image
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found at {args.image}")
        
    original_image = Image.open(args.image).convert("RGB")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Load Model
    model = load_model(args.checkpoint, args.resolution)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        # returns dict with keys like 'x_norm_patchtokens'
        # forward_features handles formatting
        features_dict = model.forward_features(input_tensor)
        patch_tokens = features_dict["x_norm_patchtokens"] # [B, N, D]
        
    # Process PCA
    features = patch_tokens[0].cpu().numpy() # [N, D]
    
    # PCA
    print("Computing PCA...")
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)
    
    # Min-Max Normalize to [0, 1] for RGB
    pca_features = (pca_features - pca_features.min(0)) / (pca_features.max(0) - pca_features.min(0))
    
    # Reshape to image
    # N = H_grid * W_grid
    grid_size = int(np.sqrt(features.shape[0]))
    pca_image = pca_features.reshape(grid_size, grid_size, 3)
    
    # Upscale to output resolution
    pca_tensor = torch.from_numpy(pca_image).permute(2, 0, 1).unsqueeze(0).float() # [1, 3, grid, grid]
    pca_output = F.interpolate(pca_tensor, size=(args.resolution, args.resolution), mode='nearest')
    pca_output = pca_output.squeeze(0).permute(1, 2, 0).numpy()
    
    # Plotting
    print(f"Saving output to {args.output}...")
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original Image (resized)
    ax[0].imshow(original_image.resize((args.resolution, args.resolution)))
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    # PCA
    ax[1].imshow(pca_output)
    ax[1].set_title(f"DINOv3 PCA (Patch Size {args.patch_size})")
    ax[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(args.output)
    plt.close()
    print("Done!")

if __name__ == "__main__":
    main()
