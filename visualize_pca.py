
import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from dinov3.models import build_model_for_eval
from dinov3.configs import setup_config, DinoV3SetupArgs
import dinov3.distributed as distributed

def get_args():
    parser = argparse.ArgumentParser(description="Visualize DINOv3 PCA")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", type=str, default="pca_image.png", help="Path to save the output image")
    parser.add_argument("--resolution", type=int, default=1024, help="Input resolution")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--img_size", type=int, default=1024, help="Model image size (for interpolation)")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for background removal (optional)")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the config file used for training")
    parser.add_argument("--student", action="store_true", help="Visualize student instead of teacher")
    return parser.parse_args()

def load_model(checkpoint_path, config_file, use_student=False):
    # Initialize distributed (required for config parsing)
    distributed.enable()
    
    # 1. Setup Config
    print(f"Loading config from: {config_file}")
    setup_args = DinoV3SetupArgs(config_file=config_file)
    cfg = setup_config(setup_args, strict_cfg=False)
    
    # 2. Build model for eval
    print(f"Building model from config: {config_file}")
    model = build_model_for_eval(cfg, checkpoint_path)
    
    # If we want the student, we have a problem because build_model_for_eval loads "teacher" key.
    # But for Stage 3 consolidation, we saved it under "teacher" anyway.
    
    model.eval()
    model.cuda()
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
    model = load_model(args.checkpoint, args.config_file, args.student)
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        # returns dict with keys like 'x_norm_patchtokens'
        # forward_features handles formatting
        features_dict = model.forward_features(input_tensor)
        patch_tokens = features_dict["x_norm_patchtokens"] # [B, N, D]
        
    # Process PCA
    features = patch_tokens[0].cpu().numpy() # [N, D]
    
    # Compute PCA
    print("Computing PCA on spatial tokens...")
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)
    
    # 1. Percentile Clipping (Removes outliers that wash out colors)
    v_min = np.percentile(pca_features, 1, axis=0)
    v_max = np.percentile(pca_features, 99, axis=0)
    pca_features = np.clip(pca_features, v_min, v_max)
    
    # 2. Min-Max Normalize to [0, 1] for RGB
    pca_features = (pca_features - pca_features.min(0)) / (pca_features.max(0) - pca_features.min(0))
    
    # 3. Reshape to image
    h_grid = args.resolution // args.patch_size
    w_grid = args.resolution // args.patch_size
    
    if h_grid * w_grid != features.shape[0]:
        print(f"Warning: Token count mismatch! Grid: {h_grid*w_grid}, Features: {features.shape[0]}.")
        grid_size = int(np.sqrt(features.shape[0]))
        h_grid, w_grid = grid_size, grid_size
        
    pca_image = pca_features[:h_grid*w_grid].reshape(h_grid, w_grid, 3)
    
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
