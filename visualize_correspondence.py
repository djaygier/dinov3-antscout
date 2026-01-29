"""
Correspondence visualization for DINOv3 features.
Click a point to see cosine similarity heatmap to all other patches.
"""
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="DINOv3 Correspondence Visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="correspondence.png")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--query_x", type=int, default=None, help="Query patch x (0 to grid_size-1)")
    parser.add_argument("--query_y", type=int, default=None, help="Query patch y (0 to grid_size-1)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode - click to query")
    args = parser.parse_args()

    # Setup
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    
    from dinov3.configs import setup_config
    from dinov3.models import build_model_for_eval
    from omegaconf import OmegaConf

    # Load config and model
    print(f"Loading config from: {args.config_file}")
    cfg = setup_config(argparse.Namespace(config_file=args.config_file, opts=[], output_dir=None), strict_cfg=False)
    print(f"Building model...")
    model = build_model_for_eval(cfg, args.checkpoint)
    model.eval()

    # Load and preprocess image
    img = Image.open(args.image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img).unsqueeze(0).cuda()

    # Get features
    print("Extracting features...")
    with torch.no_grad():
        features_dict = model.forward_features(input_tensor)
        patch_tokens = features_dict["x_norm_patchtokens"]  # [1, N, D]
    
    features = patch_tokens[0]  # [N, D]
    features = F.normalize(features, dim=-1)  # L2 normalize for cosine similarity
    
    grid_size = args.resolution // args.patch_size
    print(f"Grid size: {grid_size}x{grid_size} ({features.shape[0]} patches)")
    
    # Reshape for visualization
    features_grid = features.reshape(grid_size, grid_size, -1)
    
    def compute_similarity_map(query_y, query_x):
        """Compute cosine similarity between query patch and all other patches."""
        query_feat = features_grid[query_y, query_x]  # [D]
        # Cosine similarity (features are already normalized)
        sim = torch.einsum("d,hwd->hw", query_feat, features_grid).cpu().numpy()
        
        # Auto-scale for visibility: 
        # Typically features might have low absolute variance.
        # We normalize the heatmap to [0, 1] based on its own distribution.
        sim_min = sim.min()
        sim_max = sim.max()
        sim_norm = (sim - sim_min) / (sim_max - sim_min + 1e-8)
        
        # 2nd option: Heatmap of standard deviations or just raw sim with percentile clipping
        v_min = np.percentile(sim, 2)
        v_max = np.percentile(sim, 98)
        sim_clipped = np.clip(sim, v_min, v_max)
        sim_clipped = (sim_clipped - v_min) / (v_max - v_min + 1e-8)
        
        print(f"Similarity stats: Min={sim_min:.4f}, Max={sim_max:.4f}, Mean={sim.mean():.4f}")
        return sim_norm, sim_clipped
    
    def visualize(query_y, query_x, save_path=None):
        """Create visualization with query point and similarity heatmap."""
        sim_norm, sim_clipped = compute_similarity_map(query_y, query_x)
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(img.resize((args.resolution, args.resolution)))
        # Mark query point (in image coordinates)
        img_x = query_x * args.patch_size + args.patch_size // 2
        img_y = query_y * args.patch_size + args.patch_size // 2
        axes[0].scatter([img_x], [img_y], c='red', s=100, marker='x', linewidths=3)
        axes[0].set_title(f"Query Point ({query_x}, {query_y})")
        axes[0].axis('off')
        
        # Similarity heatmap (Full Range)
        im1 = axes[1].imshow(sim_norm, cmap='magma')
        axes[1].scatter([query_x], [query_y], c='cyan', s=50, marker='x', linewidths=2)
        axes[1].set_title("Similarity (Min-Max Scaled)")
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        
        # Similarity heatmap (Robust Percentile Scaled)
        im2 = axes[2].imshow(sim_clipped, cmap='magma')
        axes[2].scatter([query_x], [query_y], c='cyan', s=50, marker='x', linewidths=2)
        axes[2].set_title("Similarity (Robust 2-98%)")
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        # Overlay on image (using robust map)
        sim_upscaled = np.array(Image.fromarray((sim_clipped * 255).astype(np.uint8)).resize(
            (args.resolution, args.resolution), Image.BILINEAR)) / 255.0
        img_np = np.array(img.resize((args.resolution, args.resolution))) / 255.0
        # Use a more visible overlay
        overlay = img_np * 0.4 + plt.cm.magma(sim_upscaled)[:,:,:3] * 0.6
        axes[3].imshow(overlay)
        axes[3].scatter([img_x], [img_y], c='cyan', s=100, marker='x', linewidths=3)
        axes[3].set_title("Overlay (Robust)")
        axes[3].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        return fig
    
    if args.interactive:
        # Interactive mode
        print("\nInteractive mode: Click on the image to query feature similarity.")
        print("Close the window to exit.")
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img.resize((args.resolution, args.resolution)))
        ax.set_title("Click anywhere to see feature correspondence")
        
        def onclick(event):
            if event.inaxes != ax:
                return
            img_x, img_y = int(event.xdata), int(event.ydata)
            query_x = img_x // args.patch_size
            query_y = img_y // args.patch_size
            query_x = min(max(0, query_x), grid_size - 1)
            query_y = min(max(0, query_y), grid_size - 1)
            print(f"Query patch: ({query_x}, {query_y})")
            visualize(query_y, query_x)
            plt.show()
        
        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
    else:
        # Static mode
        if args.query_x is None or args.query_y is None:
            # Default: center of image
            query_x = grid_size // 2
            query_y = grid_size // 2
        else:
            query_x = args.query_x
            query_y = args.query_y
        
        visualize(query_y, query_x, args.output)
        plt.show()

if __name__ == "__main__":
    main()
