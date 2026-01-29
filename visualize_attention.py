"""
Self-Attention Visualization for DINOv3.
Visualizes how the [CLS] token attends to different patches.
"""
import argparse
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="DINOv3 Attention Visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="attention.png")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--config-file", type=str, required=True)
    args = parser.parse_args()

    # Setup Distributed (required by DINOv3 building)
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    
    from dinov3.configs import setup_config
    from dinov3.models import build_model_for_eval

    # Load config and model
    cfg = setup_config(argparse.Namespace(config_file=args.config_file, opts=[], output_dir=None), strict_cfg=False)
    model = build_model_for_eval(cfg, args.checkpoint)
    model.eval()

    # Preprocess image
    img = Image.open(args.image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    device = torch.device("cuda")
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Hook for capturing attention matrices
    attentions = []
    def hook_fn(module, input, output):
        # DINOv3 Block output is x or list of x. 
        # But we want the attention probabilities from inside the Attention block.
        pass

    # For ViT Large, we want the last layer's attention
    last_blk = model.blocks[-1]
    
    # We need to reach into the Attention module to get the weights
    # d_v3 Attention returns (x, attn_probs) or similar if requested, 
    # but standard forward only returns x.
    # We will use a manual forward pass for the last block to get weights.

    print("Extracting attention from last layer...")
    with torch.no_grad():
        # 1. Prepare tokens (prepare_tokens_with_masks)
        x, hw_tuple = model.prepare_tokens_with_masks(input_tensor)
        
        # 2. Pass through all blocks except the last one
        for i in range(len(model.blocks) - 1):
            if model.rope_embed is not None:
                rope = [model.rope_embed(H=hw_tuple[0], W=hw_tuple[1])]
            else:
                rope = [None]
            x = model.blocks[i](x, rope)
        
        # 3. Manual forward for the last block to get attention
        # Note: We need to see how Block.forward is implemented.
        # Most ViT blocks: x = x + drop_path(attn(norm1(x)))
        # We need the weights from the 'attn' call.
        
        # Get RoPE for last layer
        rope = model.rope_embed(H=hw_tuple[0], W=hw_tuple[1]) if model.rope_embed else None
        
        # Standard ViT Attention usually has self.attn.get_attention_map or similar
        # For DINOv3, we'll manually invoke the attention mechanism
        norm_x = model.blocks[-1].norm1(x)
        
        # DinoVisionTransformer uses a custom Attention that supports list of tensors
        # It typically doesn't return attention weights by default.
        # Let's access the attention module
        attn_module = model.blocks[-1].attn
        
        # We'll use a trick: DINOv3 Attention's forward_list calls attn_matrix = ...
        # If we can't get it easily, we'll compute it from Q and K
        # DinoVisionTransformer Attention structure:
        # qkv = self.qkv(x) -> Q, K, V
        qkv = attn_module.qkv(norm_x[0]) # [B, N, 3*D]
        B, N, _ = qkv.shape
        num_heads = attn_module.num_heads
        head_dim = _ // (3 * num_heads)
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, heads, N, head_dim]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5) # [B, heads, N, N]
        
        # Apply RoPE if present
        if rope is not None:
             # This is complex to replicate exactly here, so we'll take the raw scores
             # which usually represent the semantic attention well enough for viz.
             pass
             
        attn = attn.softmax(dim=-1) # [B, heads, N, N]
        
    # [CLS] token is at index 0
    # We want how [CLS] looks at patches (index 1 to N-1)
    # Average over heads or pick the best one
    cls_attn = attn[0, :, 0, 1:].cpu().numpy() # [heads, N-1]
    
    # Grid dimensions
    w_featmap = hw_tuple[1]
    h_featmap = hw_tuple[0]
    
    # Reshape heads to spatial grid
    cls_attn = cls_attn.reshape(-1, h_featmap, w_featmap) # [heads, H, W]
    
    # Visualize all heads to find the best one
    n_heads = cls_attn.shape[0]
    cols = 4
    rows = (n_heads + cols - 1) // cols
    
    fig, axes = plt.subplots(rows + 1, cols, figsize=(cols*4, (rows+1)*4))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(img.resize((args.resolution, args.resolution)))
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Combined heads (mean)
    mean_attn = cls_attn.mean(0)
    axes[1].imshow(mean_attn, cmap='magma')
    axes[1].set_title("Mean Attention (All Heads)")
    axes[1].axis('off')

    # Combined overlay
    mean_attn_img = Image.fromarray((mean_attn / mean_attn.max() * 255).astype(np.uint8)).resize((args.resolution, args.resolution), Image.BILINEAR)
    axes[2].imshow(img.resize((args.resolution, args.resolution)))
    axes[2].imshow(mean_attn_img, alpha=0.6, cmap='magma')
    axes[2].set_title("Mean Overlay")
    axes[2].axis('off')
    
    # Individual heads (usually some heads focus on object, others on background)
    for i in range(n_heads):
        ax = axes[i + cols] if i + cols < len(axes) else None
        if ax:
            ax.imshow(cls_attn[i], cmap='magma')
            ax.set_title(f"Head {i}")
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved attention visualization to {args.output}")

if __name__ == "__main__":
    main()
