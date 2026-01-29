#!/usr/bin/env python3
"""
Wrap a DINOv3 pretrained backbone checkpoint to be compatible with the training code.

The training code expects checkpoints with a 'teacher' key containing the model weights.
Official Meta pretrained checkpoints have a flat structure without this key.

Usage:
    python wrap_pretrained_checkpoint.py \
        --input /workspace/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --output /workspace/dinov3_vitl16_wrapped.pth
"""

import argparse
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Wrap pretrained checkpoint for training")
    parser.add_argument("--input", type=str, required=True, help="Path to the original pretrained checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path for the wrapped checkpoint")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.input}")
    state_dict = torch.load(args.input, map_location="cpu", weights_only=False)

    # Check if already wrapped
    if "teacher" in state_dict:
        print("Checkpoint already has 'teacher' key. Checking structure...")
        print(f"Keys in checkpoint: {list(state_dict.keys())}")
        print("No wrapping needed.")
        return

    # Print original structure
    print(f"Original checkpoint has {len(state_dict)} keys")
    sample_keys = list(state_dict.keys())[:5]
    print(f"Sample keys: {sample_keys}")

    # Add prefixes to match expected structure
    # The training code expects: teacher.backbone.X, teacher.dino_head.X, etc.
    # But if loading just the backbone, we need to add "backbone." prefix
    
    # Check if keys already have backbone prefix
    has_backbone_prefix = any(k.startswith("backbone.") for k in state_dict.keys())
    
    if has_backbone_prefix:
        print("Keys already have 'backbone.' prefix")
        wrapped = {"teacher": state_dict}
    else:
        # Add backbone prefix to all keys
        print("Adding 'backbone.' prefix to all keys")
        prefixed = {f"backbone.{k}": v for k, v in state_dict.items()}
        wrapped = {"teacher": prefixed}

    # Save wrapped checkpoint
    print(f"Saving wrapped checkpoint to {args.output}")
    torch.save(wrapped, args.output)

    # Verify
    verify = torch.load(args.output, map_location="cpu", weights_only=False)
    print(f"Verification - Keys in wrapped checkpoint: {list(verify.keys())}")
    print(f"Verification - Keys inside 'teacher': {list(verify['teacher'].keys())[:5]}...")
    print("Done!")


if __name__ == "__main__":
    main()
