
import torch
import argparse
from dinov3.models import build_model_for_eval
from dinov3.configs import setup_config, DinoV3SetupArgs
import dinov3.distributed as distributed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    distributed.enable()

    # 1. Load config
    setup_args = DinoV3SetupArgs(config_file=args.config)
    cfg = setup_config(setup_args, strict_cfg=False)

    # 2. Initialize TWO models
    # Model A: Random init
    print("Building random model...")
    model_random = build_model_for_eval(cfg, pretrained_weights=None)
    
    # Model B: Loaded from checkpoint
    print(f"Building model from {args.checkpoint}...")
    model_loaded = build_model_for_eval(cfg, pretrained_weights=args.checkpoint)

    # 3. Compare weights of a specific layer (e.g., first block attention)
    # We grab the keys from the state dict to be safe
    key = "backbone.blocks.0.attn.qkv.weight"
    
    rand_weight = model_random.state_dict()[key]
    load_weight = model_loaded.state_dict()[key]

    print(f"\nComparing layer: {key}")
    print(f"Random Weight Mean: {rand_weight.mean().item():.6f}, Std: {rand_weight.std().item():.6f}")
    print(f"Loaded Weight Mean: {load_weight.mean().item():.6f}, Std: {load_weight.std().item():.6f}")

    if torch.equal(rand_weight, load_weight):
        print("\n[CRITICAL] The loaded weights are IDENTICAL to random initialization!")
        print("Conclusion: The checkpoint loading FAILED silently.")
    else:
        print("\n[SUCCESS] The weights are different from random initialization.")
        print("Conclusion: The checkpoint contains learned weights (or at least different ones).")

if __name__ == "__main__":
    main()
