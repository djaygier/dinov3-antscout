"""
Debug script to verify Gram teacher checkpoint loading.
Run this during training initialization or standalone.
"""
import torch
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    
    print(f"\n=== TOP-LEVEL KEYS ===")
    print(list(ckpt.keys()))
    
    if "teacher" in ckpt:
        teacher_sd = ckpt["teacher"]
    else:
        teacher_sd = ckpt
    
    print(f"\n=== TEACHER STATE DICT ({len(teacher_sd)} keys) ===")
    print("First 10 keys:")
    for i, (k, v) in enumerate(teacher_sd.items()):
        if i >= 10:
            break
        print(f"  {k}: {v.shape}")
    
    # Check for expected structure
    backbone_keys = [k for k in teacher_sd.keys() if k.startswith("backbone.")]
    non_backbone_keys = [k for k in teacher_sd.keys() if not k.startswith("backbone.")]
    
    print(f"\n=== KEY STRUCTURE ===")
    print(f"Keys with 'backbone.' prefix: {len(backbone_keys)}")
    print(f"Keys WITHOUT 'backbone.' prefix: {len(non_backbone_keys)}")
    
    if non_backbone_keys:
        print(f"\nNon-backbone keys (first 5): {non_backbone_keys[:5]}")
    
    # Check for storage_tokens key (critical for architecture matching)
    storage_token_keys = [k for k in teacher_sd.keys() if "storage_tokens" in k]
    print(f"\n=== STORAGE TOKENS CHECK ===")
    if storage_token_keys:
        print(f"Found storage_tokens keys: {storage_token_keys}")
        for k in storage_token_keys:
            print(f"  {k}: {teacher_sd[k].shape}")
    else:
        print("NO storage_tokens keys found - model was trained with n_storage_tokens=0")
    
    # Sample weight statistics to verify it's not garbage
    sample_key = next(iter(teacher_sd.keys()))
    sample_val = teacher_sd[sample_key]
    print(f"\n=== SAMPLE WEIGHT STATISTICS ===")
    print(f"Key: {sample_key}")
    print(f"Shape: {sample_val.shape}")
    print(f"Mean: {sample_val.float().mean().item():.6f}")
    print(f"Std: {sample_val.float().std().item():.6f}")
    print(f"Min: {sample_val.float().min().item():.6f}")
    print(f"Max: {sample_val.float().max().item():.6f}")

if __name__ == "__main__":
    main()
