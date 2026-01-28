
import torch
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=str, required=True, help="Path to official eval checkpoint (e.g., .../eval/training_3749/teacher_checkpoint.pth)")
    parser.add_argument("--consolidated", type=str, required=True, help="Path to consolidated checkpoint")
    args = parser.parse_args()

    print("Loading official checkpoint...")
    official = torch.load(args.official, map_location="cpu")
    print("Loading consolidated checkpoint...")
    consolidated = torch.load(args.consolidated, map_location="cpu")

    # 1. Check top-level keys
    print("\n=== TOP-LEVEL KEYS ===")
    print(f"Official: {list(official.keys())}")
    print(f"Consolidated: {list(consolidated.keys())}")

    # 2. Check "teacher" state dict keys
    official_sd = official.get("teacher", official)
    consolidated_sd = consolidated.get("teacher", consolidated)
    
    official_keys = set(official_sd.keys())
    consolidated_keys = set(consolidated_sd.keys())
    
    print(f"\n=== STATE DICT KEY COUNT ===")
    print(f"Official: {len(official_keys)} keys")
    print(f"Consolidated: {len(consolidated_keys)} keys")

    # 3. Key differences
    only_official = official_keys - consolidated_keys
    only_consolidated = consolidated_keys - official_keys
    common = official_keys.intersection(consolidated_keys)

    print(f"\n=== KEY DIFFERENCES ===")
    print(f"Keys ONLY in Official ({len(only_official)}): {list(only_official)[:10]}")
    print(f"Keys ONLY in Consolidated ({len(only_consolidated)}): {list(only_consolidated)[:10]}")
    print(f"Common keys: {len(common)}")

    # 4. Sample key prefix analysis
    print("\n=== KEY PREFIX ANALYSIS ===")
    official_prefixes = set(k.split(".")[0] for k in official_keys)
    consolidated_prefixes = set(k.split(".")[0] for k in consolidated_keys)
    print(f"Official prefixes: {official_prefixes}")
    print(f"Consolidated prefixes: {consolidated_prefixes}")

    # 5. Compare actual values for a common key
    if common:
        sample_key = next(iter(common))
        official_val = official_sd[sample_key]
        consolidated_val = consolidated_sd[sample_key]
        
        print(f"\n=== VALUE COMPARISON for '{sample_key}' ===")
        print(f"Official shape: {official_val.shape}, dtype: {official_val.dtype}")
        print(f"Consolidated shape: {consolidated_val.shape}, dtype: {consolidated_val.dtype}")
        print(f"Official mean: {official_val.float().mean().item():.6f}, std: {official_val.float().std().item():.6f}")
        print(f"Consolidated mean: {consolidated_val.float().mean().item():.6f}, std: {consolidated_val.float().std().item():.6f}")
        
        if torch.allclose(official_val.float(), consolidated_val.float(), atol=1e-5):
            print("VALUES MATCH!")
        else:
            diff = (official_val.float() - consolidated_val.float()).abs()
            print(f"VALUES DO NOT MATCH! Max diff: {diff.max().item():.6f}, Mean diff: {diff.mean().item():.6f}")
    
    print("\n=== DIAGNOSIS ===")
    if only_official and not only_consolidated:
        print("ISSUE: Official has extra keys (likely 'backbone.' prefix). Consolidation script may be stripping the prefix incorrectly.")
    elif only_consolidated and not only_official:
        print("ISSUE: Consolidated has extra keys. Consolidation script may be adding unexpected keys.")
    elif not common:
        print("CRITICAL: No common keys at all! The checkpoints are completely different.")
    else:
        print("Keys seem consistent. Checking values...")
        
        # Deep comparison of all common keys
        mismatches = 0
        for k in list(common)[:20]:  # Check first 20
            if not torch.allclose(official_sd[k].float(), consolidated_sd[k].float(), atol=1e-5):
                mismatches += 1
        if mismatches > 0:
            print(f"Found {mismatches} value mismatches in first 20 keys. The weights are different!")
        else:
            print("First 20 keys have matching values.")

if __name__ == "__main__":
    main()
