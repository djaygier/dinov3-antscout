import torch
import sys

def check_ckpt(path):
    print(f"Checking checkpoint: {path}")
    try:
        ckpt = torch.load(path, map_location="cpu")
        if "teacher" in ckpt:
            state_dict = ckpt["teacher"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        
        keys = state_dict.keys()
        storage_tokens = [k for k in keys if "storage_tokens" in k]
        bias_masks = [k for k in keys if "bias_mask" in k]
        cls_norm = [k for k in keys if "cls_norm" in k]
        
        print(f"Found {len(keys)} keys total.")
        print(f"Storage tokens keys: {storage_tokens}")
        print(f"Bias mask keys (first 3): {bias_masks[:3]}")
        print(f"CLS norm keys: {cls_norm}")
        
        # Check dim of a block
        block0_qkv = [k for k in keys if "blocks.0.attn.qkv.weight" in k]
        if block0_qkv:
            print(f"Block 0 QKV shape: {state_dict[block0_qkv[0]].shape}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_ckpt(sys.argv[1])
    else:
        print("Please provide a path to a checkpoint.")
