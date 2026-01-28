
import torch
import argparse
from pathlib import Path
from dinov3.checkpointer import load_checkpoint
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.configs import setup_config

def get_args():
    parser = argparse.ArgumentParser(description="Consolidate Sharded Checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--src", type=str, required=True, help="Path to sharded checkpoint directory")
    parser.add_argument("--dst", type=str, required=True, help="Output path for teacher_checkpoint.pth")
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Setup config (we need this to build the model structure)
    # Using a dummy argv for setup_config
    class DummyArgs:
        def __init__(self, config_file):
            self.config_file = config_file
            self.output_dir = "tmp_consolidate"
            self.opts = []
            
    cfg = setup_config(DummyArgs(args.config), strict_cfg=False)
    
    # 2. Build model structure (on CPU/Meta to avoid VRAM)
    print("Building model structure...")
    model = SSLMetaArch(cfg)
    
    # 3. Load sharded weights
    print(f"Loading sharded weights from {args.src}...")
    # load_checkpoint handles FSDP sharded vs consolidated
    load_checkpoint(args.src, model)
    
    # 4. Extract Teacher EMA state dict
    # DINOv3 evaluation usually uses model_ema (the teacher)
    teacher_state_dict = model.model_ema.state_dict()
    
    # Optional: consolidate DTensors if they exist (though load_checkpoint usually returns local tensors)
    # But just in case:
    consolidated_dict = {}
    for k, v in teacher_state_dict.items():
        if hasattr(v, "full_tensor"):
            consolidated_dict[k] = v.full_tensor()
        else:
            consolidated_dict[k] = v

    # 5. Save consolidated file
    print(f"Saving consolidated teacher to {args.dst}...")
    torch.save({"teacher": consolidated_dict}, args.dst)
    print("Done!")

if __name__ == "__main__":
    main()
