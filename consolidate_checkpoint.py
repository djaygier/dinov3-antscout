
import torch
import argparse
from pathlib import Path
from dinov3.checkpointer import load_checkpoint
from dinov3.train.ssl_meta_arch import SSLMetaArch
from dinov3.configs import setup_config
import dinov3.distributed as distributed

def get_args():
    parser = argparse.ArgumentParser(description="Consolidate Sharded Checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--src", type=str, required=True, help="Path to sharded checkpoint directory")
    parser.add_argument("--dst", type=str, required=True, help="Output path for teacher_checkpoint.pth")
    return parser.parse_args()

def main():
    args = get_args()
    
    # 0. Initialize distributed mode (required for setup_config)
    distributed.enable()
    
    # 1. Setup config
    output_dir = Path("tmp_consolidate")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    class DummyArgs:
        def __init__(self, config_file, output_dir):
            self.config_file = config_file
            self.output_dir = str(output_dir)
            self.opts = []
            
    cfg = setup_config(DummyArgs(args.config, output_dir), strict_cfg=False)
    
    # 2. Build model structure
    print("Building model structure (initially on meta device)...")
    model = SSLMetaArch(cfg)
    
    # Materialize meta tensors on CPU before loading
    print("Materializing model on CPU...")
    model.to_empty(device="cpu")
    
    # 3. Load sharded weights
    print(f"Loading sharded weights from {args.src} into CPU model...")
    load_checkpoint(args.src, model=model)
    
    # 4. Extract Teacher EMA state dict
    print("Extracting Teacher (EMA) backbone...")
    teacher_state_dict = model.teacher.backbone.state_dict()
    dst_path = Path(args.dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"teacher": teacher_state_dict}, dst_path)

    # 5. Extract Student state dict
    print("Extracting Student backbone...")
    student_state_dict = model.student.backbone.state_dict()
    student_dst = dst_path.parent / dst_path.name.replace("teacher", "student")
    if student_dst == dst_path:
         student_dst = dst_path.parent / (dst_path.stem + "_student" + dst_path.suffix)
    torch.save({"teacher": student_state_dict}, student_dst)

    print(f"Saved Teacher to: {dst_path}")
    print(f"Saved Student to: {student_dst}")
    print("Done!")

if __name__ == "__main__":
    main()
