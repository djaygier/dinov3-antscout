import os
import shutil
import torch
from PIL import Image
from dinov3.data import make_dataset

def setup_dummy_dataset(root):
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, "class1"))
    os.makedirs(os.path.join(root, "class2"))
    
    # Create a dummy image
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save(os.path.join(root, "class1", "img1.jpg"))
    img.save(os.path.join(root, "class2", "img2.jpg"))

def test_image_folder_loading():
    dummy_root = "dummy_dataset"
    setup_dummy_dataset(dummy_root)
    
    try:
        dataset_str = f"ImageFolder:root={dummy_root}"
        dataset = make_dataset(dataset_str=dataset_str)
        
        print(f"Successfully loaded dataset: {dataset}")
        print(f"Dataset length: {len(dataset)}")
        assert len(dataset) == 2, f"Expected 2 samples, got {len(dataset)}"
        
        img, target = dataset[0]
        print(f"Sample 0 type: {type(img)}, target: {target}")
        
    finally:
        if os.path.exists(dummy_root):
            shutil.rmtree(dummy_root)

if __name__ == "__main__":
    test_image_folder_loading()
