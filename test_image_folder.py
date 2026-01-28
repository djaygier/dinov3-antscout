import os
import shutil
import torch
from PIL import Image
from dinov3.data import make_dataset

def setup_dummy_nested_dataset(root):
    if os.path.exists(root):
        shutil.rmtree(root)
    # Class subdirectories
    os.makedirs(os.path.join(root, "class1", "sub1"))
    os.makedirs(os.path.join(root, "class2", "sub2", "sub3"))
    
    # Create dummy images in nested folders
    img = Image.new('RGB', (100, 100), color = 'red')
    img.save(os.path.join(root, "class1", "sub1", "img1.jpg"))
    img.save(os.path.join(root, "class2", "sub2", "sub3", "img2.jpg"))

def test_image_folder_loading_recursive():
    dummy_root = "dummy_dataset_nested"
    setup_dummy_nested_dataset(dummy_root)
    
    try:
        dataset_str = f"ImageFolder:root={dummy_root}"
        dataset = make_dataset(dataset_str=dataset_str)
        
        print(f"Successfully loaded dataset: {dataset}")
        print(f"Dataset length: {len(dataset)}")
        assert len(dataset) == 2, f"Expected 2 samples, got {len(dataset)}"
        
        # Verify samples can be accessed
        img1, target1 = dataset[0]
        img2, target2 = dataset[1]
        print(f"Sample 0 target: {target1}")
        print(f"Sample 1 target: {target2}")
        
    finally:
        if os.path.exists(dummy_root):
            shutil.rmtree(dummy_root)

if __name__ == "__main__":
    test_image_folder_loading_recursive()
