# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from typing import Any, Callable, Optional, Tuple

from .decoders import ImageDataDecoder, TargetDecoder
from .extended import ExtendedVisionDataset


class ImageFolderDataset(ExtendedVisionDataset):
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=ImageDataDecoder,
            target_decoder=TargetDecoder,
        )
        self.samples = self._make_recursive_dataset(root)

    def _make_recursive_dataset(self, root: str) -> list[tuple[str, int]]:
        samples = []
        # Get immediate subdirectories as classes
        classes = sorted(d.name for d in os.scandir(root) if d.is_dir())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        
        for target_class in classes:
            target_dir = os.path.join(root, target_class)
            for root_dir, _, filenames in os.walk(target_dir):
                for filename in filenames:
                    if filename.lower().endswith(extensions):
                        path = os.path.join(root_dir, filename)
                        samples.append((path, class_to_idx[target_class]))
        
        if not samples:
            raise FileNotFoundError(f"Found no valid file for the dataset in {root}")
            
        return samples

    def get_image_data(self, index: int) -> bytes:
        image_path, _ = self.samples[index]
        with open(image_path, "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        _, target = self.samples[index]
        return target

    def get_targets(self) -> Any:
        return [target for _, target in self.samples]

    def __len__(self) -> int:
        return len(self.samples)
