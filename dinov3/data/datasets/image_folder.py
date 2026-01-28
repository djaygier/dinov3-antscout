# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import ImageFolder

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
        self.dataset = ImageFolder(root)

    def get_image_data(self, index: int) -> bytes:
        image_path, _ = self.dataset.samples[index]
        with open(image_path, "rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        _, target = self.dataset.samples[index]
        return target

    def get_targets(self) -> Any:
        return [target for _, target in self.dataset.samples]

    def __len__(self) -> int:
        return len(self.dataset)
