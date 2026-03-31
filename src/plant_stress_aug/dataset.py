import random
from typing import List

import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .augmentations import AUGMENTATION_FUNCTIONS


class ImageFolderCustom(ImageFolder):
    """Apply class-specific augmentation probabilities encoded as a chromosome."""

    def __init__(self, dataset_dir: str, augmentation_names: List[str], chromosome):
        super().__init__(dataset_dir)
        self.augmentation_names = augmentation_names
        self.chromosome = np.reshape(chromosome, (len(self.classes), -1))

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = transforms.Resize((224, 224))(sample)

        augment_probabilities = self.chromosome[target]
        actual_probabilities = [round(random.random(), 1) for _ in range(len(augment_probabilities))]

        for act_p, aug_p, aug_name in zip(actual_probabilities, augment_probabilities, self.augmentation_names):
            if act_p > aug_p and aug_name in AUGMENTATION_FUNCTIONS:
                sample = AUGMENTATION_FUNCTIONS[aug_name](sample)

        sample = transforms.ToTensor()(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
