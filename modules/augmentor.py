import pandas as pd
from PIL import Image
import torchvision.transforms as T


class ImageAugmentor:
    def __init__(self,  base_transforms = None, target_transforms = None):

        # Base light augmentation for all images
        self.base_transforms = base_transforms or T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ])

        # Heavier augmentation for minority classes
        self.target_transforms = target_transforms or T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomRotation(30),
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
    
    def augment(self,image: Image.Image, is_target_class = False):
        if is_target_class:
            return self.target_transforms(image)
        else:
            return self.base_transforms(image)