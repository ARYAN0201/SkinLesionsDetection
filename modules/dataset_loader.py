import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class SkinLesionDataset(Dataset):
    def __init__(self, df, label_map, augmentor = None,  transform = None):
        self.df = df.reset_index(drop = True)
        self.label_map = label_map
        self.augmentor = augmentor
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')

        is_target = row['dx'] in self.label_map['target_classes'] if 'target_classes' in self.label_map else False
        if self.augmentor:
            image = self.augmentor.augment(image, is_target_class = is_target)

        if self.transform:
            image = self.transform(image)

        label_idx = self.label_map['class_to_idx'][row['dx']]
        label_onehot = torch.nn.functional.one_hot(torch.tensor(label_idx), num_classes=len(self.label_map['class_to_idx'])).float()

        return image, label_onehot
        
def get_dataloader(df, label_map, augmentor, batch_size=16, shuffle=True):
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
            ])

        dataset = SkinLesionDataset(df, label_map, augmentor=augmentor, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader