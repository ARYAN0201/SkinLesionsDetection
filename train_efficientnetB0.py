import os
import pandas as pd
import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from modules.preprocessing import HAMPreprocessor
from modules.augmentor import ImageAugmentor
from modules.data_balancer import DataBalancer
from modules.dataset_loader import get_dataloader
from modules.trainer import Trainer
from sklearn.model_selection import train_test_split

image_dirs = ["/Users/aryandahiya/data/HAM10000_images"]
processor = HAMPreprocessor("/Users/aryandahiya/data/HAM10000_metadata.csv", image_dirs)
df = processor.preprocess()

labels = sorted(df['dx'].unique())
label_map = {
    'class_to_idx': {cls: i for i, cls in enumerate(labels)},
    'target_classes': ['bcc', 'bkl', 'mel', 'akiec', 'df', 'vasc']
}

train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=df['dx'], random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df['dx'], random_state=42)

# Balance only training data
target_count = train_df['dx'].value_counts().max()
balancer = DataBalancer(train_df, target_count)
balanced_train_df = balancer.balance()

augmentor = ImageAugmentor()
train_loader = get_dataloader(balanced_train_df, label_map, augmentor, batch_size=16)
val_loader = get_dataloader(val_df, label_map, augmentor = None, batch_size=16)
test_loader = get_dataloader(test_df, label_map, augmentor=None, batch_size=16)

print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, len(labels))
model.to(device)

trainer = Trainer(model, device, train_loader, val_loader)

checkpoint_dir = "checkpoints_efficientnet"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = None
start_epoch = 1
if checkpoint_path and os.path.isfile(checkpoint_path):
    start_epoch = trainer.load_checkpoint(checkpoint_path)
else:
    print("No checkpoint found or path is None, training from scratch.")

total_epochs = 1
remaining_epochs = total_epochs - (start_epoch - 1)
trainer.fit(remaining_epochs, checkpoint_dir=checkpoint_dir, start_epoch=start_epoch)