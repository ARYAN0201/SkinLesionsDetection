import os
import pandas as pd
import torch
from modules.preprocessing import HAMPreprocessor
from modules.augmentor import ImageAugmentor
from modules.data_balancer import DataBalancer
from modules.dataset_loader import get_dataloader
from torchvision.models import resnet50, ResNet50_Weights
from modules.trainer import Trainer
from modules.custom_cnn import Custom_CNN

# 1. Paths
image_dirs = [
    "/home/Ujjwal/Aryan/HAM10000/ham10000_images_part_1",
    "/home/Ujjwal/Aryan/HAM10000/HAM10000_images_part_1",
    "/home/Ujjwal/Aryan/HAM10000/ham10000_images_part_2",
    "/home/Ujjwal/Aryan/HAM10000/HAM10000_images_part_2"
]

# 2. Preprocess
processor = HAMPreprocessor("/home/Ujjwal/Aryan/HAM10000/HAM10000_metadata.csv", image_dirs)
df = processor.preprocess()

# 3. Define class info
labels = sorted(df['dx'].unique())
label_map = {
    'class_to_idx': {cls: i for i, cls in enumerate(labels)},
    'target_classes': ['bcc','bkl', 'mel',' akiec', 'df', 'vasc']  # classes to augment heavily
}

# 4. Balance data
target_count = df['dx'].value_counts().max()
balancer = DataBalancer(df, target_count)
balanced_df = balancer.balance()

# 5. Augmentation
augmentor = ImageAugmentor()

# 6. Dataloader
train_loader = get_dataloader(balanced_df, label_map, augmentor, batch_size=16)

# 7. Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = Custom_CNN()
model.to(device)

# 8. Trainer
trainer = Trainer(model, device, train_loader)

# 9. Checkpoint directory
checkpoint_dir = "checkpoints_custom_cnn"
os.makedirs(checkpoint_dir, exist_ok=True)

# 10. Optionally resume from checkpoint (set to None to train from scratch)
checkpoint_path = None  # e.g. "checkpoints/checkpoint_epoch_5.pth"

start_epoch = 1
if checkpoint_path and os.path.isfile(checkpoint_path):
    start_epoch = trainer.load_checkpoint(checkpoint_path)
else:
    print("No checkpoint found or path is None, training from scratch.")

# 11. Train remaining epochs
total_epochs = 5
remaining_epochs = total_epochs - (start_epoch - 1)
trainer.fit(remaining_epochs, checkpoint_dir=checkpoint_dir, start_epoch=start_epoch)
