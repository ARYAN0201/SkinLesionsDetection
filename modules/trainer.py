import os
import torch
from tqdm import tqdm

class Trainer:
    def __init__(self, model, device, train_loader, val_loader = None, optimizer=None, criterion=None):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer or torch.optim.Adam(model.parameters())
        self.criterion = criterion or torch.nn.CrossEntropyLoss()
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc="Training Batches", leave=False)
        for images, labels in loop:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        return total_loss / len(self.train_loader)

    def save_checkpoint(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved at epoch {epoch} to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}, resuming from epoch {checkpoint['epoch'] + 1}")
        return checkpoint['epoch'] + 1

    def fit(self, epochs, checkpoint_dir="checkpoints", start_epoch=1):
        os.makedirs(checkpoint_dir, exist_ok=True)
        epoch_loop = tqdm(range(start_epoch, start_epoch + epochs), desc="Epochs")
        for epoch in epoch_loop:
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate() if self.val_loader else (0, 0)
            
            epoch_loop.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.4f}")

            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            self.save_checkpoint(checkpoint_path, epoch)

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                preds = torch.sigmoid(outputs)
                predicted = (preds > 0.5).float()

                correct += (predicted == labels).sum().item()
                total += labels.numel()

        avg_loss = val_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy