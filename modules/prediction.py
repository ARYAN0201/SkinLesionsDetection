import torch
from torchvision import transforms
from PIL import Image

class Predictor:
    def __init__(self, model, label_map, checkpoint_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.label_map = label_map
        self.idx_to_class = {v: k for k, v in label_map['class_to_idx'].items()}

        self.model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze().cpu()

        predictions = {
            self.idx_to_class[i]: float(prob)
            for i, prob in enumerate(probs)
        }

        top_class = max(predictions, key=predictions.get)
        return top_class, predictions