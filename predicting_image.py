import argparse
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from modules.prediction import Predictor
from modules.custom_cnn import Custom_CNN  

def load_model(model_name, num_classes):
    if model_name == "resnet":
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, num_classes),
        )
    elif model_name == "efficientnet":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = torch.nn.Sequential(
            torch.nn.Linear(model.classifier[1].in_features, num_classes),
        )
    elif model_name == "custom":
        model = Custom_CNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Skin Lesion Classifier")
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "efficientnet", "custom"], help="Model to use")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path")

    args = parser.parse_args()

    # Make sure this matches your training setup
    label_map = {
        'class_to_idx': {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
    }

    checkpoint_default_paths = {
        "resnet": "checkpoints_resnet/checkpoint_epoch_5.pth",
        "efficientnet": "checkpoints_efficientnet/checkpoint_epoch_5.pth",
        "custom": "checkpoints_custom/checkpoint_epoch_5.pth",
    }

    checkpoint_path = args.checkpoint or checkpoint_default_paths[args.model]

    model = load_model(args.model, num_classes=len(label_map['class_to_idx']))
    predictor = Predictor(model, label_map, checkpoint_path)

    predicted_class, class_probs = predictor.predict(args.image)

    print(f"\nPredicted class: {predicted_class}")
    print("Class probabilities:")
    for cls, prob in class_probs.items():
        print(f"{cls}: {prob:.4f}")

if __name__ == "__main__":
    main()