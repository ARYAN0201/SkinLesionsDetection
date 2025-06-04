import torch
import pandas as pd
import argparse
from sklearn.metrics import classification_report
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader
from modules.custom_cnn import Custom_CNN
from modules.dataset_loader import SkinLesionDataset  
from modules.augmentor import get_augmentor  
from torchvision import transforms

def load_model(model_name, num_classes):
    if model_name == "resnet":
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=weights)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        checkpoint = "checkpoints_resnet/checkpoint_epoch_5.pth"
    elif model_name == "efficientnet":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        checkpoint = "checkpoints_efficientnet/checkpoint_epoch_5.pth"
    elif model_name == "custom":
        model = Custom_CNN(num_classes=num_classes)
        checkpoint = "checkpoints_custom/checkpoint_epoch_5.pth"
    else:
        raise ValueError("Invalid model name.")

    model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
    model.eval()
    return model

def get_test_loader(test_df, label_map, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = SkinLesionDataset(test_df, label_map, augmentor=None, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def evaluate_model(model, data_loader, device, class_names, model_name):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            labels = labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)

    import numpy as np
    pred_classes = np.argmax(all_preds, axis=1)
    true_classes = np.argmax(all_labels, axis=1)

    report = classification_report(true_classes, pred_classes, target_names=class_names, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report['model'] = model_name
    return df_report, true_classes, pred_classes

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label_map = {
        'class_to_idx': {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
    }
    class_names = list(label_map['class_to_idx'].keys())

    print("Loading test data...")
    test_df = pd.read_csv("splits/test.csv")  # adjust path if needed
    test_loader = get_test_loader(test_df, label_map)
    
    all_reports = []
    for model_name in ["custom", "resnet", "efficientnet"]:
        print(f"\nEvaluating {model_name.upper()} model")
        model = load_model(model_name, num_classes=len(class_names))
        model = model.to(device)

        df_report, true_classes, pred_classes = evaluate_model(model, test_loader, device, class_names, model_name)
        all_reports.append(df_report)
    
    full_report_df = pd.concat(all_reports)
    full_report_df.to_csv("benchmarks/benchmark_results.csv")

if __name__ == "__main__":
    main()