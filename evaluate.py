import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FaceDataset
from models import get_mobilenet, SimpleViT
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def evaluate(args):
    val_ds = FaceDataset(args.data_dir, split='val', transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    num_classes = len(val_ds.class_to_idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.arch=='cnn':
        model = get_mobilenet(num_classes, pretrained=False)
    else:
        model = SimpleViT(num_classes=num_classes, vit_name=args.vit_name, pretrained=False)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device).eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_pred.extend(list(preds))
            y_true.extend(list(labels.numpy()))
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    print('Confusion matrix:\n', cm)
    print(f'Accuracy: {acc:.4f}  F1: {f1:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--arch', choices=['cnn','vit'], default='cnn')
    parser.add_argument('--vit-name', default='vit_base_patch16_224')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    evaluate(args)
