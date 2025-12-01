import argparse, os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import FaceDataset
from models import SimpleViT
from tqdm import tqdm
import json

def train(args):
    train_ds = FaceDataset(args.data_dir, split='train', transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))
    val_ds = FaceDataset(args.data_dir, split='val', transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
    ]))
    num_classes = len(train_ds.class_to_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleViT(num_classes=num_classes, vit_name=args.vit_name, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        # val
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds==labels).sum().item()
                total += labels.size(0)
        acc = correct/total if total>0 else 0.0
        print(f"Validation accuracy: {acc:.4f}")
        

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.out)
            # Simpan mapping class → index
            with open(args.out.replace(".pth", "_classes.json"), "w") as f:
                json.dump(train_ds.class_to_idx, f)

    print("Training finished. Best val acc:", best_acc)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--vit-name', default='vit_base_patch16_224')
    parser.add_argument('--out', default='models/vit_best.pth')
    args = parser.parse_args()
    train(args)
