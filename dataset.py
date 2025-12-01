import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class FaceDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root = os.path.join(root_dir, split)
        self.transform = transform or T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),
        ])
        self.samples = []
        self.class_to_idx = {}
        for i, cls in enumerate(sorted(os.listdir(self.root))):
            cls_path = os.path.join(self.root, cls)
            if not os.path.isdir(cls_path):
                continue
            self.class_to_idx[cls] = i
            for fn in os.listdir(cls_path):
                if fn.lower().endswith(('.jpg','.jpeg','.png')):
                    self.samples.append((os.path.join(cls_path, fn), i))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label
