
import os, shutil, random
import argparse

def split_dataset(root, val_ratio=0.2):
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    os.makedirs(val_dir, exist_ok=True)

    for person in os.listdir(train_dir):
        person_path = os.path.join(train_dir, person)
        if not os.path.isdir(person_path):
            continue

        images = [f for f in os.listdir(person_path) if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))]
        random.shuffle(images)

        val_count = int(len(images) * val_ratio)
        val_images = images[:val_count]

        out_person_val = os.path.join(val_dir, person)
        os.makedirs(out_person_val, exist_ok=True)

        for img in val_images:
            src = os.path.join(person_path, img)
            dst = os.path.join(out_person_val, img)
            shutil.move(src, dst)

        print(f"Moved {len(val_images)} images from {person} to val/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    args = parser.parse_args()
    split_dataset(args.data_dir, args.val_ratio)
