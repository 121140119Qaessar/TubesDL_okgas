import os
from PIL import Image

root = "./dataset/train"

bad_files = []

for person in os.listdir(root):
    person_path = os.path.join(root, person)
    if not os.path.isdir(person_path):
        continue

    for file in os.listdir(person_path):
        path = os.path.join(person_path, file)
        try:
            img = Image.open(path)
            img.verify()  # verify file integrity
        except Exception as e:
            print("BAD:", path)
            bad_files.append(path)

print("\nTotal bad files:", len(bad_files))
print(sorted(os.listdir("dataset/train")))
print(sorted(os.listdir("dataset/val")))
