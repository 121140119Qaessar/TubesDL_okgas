import os

data_dir = "dataset/train"

kelas = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

print("Jumlah kelas:", len(kelas))
print("Daftar kelas:")
for k in kelas:
    print("-", k)
