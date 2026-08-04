import os
import kagglehub

# 1. Fetch cached dataset path directly
print("Locating downloaded PlantVillage dataset...")
dataset_path = kagglehub.dataset_download("emmarex/plantdisease")

# Find the main directory containing plant disease subfolders
data_dir = dataset_path
for root, dirs, files in os.walk(dataset_path):
    if len(dirs) > 5:
        data_dir = root
        break

print(f"Dataset root identified at: {data_dir}")

# Print discovered classes
classes = sorted(os.listdir(data_dir))
print(f"\nTotal classes found: {len(classes)}")
print("Sample classes:", classes[:5])