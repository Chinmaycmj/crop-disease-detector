import kagglehub
import os

print("Starting PlantVillage dataset download via kagglehub...")

# Download latest version of the PlantVillage dataset
path = kagglehub.dataset_download("emmarex/plantdisease")

print("\n✅ Download Complete!")
print("Dataset cached at path:", path)

# Print available subfolders/classes
if os.path.exists(path):
    subfolders = os.listdir(path)
    print("\nDataset contents:", subfolders)