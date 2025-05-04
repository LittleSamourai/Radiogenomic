from src.real_load_data import load_multimodal_slice
import matplotlib.pyplot as plt
import os

# Chemin vers le dossier du patient
patient_path = os.path.join("data", "BraTS2021_00153")
slice_index = 100

tensor = load_multimodal_slice(patient_path, slice_index)
print("Shape :", tensor.shape)  # (H, W, 4)

# Affichage des 4 modalités côte à côte
fig, axs = plt.subplots(1, 4, figsize=(12, 3))
modalities = ["flair", "t1", "t1ce", "t2"]

for i in range(4):
    axs[i].imshow(tensor[:, :, i], cmap='gray')
    axs[i].set_title(modalities[i])
    axs[i].axis("off")

plt.tight_layout()
plt.show()
