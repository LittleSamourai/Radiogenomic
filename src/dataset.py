import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from real_load_data import load_multimodal_slice # Réutiliser du code

"""
class BrainTumorDataset(Dataset):

    def __init__(self, data_dir, labels_csv, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.labels_df = pd.read_csv(labels_csv)
        all_patient_ids = self.labels_df['BraTS21ID'].astype(str).str.zfill(5).tolist()

        # NE GARDER que les patients qui existent physiquement
        self.patient_ids = []
        for patient_id in all_patient_ids:
            patient_folder = os.path.join(self.data_dir, patient_id)
            if os.path.exists(patient_folder):
                self.patient_ids.append(patient_id)

        print(f"Nombre de patients trouvés avec données : {len(self.patient_ids)}")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label_row = self.labels_df[self.labels_df['BraTS21ID'] == int(patient_id)]
        label = label_row['MGMT_value'].values[0]

        patient_folder = os.path.join(self.data_dir, patient_id)

        print(f"Chargement du patient : {patient_id}")  

        try:
            image = load_multimodal_slice(patient_folder)
        except Exception as e:
            print(f"Erreur chargement patient {patient_id} : {e}")
            raise e

        image = torch.tensor(image, dtype=torch.float).permute(2, 0, 1)

        return image, torch.tensor(label, dtype=torch.long)
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from real_load_data import load_multimodal_slice

class BrainTumorDataset(Dataset):
    def __init__(self, data_dir, labels_csv, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.labels_df = pd.read_csv(labels_csv)
        all_patient_ids = self.labels_df['BraTS21ID'].astype(str).str.zfill(5).tolist()

        # Ne garder que les patients pour lesquels les données existent
        self.patient_ids = []
        for patient_id in all_patient_ids:
            patient_folder = os.path.join(self.data_dir, patient_id)
            if os.path.exists(patient_folder):
                self.patient_ids.append(patient_id)

        print(f"Nombre de patients trouvés avec données : {len(self.patient_ids)}")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label_row = self.labels_df[self.labels_df['BraTS21ID'] == int(patient_id)]
        label = label_row['MGMT_value'].values[0]

        patient_folder = os.path.join(self.data_dir, patient_id)
        print(f"Chargement du patient : {patient_id}")

        try:
            image = load_multimodal_slice(patient_folder)  # Ex: (240, 240, 20) ou (20, 240, 240)
        except Exception as e:
            print(f"Erreur chargement patient {patient_id} : {e}")
            raise e

        image = torch.tensor(image, dtype=torch.float32)

        if image.ndim != 3:
            raise ValueError(f"[ERREUR] Image avec {image.ndim} dimensions (attendu : 3) pour le patient {patient_id}")

        if image.shape[0] == 240 and image.shape[1] == 240:
            # Format (H, W, C) → on permute pour avoir (C, H, W)
            image = image.permute(2, 0, 1)
        elif image.shape[0] == 20:
            # Déjà au format (C, H, W) — on garde tel quel
            pass
        else:
            raise ValueError(f"[ERREUR] Format inattendu pour l'image du patient {patient_id} : {image.shape}")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
