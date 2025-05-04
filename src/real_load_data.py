import os
import pydicom
import numpy as np
import nibabel as nib
from skimage.transform import resize

def load_dicom(path):
    dicom_data = pydicom.dcmread(path)
    return dicom_data.pixel_array

"""
def load_multimodal_slice(patient_dir, slice_idx=None, target_size=(240, 240)):
    import os
    import numpy as np
    import pydicom
    from skimage.transform import resize

    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
    slices = []

    for modality in modalities:
        modality_path = os.path.join(patient_dir, modality)
        dicom_files = [os.path.join(modality_path, f) for f in os.listdir(modality_path) if f.endswith('.dcm')]

        if len(dicom_files) == 0:
            raise FileNotFoundError(f"Aucun fichier DICOM trouvé pour la modalité {modality} dans {modality_path}")

        dicoms = [(pydicom.dcmread(f), f) for f in dicom_files]
        dicoms.sort(key=lambda x: x[0].InstanceNumber)

        # Choisir une slice
        chosen_idx = slice_idx
        if slice_idx is None or slice_idx >= len(dicoms):
            chosen_idx = len(dicoms) // 2

        volume = dicoms[chosen_idx][0].pixel_array

        # 🔥 Ajout ici : resize pour garantir la même taille
        if volume.shape != target_size:
            volume = resize(volume, target_size, preserve_range=True, anti_aliasing=True)
            volume = volume.astype(np.float32)

        slices.append(volume)

    stacked = np.stack(slices, axis=-1)
    return stacked
"""

import os
import numpy as np
import pydicom
from skimage.transform import resize

def load_dicom_volume(dicom_dir):
    slices = []
    for fname in os.listdir(dicom_dir):
        if fname.endswith(".dcm"):
            path = os.path.join(dicom_dir, fname)
            ds = pydicom.dcmread(path)
            slices.append((ds.InstanceNumber, ds.pixel_array))
    slices.sort(key=lambda x: x[0])
    volume = [s[1] for s in slices]
    return np.array(volume)

def load_multimodal_slice(patient_dir, output_size=(240, 240), num_slices=5):
    modalities = ["FLAIR", "T1w", "T1wCE", "T2w"]
    stacked_slices = []

    for modality in modalities:
        modality_dir = os.path.join(patient_dir, modality)
        volume = load_dicom_volume(modality_dir)

        # Choisir les slices centrales
        total_slices = volume.shape[0]
        center = total_slices // 2
        half = num_slices // 2
        start = max(center - half, 0)
        end = min(center + half + 1, total_slices)

        selected = volume[start:end]
        # Si pas assez de slices (ex: 3 sur les 5 demandées), on complète
        while selected.shape[0] < num_slices:
            if start > 0:
                start -= 1
                selected = np.insert(selected, 0, volume[start], axis=0)
            elif end < total_slices:
                selected = np.append(selected, [volume[end]], axis=0)
                end += 1
            else:
                break  # pas d'autres slices dispos

        for slice_img in selected:
            resized = resize(slice_img, output_size, mode='constant', preserve_range=True)
            normed = (resized - np.min(resized)) / (np.max(resized) - np.min(resized) + 1e-8)
            stacked_slices.append(normed)

    return np.array(stacked_slices, dtype=np.float32)  # shape = (20, 240, 240)
