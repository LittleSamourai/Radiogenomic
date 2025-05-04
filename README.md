Author : Samir Tiouririne

# Radiogenomic Brain Tumor Classification using CNNs

This project applies deep learning to predict MGMT promoter methylation status in glioma patients using MRI scans. The solution uses a lightweight 2D CNN trained on data from the BraTS 2021 radiogenomic challenge.

## Project Goals
- Predict MGMT promoter methylation (binary classification)
- Use only MRI data (no clinical or genomic data)
- Build a clean and reproducible deep learning pipeline

## Dataset
- BraTS 2021 radiogenomic dataset
- 4 MRI modalities: FLAIR, T1w, T1wCE, T2w
- Labels: methylated (1) or unmethylated (0)
- Due to extraction issues, only 194 patients were used

## Preprocessing
- Central slices extracted from each modality
- Images resized and normalized
- Stacked into tensors of shape (channels, height, width)

## Model
- 2D CNN implemented with PyTorch
- 3 convolutional layers + 2 fully connected layers
- Trained using CrossEntropyLoss and Adam optimizer

## Training Setup
- 80/20 train/validation split (saved to file for consistency)
- 5 training epochs
- ROC curve used to evaluate classification performance

## Results
- AUC with 1 slice per modality: ~0.50
- AUC with 5 central slices per modality: ~0.58

## File Structure
- `src/` : code (dataset, model, train, eval)
- `data/` : patient MRI data
- `model.pth` : saved model
- `split_indices.pt` : saved train/val split
- `roc_curve.png` : output ROC curve
-

## How to Run
- Launch the train.py
