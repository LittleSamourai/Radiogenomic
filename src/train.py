import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import BrainTumorDataset
from model import SimpleBrainTumorCNN

# Charger le Dataset complet
dataset = BrainTumorDataset(data_dir='data', labels_csv='train_labels.csv')

# Split en train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Sauvegarder les indices pour cohérence
train_indices = train_dataset.indices if hasattr(train_dataset, 'indices') else range(train_size)
val_indices = val_dataset.indices if hasattr(val_dataset, 'indices') else range(train_size, len(dataset))

torch.save({'train': train_indices, 'val': val_indices}, 'split_indices.pt')


# Créer les DataLoaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialiser lz modèle
model = SimpleBrainTumorCNN()

# Définir la loss et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Entraîner
for epoch in range(5):  # 5 epochs pour tester
    running_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} | Train Loss: {running_loss/len(train_loader):.4f}")

# Enregistrement du modèle
torch.save(model.state_dict(), "model.pth")
print("Modèle sauvegardé sous model.pth")