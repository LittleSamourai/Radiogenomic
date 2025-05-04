import torch
from src.model import SimpleBrainTumorCNN

# Créer un modèle
model = SimpleBrainTumorCNN()

# Simuler un batch d'images
dummy_input = torch.randn(8, 4, 240, 240)  # (batch_size=8, channels=4, height=240, width=240)

# Faire une passe avant
outputs = model(dummy_input)

# Afficher la forme de sortie
print("Shape de la sortie :", outputs.shape)

