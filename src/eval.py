try:
    print(">>> Début du script")
    import torch
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    from dataset import BrainTumorDataset
    from model import SimpleBrainTumorCNN
    from torch.utils.data import DataLoader, Subset

    print(">>> Imports OK")

    # Charger le dataset complet
    dataset = BrainTumorDataset(data_dir='data', labels_csv='train_labels.csv')

    # Charger les indices du split sauvegardé
    split = torch.load("split_indices.pt")
    val_indices = split['val']

    # Créer le dataset de validation à partir des indices
    val_dataset = Subset(dataset, val_indices)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    print(">>> Dataset chargé (val set)")

    # Charger le modèle
    model = SimpleBrainTumorCNN()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    print(">>> Modèle chargé")

    # Collecte des proba
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.numpy())
            all_labels.extend(labels.numpy())

    # Calculer ROC et AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    print(f">>> AUC : {roc_auc:.4f}")

    # Tracer et sauvegarder la courbe
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve on Validation Set')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print(">>> Image enregistrée en roc_curve.png")
    plt.show()

except Exception as e:
    print(f"[ERREUR] : {e}")
