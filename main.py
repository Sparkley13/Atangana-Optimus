"""
Module: main.py
Description: Script de déploiement et validation de l'optimiseur Atangana-Optimus.
"""

import torch
import torch.nn as nn
from optimizer import AtanganaOptimus
from schedulers import AlphaScheduler

def run_validation():
    # 1. Configuration et Données Synthétiques
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    y = 3 * X + 0.5 + torch.randn(100, 1) * 0.1 # y = 3x + 0.5 + bruit

    # 2. Modèle Linéaire Simple
    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()

    # 3. Initialisation de l'écosystème Atangana
    optimizer = AtanganaOptimus(
        model.parameters(), 
        lr=0.01, 
        alpha=0.9,       # On commence avec une mémoire modérée
        weight_decay=1e-4
    )
    
    scheduler = AlphaScheduler(
        optimizer, 
        mode='decay', 
        start_alpha=0.95, 
        end_alpha=0.6
    )

    # 4. Boucle d'entraînement
    print("Début de la validation d'Atangana-Optimus...")
    print("-" * 40)
    
    for epoch in range(1, 11):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass et Optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Mise à jour de l'Alpha (Logique fractionnaire)
        current_alpha = scheduler.step()

        if epoch % 2 == 0:
            print(f"Époque [{epoch}/10] | Loss: {loss.item():.4f} | Alpha: {current_alpha:.4f}")

    print("-" * 40)
    print("Validation terminée avec succès.")

if __name__ == "__main__":
    run_validation()