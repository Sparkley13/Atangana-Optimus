# Atangana-Optimus

Optimiseur PyTorch inspiré de l'opérateur d'Atangana-Baleanu pour l'apprentissage automatique fractionnaire.

## Description

Atangana-Optimus est un optimiseur PyTorch novateur qui implémente les principes des dérivées fractionnaires via l'opérateur d'Atangana-Baleanu. Cet optimiseur apporte une mémoire temporelle intrinsèque aux réseaux de neurones, permettant potentiellement une convergence plus stable et efficace.

## Fonctionnalités

- **Mémoire fractionnaire** : Accumulation temporelle des gradients via un noyau non-singulier
- **Ordre fractionnaire ajustable** : Paramètre `alpha` contrôlant l'ordre de dérivation (0 < α ≤ 1)
- **Momentum fractionnaire** : Extension du momentum classique aux dérivées fractionnaires
- **Protection numérique** : Stabilité garantie même avec des gradients extrêmes
- **Compatible PyTorch** : Interface standard d'optimiseur PyTorch

## Installation

```bash
pip install torch numpy scipy matplotlib
```

## Utilisation rapide

```python
import torch
from optimizer import AtanganaOptimus
from schedulers import AlphaScheduler

# Modèle simple
model = torch.nn.Linear(10, 1)
optimizer = AtanganaOptimus(model.parameters(), lr=0.01, alpha=0.9)
scheduler = AlphaScheduler(optimizer, mode='decay', start_alpha=0.9, end_alpha=0.5)

# Boucle d'entraînement
for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Mise à jour de alpha
```

## Structure du projet

```
Atangana-Optimus/
├── kernels.py          # Noyau d'Atangana-Baleanu et approximations
├── memory.py           # Gestion de la mémoire fractionnaire
├── optimizer.py        # Optimiseur AtanganaOptimus
├── schedulers.py       # Scheduler pour alpha
├── main.py             # Validation simple
├── benchmark.py        # Comparaison avec Adam
├── tests/
│   ├── test_atangana.py      # Tests unitaires
│   └── vision_benchmark.py   # Benchmark sur MNIST
└── .gitignore
```

## Benchmarks

### Régression linéaire
```bash
python benchmark.py
```
Compare AtanganaOptimus vs Adam sur 5 runs indépendants avec statistiques.

### Classification MNIST
```bash
python tests/vision_benchmark.py
```
Test sur réseau de neurones convolutionnel pour classification d'images.

## Tests

```bash
python -m pytest tests/
```

## Paramètres

- `lr` : Learning rate (défaut: 1e-3)
- `alpha` : Ordre fractionnaire, ∈ (0,1] (défaut: 0.8)
- `momentum` : Facteur de momentum (défaut: 0.9)
- `eps` : Protection numérique pour 1-alpha (défaut: 1e-4)
- `max_update` : Clip des mises à jour (défaut: 1.0)

## Théorie

L'optimiseur implémente la formule :
```
w_{t+1} = w_t - lr * (B(α)/(1-α)) * (∇L_t + mémoire_t)
```

Où :
- B(α) est le facteur de normalisation d'Atangana-Baleanu
- mémoire_t est l'accumulateur fractionnaire des gradients passés
- α contrôle l'ordre de dérivation fractionnaire

## Licence

MIT License - voir LICENSE pour plus de détails.

## Citation

Si vous utilisez ce code dans vos recherches, veuillez citer :

```
@software{atangana_optimus,
  title={Atangana-Optimus: Fractional-Order Optimizer for PyTorch},
  author={Linus tshipichick},
  year={2024},
  url={https://github.com/sparckley13/Atangana-Optimus}
}
```