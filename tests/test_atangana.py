import torch
from optimizer import AtanganaOptimus
from memory import GLFractionalMemory


def test_fractional_memory_update():
    torch.manual_seed(0)
    device = torch.device('cpu')
    mem = GLFractionalMemory(memory_size=10, device=device)

    grad = torch.tensor([[1.0, -2.0], [0.5, 0.0]])
    alpha = 0.8

    # Test de push et compute
    mem.push(grad)
    weights = torch.tensor([1.0, -alpha])  # Poids simplifiés pour le test
    result = mem.compute(weights)
    assert result.shape == grad.shape
    assert torch.isfinite(result).all()


def test_atangana_optimizer_initializes_memory_and_updates_params():
    torch.manual_seed(0)

    model = torch.nn.Linear(1, 1)
    optimizer = AtanganaOptimus(model.parameters(), lr=0.1, alpha=0.9)

    X = torch.tensor([[1.0], [2.0]])
    y = torch.tensor([[2.0], [4.0]])
    criterion = torch.nn.MSELoss()

    # Sauvegarde des paramètres initiaux
    params_before = [p.clone().detach() for p in model.parameters()]

    # Backward + step
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()

    # Vérifie que l'état a été initialisé
    for p in model.parameters():
        state = optimizer.state[p]
        assert 'grads' in state
        assert 'velocity' in state
        assert 'weights' in state

    # Vérifie que les paramètres ont été mis à jour
    params_after = [p.clone().detach() for p in model.parameters()]
    for before, after in zip(params_before, params_after):
        assert not torch.allclose(before, after)


def test_atangana_optimizer_numerical_stability():
    torch.manual_seed(0)

    model = torch.nn.Linear(2, 1)  # Augmenter la taille pour avoir plus de paramètres
    optimizer = AtanganaOptimus(model.parameters(), lr=0.1, alpha=0.9, eps=1e-6, max_update=10.0)

    X = torch.tensor([[1.0, 0.5], [2.0, 1.0]])
    y = torch.tensor([[2.0], [4.0]])
    criterion = torch.nn.MSELoss()

    # Injecter des gradients NaN/inf pour tester la robustesse
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()

    # Simuler des gradients problématiques
    for p in model.parameters():
        if p.grad is not None:
            p.grad.fill_(float('nan'))  # Remplir tout le gradient avec NaN
            p.grad[0] = float('inf')    # Injecter inf dans le premier élément

    # Vérifier que step() ne crash pas et que les paramètres restent finis
    try:
        optimizer.step()
        for p in model.parameters():
            assert torch.isfinite(p).all(), "Paramètres contiennent des valeurs non finies après step()"
    except Exception as e:
        assert False, f"Optimizer a crashé avec des gradients NaN/inf: {e}"
