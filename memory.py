import torch
from collections import deque

class GLFractionalMemory:
    """
    Mémoire fractionnaire discrète de type Grünwald–Letnikov (tronquée).
    """

    def __init__(self, memory_size: int, device: torch.device):
        self.memory_size = memory_size
        self.device = device
        self.grads = deque(maxlen=memory_size)

    def push(self, grad: torch.Tensor):
        """Ajoute un gradient à l'historique."""
        self.grads.appendleft(grad.clone())

    def compute(self, weights: torch.Tensor) -> torch.Tensor:
        """
        Calcule le gradient fractionnaire :
        sum_k w_k * g_{t-k}
        """
        frac_grad = torch.zeros_like(self.grads[0])
        for w, g in zip(weights, self.grads):
            frac_grad.add_(g, alpha=w)
        return frac_grad

    def damp(self, factor: float):
        """Amortissement de la mémoire (scheduler alpha)."""
        for i in range(len(self.grads)):
            self.grads[i].mul_(factor)

    def reset(self):
        self.grads.clear()
