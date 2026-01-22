import torch

class FractionalKernel:
    """
    Noyaux fractionnaires discrets cohérents avec
    une dérivée de type Grünwald–Letnikov tronquée.
    """

    @staticmethod
    def gl_weights(alpha: float, K: int, device=None) -> torch.Tensor:
        """
        Calcule les poids :
        w_k = (-1)^k * binom(alpha, k)

        Complexité : O(K)
        """
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha doit être dans (0,1)")

        device = device or torch.device("cpu")

        w = torch.zeros(K, device=device)
        w[0] = 1.0

        for k in range(1, K):
            w[k] = w[k - 1] * (alpha - (k - 1)) / k * (-1)

        return w

    @staticmethod
    def normalize(weights: torch.Tensor) -> torch.Tensor:
        """
        Normalisation optionnelle (énergétique),
        utile pour stabiliser les premières itérations.
        """
        norm = torch.sum(torch.abs(weights))
        if norm > 0:
            return weights / norm
        return weights
