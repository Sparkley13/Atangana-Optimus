import torch
from torch.optim import Optimizer
from collections import deque
import math

class AtanganaOptimus(Optimizer):
    """
    Optimiseur fractionnaire Atangana–Baleanu discret (GL tronqué)
    avec momentum et amortissement énergétique invariant.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        alpha=0.8,
        momentum=0.9,
        weight_decay=0.0,
        eps=1e-8,
        max_update=1.0,
        damping=0.1,
        memory_size=20
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha doit être dans (0,1)")

        defaults = dict(
            lr=lr,
            alpha=alpha,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
            max_update=max_update,
            damping=damping,
            memory_size=memory_size,
        )
        super().__init__(params, defaults)

    @staticmethod
    def fractional_weights(alpha, K, device):
        """Poids GL : (-1)^k * C(alpha, k)"""
        w = torch.zeros(K, device=device)
        w[0] = 1.0
        for k in range(1, K):
            w[k] = w[k-1] * (alpha - (k - 1)) / k * (-1)
        return w

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            alpha = group["alpha"]
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            eps = group["eps"]
            gamma = group["damping"]
            K = group["memory_size"]
            max_update = group["max_update"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = torch.nan_to_num(p.grad)

                if wd > 0:
                    grad = grad.add(p.data, alpha=wd)

                state = self.state[p]

                if len(state) == 0:
                    state["grads"] = deque(maxlen=K)
                    state["velocity"] = torch.zeros_like(p.data)
                    state["weights"] = self.fractional_weights(
                        alpha, K, p.device
                    )

                state["grads"].appendleft(grad.clone())

                # Gradient fractionnaire discret
                frac_grad = torch.zeros_like(p.data)
                for w, g in zip(state["weights"], state["grads"]):
                    frac_grad.add_(g, alpha=w)

                # Amortissement énergétique invariant
                rms = torch.sqrt(torch.mean(frac_grad ** 2) + eps)
                damping_factor = 1.0 / (1.0 + gamma * rms)

                # Momentum
                velocity = state["velocity"]
                velocity.mul_(mu).add_(frac_grad, alpha=damping_factor)

                # Clipping
                velocity.clamp_(-max_update, max_update)

                # Mise à jour finale
                p.data.add_(velocity, alpha=-lr)

        return loss
