# schedulers.py
import math

class AlphaScheduler:
    """
    Planifie l'évolution de l'ordre fractionnaire alpha.
    Alpha décroît progressivement pour introduire la mémoire longue.
    """

    def __init__(
        self,
        optimizer,
        mode: str = "decay",
        start_alpha: float = 0.99,
        end_alpha: float = 0.6,
        max_steps: int = 10_000
    ):
        assert 0 < end_alpha <= start_alpha <= 1.0
        self.optimizer = optimizer
        self.mode = mode
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha
        self.max_steps = max_steps
        self.current_step = 0

    def step(self) -> float:
        self.current_step += 1
        alpha = self._calculate_alpha()

        for group in self.optimizer.param_groups:
            group["alpha"] = alpha

        return alpha

    def _calculate_alpha(self) -> float:
        t = min(self.current_step / self.max_steps, 1.0)

        if self.mode == "decay":
            # Décroissance exponentielle normalisée
            return self.end_alpha + (self.start_alpha - self.end_alpha) * math.exp(-5 * t)

        elif self.mode == "cosine":
            # Cosine decay bornée
            return self.end_alpha + 0.5 * (self.start_alpha - self.end_alpha) * (1 + math.cos(math.pi * t))

        return self.start_alpha
