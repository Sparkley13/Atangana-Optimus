import math
import torch

class AlphaScheduler:
    """
    Scheduler rigoureux de l'ordre fractionnaire alpha
    compatible avec une dérivée GL tronquée.
    """

    def __init__(
        self,
        optimizer,
        mode="decay",
        start_alpha=0.95,
        end_alpha=0.6,
        max_delta=0.01,
        memory_damping=0.5,
    ):
        """
        Args:
            max_delta: variation maximale de alpha par step (adiabatique)
            memory_damping: facteur d'atténuation de la mémoire lors d'un changement
        """
        self.optimizer = optimizer
        self.mode = mode
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha
        self.max_delta = max_delta
        self.memory_damping = memory_damping
        self.current_step = 0
        self.current_alpha = start_alpha

    def step(self):
        self.current_step += 1
        target_alpha = self._target_alpha()

        # Variation adiabatique
        delta = target_alpha - self.current_alpha
        delta = max(-self.max_delta, min(self.max_delta, delta))
        new_alpha = self.current_alpha + delta

        # Injection cohérente dans l'optimiseur
        for group in self.optimizer.param_groups:
            old_alpha = group["alpha"]
            group["alpha"] = new_alpha

            # Recalibrage des états internes
            for p in group["params"]:
                state = self.optimizer.state.get(p, {})
                if not state:
                    continue

                # Mise à jour des poids GL
                K = group["memory_size"]
                device = p.device
                state["weights"] = self.optimizer.fractional_weights(
                    new_alpha, K, device
                )

                # Amortissement de la mémoire existante
                if "grads" in state:
                    for i in range(len(state["grads"])):
                        state["grads"][i].mul_(self.memory_damping)

        self.current_alpha = new_alpha
        return new_alpha

    def _target_alpha(self) -> float:
        if self.mode == "decay":
            return self.end_alpha + (self.start_alpha - self.end_alpha) * \
                   math.exp(-0.01 * self.current_step)

        elif self.mode == "cosine":
            return self.end_alpha + 0.5 * (self.start_alpha - self.end_alpha) * \
                   (1 + math.cos(math.pi * self.current_step / 100))

        return self.start_alpha
