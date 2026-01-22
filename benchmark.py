import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import copy
from optimizer import AtanganaOptimus
from schedulers import AlphaScheduler

def run_rigorous_benchmark(num_runs=5):
    epochs = 60
    lr = 0.05

    all_adam = []
    all_atangana = []
    all_alpha = []

    print(f"\nBenchmark rigoureux sur {num_runs} runs indépendants")

    for run in range(num_runs):
        torch.manual_seed(42 + run * 1000)

        # Données légèrement plus bruitées (mais convexes)
        X = torch.randn(300, 1)
        noise = torch.randn(300, 1) * 0.8
        y = 5 * X + 2.0 + noise

        base_model = nn.Linear(1, 1)
        model_adam = copy.deepcopy(base_model)
        model_atangana = copy.deepcopy(base_model)

        criterion = nn.MSELoss()

        opt_adam = torch.optim.Adam(
            model_adam.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        opt_atangana = AtanganaOptimus(
            model_atangana.parameters(),
            lr=lr,
            alpha=0.9,
            momentum=0.9,
            damping=0.1,
            memory_size=20,
            max_update=1.0
        )

        scheduler = AlphaScheduler(
            opt_atangana,
            mode="decay",
            start_alpha=0.9,
            end_alpha=0.5,
            max_delta=0.01
        )

        hist_adam = []
        hist_atangana = []
        hist_alpha = []

        for epoch in range(epochs):
            # Adam
            opt_adam.zero_grad()
            loss_adam = criterion(model_adam(X), y)
            loss_adam.backward()
            opt_adam.step()
            hist_adam.append(loss_adam.item())

            # Atangana
            opt_atangana.zero_grad()
            loss_at = criterion(model_atangana(X), y)
            loss_at.backward()
            opt_atangana.step()
            alpha_val = scheduler.step()

            hist_atangana.append(loss_at.item())
            hist_alpha.append(alpha_val)

        all_adam.append(hist_adam)
        all_atangana.append(hist_atangana)
        all_alpha.append(hist_alpha)

        print(f"Run {run+1}/{num_runs} terminé")

    # Conversion numpy
    adam = np.array(all_adam)
    atangana = np.array(all_atangana)

    # Statistiques robustes
    adam_med = np.median(adam, axis=0)
    adam_iqr = np.percentile(adam, 75, axis=0) - np.percentile(adam, 25, axis=0)

    at_med = np.median(atangana, axis=0)
    at_iqr = np.percentile(atangana, 75, axis=0) - np.percentile(atangana, 25, axis=0)

    # Visualisation
    epochs_range = np.arange(epochs)

    plt.figure(figsize=(12, 7))

    plt.plot(epochs_range, adam_med, label="Adam (médiane)", linestyle="--")
    plt.fill_between(
        epochs_range,
        adam_med - adam_iqr / 2,
        adam_med + adam_iqr / 2,
        alpha=0.15
    )

    plt.plot(
        epochs_range,
        at_med,
        label="AtanganaOptimus fractionnaire",
        linewidth=2
    )
    plt.fill_between(
        epochs_range,
        at_med - at_iqr / 2,
        at_med + at_iqr / 2,
        alpha=0.2
    )

    plt.yscale("log")
    plt.xlabel("Époques")
    plt.ylabel("MSE (log)")
    plt.title("Benchmark rigoureux : Adam vs AtanganaOptimus fractionnaire")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    plt.savefig("benchmark_fractional_rigorous.png", dpi=130)
    plt.show()

    print("\n=== BILAN FINAL (médiane ± IQR/2) ===")
    print(f"Adam     : {adam_med[-1]:.6f} ± {adam_iqr[-1]/2:.6f}")
    print(f"Atangana : {at_med[-1]:.6f} ± {at_iqr[-1]/2:.6f}")

if __name__ == "__main__":
    run_rigorous_benchmark(num_runs=5)
