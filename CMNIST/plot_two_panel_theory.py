#!/usr/bin/env python3
"""Generate the two-panel identification and ranking theory figure."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from priority7_theory import cvar, risk_profiles


def admissible_cvar(losses, epsilon, alpha, completion):
    n_domains = len(losses)
    observed_domains = max(1, int(round(n_domains * (1.0 - epsilon))))
    observed_weight = (1.0 - epsilon) / observed_domains
    values = np.concatenate([losses[:observed_domains], [completion]])
    weights = np.concatenate([
        np.full(observed_domains, observed_weight),
        [epsilon],
    ])
    return cvar(values, weights, alpha)


def build_frame(epsilons, alpha, n_domains, tradeoff):
    profiles = risk_profiles(n_domains, tradeoff)
    rows = []
    for epsilon in epsilons:
        lower_f = admissible_cvar(profiles["head_favored"], epsilon, alpha, 0.0)
        upper_f = admissible_cvar(profiles["head_favored"], epsilon, alpha, 1.0)
        lower_g = admissible_cvar(profiles["tail_favored"], epsilon, alpha, 0.0)
        upper_g = admissible_cvar(profiles["tail_favored"], epsilon, alpha, 1.0)
        world_a = admissible_cvar(profiles["head_favored"], epsilon, alpha, 0.0) - admissible_cvar(
            profiles["tail_favored"], epsilon, alpha, 1.0
        )
        world_b = admissible_cvar(profiles["head_favored"], epsilon, alpha, 1.0) - admissible_cvar(
            profiles["tail_favored"], epsilon, alpha, 0.0
        )
        rows.append({
            "epsilon": epsilon,
            "lower_f": lower_f,
            "upper_f": upper_f,
            "lower_g": lower_g,
            "upper_g": upper_g,
            "world_a_gap": world_a,
            "world_b_gap": world_b,
        })
    return pd.DataFrame(rows)


def make_figure(frame, alpha, output_path):
    colors = {
        "f": "#0072B2",
        "g": "#D55E00",
        "world_a": "#009E73",
        "world_b": "#CC79A7",
    }
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    identification, ranking = axes

    identification.plot(frame["epsilon"], frame["lower_f"], color=colors["f"], linewidth=2.4, label=r"$L_\alpha(f)$")
    identification.plot(frame["epsilon"], frame["upper_f"], color=colors["f"], linewidth=2.4, linestyle="--", label=r"$U_\alpha(f)$")
    identification.plot(frame["epsilon"], frame["lower_g"], color=colors["g"], linewidth=2.4, label=r"$L_\alpha(g)$")
    identification.plot(frame["epsilon"], frame["upper_g"], color=colors["g"], linewidth=2.4, linestyle="--", label=r"$U_\alpha(g)$")
    identification.fill_between(frame["epsilon"], frame["lower_f"], frame["upper_f"], color=colors["f"], alpha=0.10)
    identification.fill_between(frame["epsilon"], frame["lower_g"], frame["upper_g"], color=colors["g"], alpha=0.10)
    identification.set_title("(a) Identification", loc="left", fontsize=17, fontweight="bold")
    identification.set_xlabel(r"Missing mass $\epsilon$", fontsize=15)
    identification.set_ylabel(r"CVaR ($\alpha=" + f"{alpha:g}" + r"$)", fontsize=15)
    identification.set_ylim(0.0, 1.0)
    identification.grid(True, linestyle="--", alpha=0.35)
    identification.tick_params(axis="both", labelsize=13)
    identification.legend(ncol=2, fontsize=12, frameon=True)

    ranking.plot(frame["epsilon"], frame["world_a_gap"], color=colors["world_a"], linewidth=2.5, label="World A: $f$ low, $g$ high")
    ranking.plot(frame["epsilon"], frame["world_b_gap"], color=colors["world_b"], linewidth=2.5, linestyle="--", label="World B: $f$ high, $g$ low")
    ranking.axhline(0.0, color="black", linewidth=1.2)
    ambiguous = (frame["world_a_gap"] <= 0.0) & (frame["world_b_gap"] >= 0.0)
    ambiguous_indices = np.flatnonzero(ambiguous.to_numpy())
    if ambiguous_indices.size:
        crossing_index = int(ambiguous_indices[0])
        crossing_epsilon = float(frame.iloc[crossing_index]["epsilon"])
        if crossing_index > 0:
            previous = frame.iloc[crossing_index - 1]
            current = frame.iloc[crossing_index]
            crossing_epsilon = float(np.interp(
                0.0,
                [current["world_a_gap"], previous["world_a_gap"]],
                [current["epsilon"], previous["epsilon"]],
            ))
        ranking.annotate(
            "ranking no longer identified",
            xy=(crossing_epsilon, 0.0),
            xytext=(crossing_epsilon + 0.035, 0.12),
            arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.2},
            fontsize=13,
            ha="left",
        )
    ranking.fill_between(frame["epsilon"], frame["world_a_gap"], frame["world_b_gap"], color="#777777", alpha=0.10)
    ranking.set_title("(b) Ranking", loc="left", fontsize=17, fontweight="bold")
    ranking.set_xlabel(r"Missing mass $\epsilon$", fontsize=15)
    ranking.set_ylabel(r"Deployment-CVaR difference $\rho_\alpha(f)-\rho_\alpha(g)$", fontsize=15)
    ranking.grid(True, linestyle="--", alpha=0.35)
    ranking.tick_params(axis="both", labelsize=13)
    ranking.legend(fontsize=12, frameon=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="../results/P7_theory")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--n_domains", type=int, default=100)
    parser.add_argument("--tradeoff", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epsilons = np.linspace(0.0, 0.4, 81)
    frame = build_frame(epsilons, args.alpha, args.n_domains, args.tradeoff)
    frame.to_csv(output_dir / "identification_ranking_two_panel.csv", index=False)
    make_figure(frame, args.alpha, output_dir / "identification_ranking_two_panel.png")
    print(f"Wrote {output_dir / 'identification_ranking_two_panel.png'}")
    print(f"Wrote {output_dir / 'identification_ranking_two_panel.csv'}")


if __name__ == "__main__":
    main()
