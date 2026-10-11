#!/usr/bin/env python3
"""Export the canonical ImageNet-100-C E3b hyperparameter table."""
from __future__ import annotations

from pathlib import Path


ROWS = [
    ("ERM", "ResNet-50 IMAGENET1K_V2; layer4+head", "1000", "0", "AdamW; no cosine", "-"),
    ("IRM", "ResNet-50 IMAGENET1K_V2; layer4+head", "1000", "400", "AdamW -> Adam; cosine 3e-4 to 0 over 600", "penalty=0.3"),
    ("VREx", "ResNet-50 IMAGENET1K_V2; layer4+head", "1000", "400", "AdamW -> Adam; cosine 3e-4 to 0 over 600", "penalty=0.3"),
    ("EQRM", "ResNet-50 IMAGENET1K_V2; layer4+head", "1000", "400", "AdamW -> Adam; cosine 3e-4 to 0 over 600", "alpha=0.9"),
    ("GroupDRO", "ResNet-50 IMAGENET1K_V2; layer4+head", "1000", "400", "AdamW -> Adam; cosine 3e-4 to 0 over 600", "eta=0.1"),
    ("INF-TASK", "ResNet-50 IMAGENET1K_V2; layer4+head", "1000", "400", "AdamW -> Adam; cosine 3e-4 to 0 over 600", "4 Beta(1,1) lambdas/update"),
    ("IRO", "ResNet-50 IMAGENET1K_V2; layer4+head", "1000", "400", "AdamW -> Adam; cosine 3e-4 to 0 over 600", "4 Beta(1,1) lambdas/update; sampler lr=1e-6"),
]


def latex_escape(value: str) -> str:
    return value.replace("&", r"\&")


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "results" / "reports" / "E3b" / "ImgNet" / "imagenet100c_hyperparameters.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Canonical ImageNet-100-C E3b training hyperparameters. All methods use ResNet-50 with torchvision \texttt{IMAGENET1K\_V2} weights, batch size 64, learning rate $3\times10^{-4}$, and weight decay $10^{-4}$. Non-ERM methods use 400 ERM warm-up updates followed by an Adam reset and $600$-step cosine decay.}",
        r"\label{tab:imagenet100c-hyperparameters}",
        r"\begin{tabular}{@{}lllllll@{}}",
        r"\toprule",
        r"Method & Backbone/training & Steps & Warm-up & Batch/LR/WD & Optimizer/schedule & Method setting \\",
        r"\midrule",
    ]
    for row in ROWS:
        lines.append(" & ".join([latex_escape(row[0]), latex_escape(row[1]), row[2], row[3], "64 / 3e-4 / 1e-4", latex_escape(row[4]), latex_escape(row[5])]) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
