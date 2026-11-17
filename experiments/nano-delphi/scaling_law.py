# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Modifications copyright 2026 Qinghua Zhou

"""Expected final loss for Delphi-recipe runs, from the published Delphi IsoFLOP data.

Source: HF dataset ``marin-community/delphi-blog-data`` (config ``delphi-ladder``): 81 IsoFLOP runs
(3e18-3e20 FLOPs), the per-budget parabola fits and optima, and the held-out 1e21-1e23 runs. The
``loss`` column is the Paloma macro loss in nats/token (the ``hparam-scaling`` config names the same
quantity ``paloma_macro_loss``); it matches the scale of the 3e18/157M run (3.711).

Three estimators, all fitted on the 81 IsoFLOP finals only:

* ``chinchilla_total``: ``L = E + A/N^alpha + B/D^beta`` with N = total params (embeddings included,
  the convention of the Delphi run names), fitted by Huber loss on log L (Hoffmann et al. 2022).
* ``chinchilla_nonembed``: the same with N = non-embedding params.
* ``local_width``: for a width that Delphi trained at several token counts (d512: 4.3B, 12.9B, 25.8B),
  ``L = E_h + B_h / D^beta`` with beta fixed from ``chinchilla_total`` and (E_h, B_h) fitted to that
  width's runs. It extrapolates in tokens only, so it is the most direct estimate for a d512 run at a
  smaller budget.

The global fits miss the 3e18/d512 point by -0.07 (observed 3.711, fit 3.78) and the held-out
1e21-1e23 runs by +0.07 to +0.30, so treat them as a bracket, not a forecast. (The blog's 0.2%
forecast is the compute-optimal L*(C) law, which does not price off-optimal sizes such as ours.)

Run: ``python experiments/nano-delphi/scaling_law.py [--data DIR] [--out FILE]``. Without ``--data`` it
downloads the parquet from the Hub (public, no token).
"""

import argparse
import json
import math
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

VOCAB = 128_256
SEQ_LEN = 4096
BLOG_DATA = "https://huggingface.co/datasets/marin-community/delphi-blog-data/resolve/main/delphi-ladder/train.parquet"


def num_layers(hidden: int) -> int:
    """Delphi depth rule: round(H / (64 + 4 log2 H - 9))."""
    return round(hidden / (64 + 4 * math.log2(hidden) - 9))


def param_counts(hidden: int) -> tuple[int, int]:
    """(total, non-embedding) params of the Delphi Qwen3 block at width ``hidden``.

    Untied embedding and LM head, MLP ratio 4, head_dim 128, no GQA, QK-norm and RMSNorm weights.
    Matches the HF model cards (156,508,160 at H=512; 447,244,032 at H=1024).
    """
    layers = num_layers(hidden)
    head_dim = 128
    heads = hidden // head_dim
    attn = 4 * hidden * heads * head_dim
    mlp = 3 * hidden * 4 * hidden
    norms = 2 * hidden + 2 * head_dim
    per_layer = attn + mlp + norms
    non_embed = layers * per_layer + hidden  # final norm
    return non_embed + 2 * VOCAB * hidden, non_embed


def _huber(r: np.ndarray, delta: float = 1e-3) -> np.ndarray:
    a = np.abs(r)
    return np.where(a <= delta, 0.5 * r**2, delta * (a - 0.5 * delta))


def fit_chinchilla(n: np.ndarray, d: np.ndarray, loss: np.ndarray) -> dict[str, float]:
    """Huber fit of log L = LSE(a - alpha log N, b - beta log D, e) over a grid of starts."""
    logn, logd, logl = np.log(n), np.log(d), np.log(loss)

    def objective(p):
        a, b, e, alpha, beta = p
        pred = np.logaddexp(np.logaddexp(a - alpha * logn, b - beta * logd), e)
        return _huber(pred - logl).sum()

    best = None
    for a0 in (0.0, 5.0, 10.0, 20.0):
        for b0 in (0.0, 5.0, 10.0, 20.0):
            for e0 in (-1.0, 0.5, 1.0):
                for alpha0 in (0.1, 0.3, 0.5):
                    for beta0 in (0.1, 0.3, 0.5):
                        res = minimize(objective, [a0, b0, e0, alpha0, beta0], method="L-BFGS-B")
                        if best is None or res.fun < best.fun:
                            best = res
    a, b, e, alpha, beta = best.x
    return {"A": float(np.exp(a)), "B": float(np.exp(b)), "E": float(np.exp(e)), "alpha": float(alpha), "beta": float(beta)}


def chinchilla(params: dict[str, float], n: float, d: float) -> float:
    return params["E"] + params["A"] / n ** params["alpha"] + params["B"] / d ** params["beta"]


def fit_local_width(tokens: np.ndarray, loss: np.ndarray, beta: float) -> tuple[float, float]:
    """Least-squares (E_h, B_h) in L = E_h + B_h * D^-beta for one width."""
    x = tokens**-beta
    design = np.stack([np.ones_like(x), x], axis=1)
    (e, b), *_ = np.linalg.lstsq(design, loss, rcond=None)
    return float(e), float(b)


def flops_per_token(hidden: int) -> float:
    """Training FLOPs per token, C / D, in the convention of the Delphi IsoFLOP budgets.

    Forward FLOPs are 2 x (params minus the input-embedding lookup) plus 4*L*S*H for attention;
    training is 3 x forward. This reproduces ``total_gflops / tokens`` of all 81 Delphi IsoFLOP runs
    to within 0.13%."""
    total, _ = param_counts(hidden)
    layers = num_layers(hidden)
    fwd = 2 * (total - VOCAB * hidden) + 4 * layers * SEQ_LEN * hidden
    return 3 * fwd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=None, help="local delphi-ladder train.parquet")
    parser.add_argument("--out", default=str(Path(__file__).with_name("expected_loss.json")))
    args = parser.parse_args()

    path = args.data
    if path is None:
        path = str(Path(args.out).with_name("delphi-ladder.parquet"))
        urllib.request.urlretrieve(BLOG_DATA, path)
    df = pd.read_parquet(path)
    iso = df[df.kind == "isoflop"].copy()
    held = df[df.kind == "held_out"]

    hidden_of = {}
    for h in range(128, 8192, 128):
        hidden_of[param_counts(h)[0]] = h
    iso["hidden"] = iso["params"].round().astype(int).map(hidden_of)
    assert iso["hidden"].notna().all(), "parameter-count formula does not reproduce the Delphi run sizes"
    iso["non_embed"] = iso["hidden"].map(lambda h: param_counts(int(h))[1])

    fit_total = fit_chinchilla(iso["params"].to_numpy(), iso["tokens"].to_numpy(), iso["loss"].to_numpy())
    fit_ne = fit_chinchilla(iso["non_embed"].to_numpy(), iso["tokens"].to_numpy(), iso["loss"].to_numpy())
    local = {
        int(h): fit_local_width(g["tokens"].to_numpy(), g["loss"].to_numpy(), fit_total["beta"])
        for h, g in iso.groupby("hidden")
        if len(g) >= 2
    }

    def predict(hidden: int, tokens: float) -> dict[str, float]:
        total, ne = param_counts(hidden)
        c = flops_per_token(hidden) * tokens
        return {
            "hidden": hidden,
            "layers": num_layers(hidden),
            "params_total": total,
            "params_non_embed": ne,
            "tokens": tokens,
            "flops": c,
            "chinchilla_total": chinchilla(fit_total, total, tokens),
            "chinchilla_nonembed": chinchilla(fit_ne, ne, tokens),
            "local_width": (local[hidden][0] + local[hidden][1] * tokens ** -fit_total["beta"]) if hidden in local else None,
        }

    resid_total = iso["loss"] - [chinchilla(fit_total, n, d) for n, d in zip(iso["params"], iso["tokens"])]
    resid_ne = iso["loss"] - [chinchilla(fit_ne, n, d) for n, d in zip(iso["non_embed"], iso["tokens"])]
    held_check = [
        {
            "run": r.run_name,
            "loss": r.loss,
            "chinchilla_total": chinchilla(fit_total, r.params, r.tokens),
        }
        for r in held.itertuples()
    ]
    d512 = iso[iso.hidden == 512][["run_name", "tokens", "loss"]].to_dict("records")

    targets = {
        "delphi-3e18-d512 (original smallest)": predict(512, 4_304_928_768),
        "delphi-9e18-d512": predict(512, 12_914_786_304),
    }
    for budget in (1e17, 3e17, 1e18):
        for hidden in (128, 256, 384, 512):
            tokens = budget / flops_per_token(hidden)
            targets[f"nano-{budget:.0e}-d{hidden}"] = predict(hidden, tokens)

    result = {
        "source": BLOG_DATA,
        "metric": "Paloma macro loss (nats/token), final annealed checkpoint",
        "fit_chinchilla_total": fit_total,
        "fit_chinchilla_nonembed": fit_ne,
        "fit_local_width": {str(h): {"E": e, "B": b} for h, (e, b) in local.items()},
        "rmse_isoflop": {"chinchilla_total": float(np.sqrt((resid_total**2).mean())), "chinchilla_nonembed": float(np.sqrt((resid_ne**2).mean()))},
        "held_out_check": held_check,
        "observed_d512": d512,
        "predictions": targets,
    }
    Path(args.out).write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps({k: result[k] for k in ("fit_chinchilla_total", "fit_chinchilla_nonembed", "rmse_isoflop")}, indent=1))
    for row in held_check:
        print(f"held-out {row['run'][-30:]}: obs {row['loss']:.3f} pred {row['chinchilla_total']:.3f}")
    for row in d512:
        print(f"observed d512 {row['tokens']:.3e} tokens: {row['loss']:.3f}")
    print(f"{'target':40s} {'N_tot':>8s} {'N_ne':>7s} {'tokens':>9s} {'FLOPs':>8s} {'L_tot':>6s} {'L_ne':>6s} {'L_loc':>6s}")
    for k, v in targets.items():
        print(
            f"{k:40s} {v['params_total']/1e6:7.1f}M {v['params_non_embed']/1e6:6.1f}M {v['tokens']:9.3e} {v['flops']:8.1e} "
            f"{v['chinchilla_total']:6.3f} {v['chinchilla_nonembed']:6.3f} "
            + (f"{v['local_width']:6.3f}" if v["local_width"] is not None else "     -")
        )


if __name__ == "__main__":
    main()
