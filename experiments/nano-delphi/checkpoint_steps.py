# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Modifications copyright 2026 Qinghua Zhou

"""Steps at which a nano-delphi run keeps permanent checkpoints.

Three sets, merged:

* ``log2``: steps 1, 2, 4, ..., the Pythia early-checkpoint pattern in our own step units.
* ``regular``: every ``every`` steps (default: about 2% of the run).
* ``matched``: steps whose position matches a checkpoint of the reference models used by the arc-res
  weight/activation effective-rank study (Pythia-70M/160M/1B steps 128-143k, OLMo-2-1B steps
  10k/40k/140k). Each reference checkpoint is matched twice: by tokens seen, and by tokens per
  non-embedding parameter (the fairer axis here, since the 128k-token embedding and head are most of a
  small Delphi model). Matches that fall outside the run are dropped.

Levanter's ``CheckpointerConfig.keep`` takes ``{every, until}`` intervals sorted by ``until``; a
checkpoint is permanent at step ``s`` if ``s % every == 0`` under the first interval with
``until >= s``. :func:`keep_policy` encodes an arbitrary sorted step set exactly as one interval per
step (``every = until = s``), followed by an open interval that never fires before the final step.
"""

import json
import math
from dataclasses import asdict, dataclass

PYTHIA_TOKENS_PER_STEP = 1024 * 2048
# Non-embedding parameter counts from the Pythia paper (Biderman et al. 2023, Table 1).
PYTHIA_NON_EMBED = {"pythia-70m": 18_915_328, "pythia-160m": 85_056_000, "pythia-1b": 805_736_448}
# The checkpoints used by the arc-res weight-panel and effective-rank study.
PYTHIA_STEPS = (128, 512, 1000, 3000, 13000, 33000, 73000, 143000)
# OLMo-2-0425-1B: 1.48B params, 16 layers, d=2048, untied 100,352-token embedding and head.
OLMO2_1B_NON_EMBED = 1_484_916_736 - 2 * 100_352 * 2048
OLMO2_1B_TOKENS = {"step10000": 21e9, "step40000": 84e9, "step140000": 294e9, "1T": 1e12, "2T": 2e12, "4T": 4e12}


@dataclass(frozen=True)
class MatchedStep:
    step: int
    reference: str
    reference_checkpoint: str
    reference_tokens: float
    axis: str  # "tokens" or "tokens_per_non_embed_param"
    tokens: int


def matched_steps(tokens_per_step: int, non_embed_params: int, num_train_steps: int) -> list[MatchedStep]:
    refs: list[tuple[str, str, float, int]] = []
    for model, ne in PYTHIA_NON_EMBED.items():
        for s in PYTHIA_STEPS:
            refs.append((model, f"step{s}", s * PYTHIA_TOKENS_PER_STEP, ne))
    for name, tok in OLMO2_1B_TOKENS.items():
        refs.append(("olmo-2-1b", name, tok, OLMO2_1B_NON_EMBED))

    out = []
    for model, ckpt, ref_tokens, ref_ne in refs:
        for axis, tokens in (
            ("tokens", ref_tokens),
            ("tokens_per_non_embed_param", ref_tokens / ref_ne * non_embed_params),
        ):
            step = max(1, round(tokens / tokens_per_step))
            if step <= num_train_steps:
                out.append(MatchedStep(step, model, ckpt, ref_tokens, axis, step * tokens_per_step))
    return sorted(out, key=lambda m: (m.step, m.reference, m.axis))


def checkpoint_steps(
    *, tokens_per_step: int, non_embed_params: int, num_train_steps: int, regular_every: int | None = None
) -> dict[str, list[int]]:
    """The three step sets (each sorted, within [1, num_train_steps])."""
    every = regular_every or max(1, round(num_train_steps / 50))
    log2 = [2**k for k in range(int(math.log2(num_train_steps)) + 1)]
    regular = list(range(every, num_train_steps + 1, every))
    matched = sorted({m.step for m in matched_steps(tokens_per_step, non_embed_params, num_train_steps)})
    return {"log2": log2, "regular": regular, "matched": matched}


def keep_policy(steps: list[int], num_train_steps: int) -> list[dict]:
    """Levanter ``keep`` intervals that make exactly ``steps`` permanent (the final step is always
    saved by the trainer)."""
    ordered = sorted({s for s in steps if 1 <= s <= num_train_steps})
    policy: list[dict] = [{"every": s, "until": s} for s in ordered]
    # Never fires before the end: the first multiple of this interval is past num_train_steps.
    policy.append({"every": num_train_steps + 1, "until": None})
    return policy


def schedule_manifest(*, tokens_per_step: int, non_embed_params: int, num_train_steps: int) -> dict:
    sets = checkpoint_steps(
        tokens_per_step=tokens_per_step, non_embed_params=non_embed_params, num_train_steps=num_train_steps
    )
    all_steps = sorted(set().union(*sets.values()) | {num_train_steps})
    return {
        "tokens_per_step": tokens_per_step,
        "non_embed_params": non_embed_params,
        "num_train_steps": num_train_steps,
        "sets": sets,
        "all_steps": all_steps,
        "matched": [asdict(m) for m in matched_steps(tokens_per_step, non_embed_params, num_train_steps)],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--non-embed-params", type=int, required=True)
    parser.add_argument("--num-train-steps", type=int, required=True)
    args = parser.parse_args()
    manifest = schedule_manifest(
        tokens_per_step=args.batch_size * args.seq_len,
        non_embed_params=args.non_embed_params,
        num_train_steps=args.num_train_steps,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "matched"}, indent=1))
    print(f"{len(manifest['all_steps'])} permanent checkpoints")
