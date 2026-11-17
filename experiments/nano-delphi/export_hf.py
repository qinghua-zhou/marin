# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Modifications copyright 2026 Qinghua Zhou

"""Export a nano-delphi run's permanent Levanter checkpoints to Hugging Face format (for arc-res/idtraj).

    python experiments/nano-delphi/export_hf.py --variant 1e18-d512 --run-dir RUN_OUTPUT --out-dir HF_ROOT [--steps 1,2,...]

Writes ``HF_ROOT/step-{N}/`` (``config.json``, ``model.safetensors``, tokenizer) for every
``RUN_OUTPUT/step-{N}`` checkpoint (or the listed steps), skipping exports that already exist. Uses
``levanter.main.export_lm_to_hf`` on CPU, so it can run beside training without touching the GPU.

Levanter never saves step 0, so ``--seed`` also writes ``HF_ROOT/step-0`` by rebuilding the initial model
exactly as ``levanter.main.train_lm`` does: ``model.build(Vocab, key=split(PRNGKey(seed), 4)[2])`` under
the run's ``jax_threefry_partitionable`` setting. Check it against step 1: AdamH keeps every projection
matrix on its initial Frobenius-norm sphere, so their norms must match to float precision.
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from levanter.main.export_lm_to_hf import ConvertLmConfig
from levanter.main.export_lm_to_hf import main as export_main
from levanter.trainer import TrainerConfig

from mixture import TOKENIZER
from train_nano_delphi import delphi_model, model_spec, parse_variant


def fix_architecture(target: Path) -> None:
    """Levanter's Qwen3 export writes ``architectures: [LlamaForCausalLM]`` next to ``model_type: qwen3``;
    the weights include q_norm/k_norm, so loaders keyed on ``architectures`` (idtraj) need Qwen3ForCausalLM,
    as in Delphi's own HF releases."""
    import json

    path = target / "config.json"
    config = json.loads(path.read_text())
    if config.get("model_type") == "qwen3" and config.get("architectures") != ["Qwen3ForCausalLM"]:
        config["architectures"] = ["Qwen3ForCausalLM"]
        path.write_text(json.dumps(config, indent=2) + "\n")


def export_init(model_config, seed: int, target: Path) -> None:
    import jax
    import jax.random as jrandom
    from haliax import Axis
    from jax.sharding import AxisType
    from levanter.compat.hf_checkpoints import load_tokenizer
    from levanter.utils.jax_utils import local_cpu_mesh

    # Same JAX settings as the training run (train_nano_delphi.py TrainerConfig.jax_config).
    jax.config.update("jax_threefry_partitionable", True)
    jax.config.update("jax_softmax_custom_jvp", True)
    tokenizer = load_tokenizer(TOKENIZER)
    vocab = Axis("vocab", len(tokenizer))
    model_key = jrandom.split(jrandom.PRNGKey(seed), 4)[2]  # data, loader, model, training
    axis_type = AxisType.Explicit if model_config.requires_explicit_mesh_axes else AxisType.Auto
    with local_cpu_mesh(axis_type):  # same CPU mesh export_lm_to_hf uses
        model = model_config.build(vocab, key=model_key)
        converter = model_config.hf_checkpoint_converter().replaced(tokenizer=tokenizer)
        converter.save_pretrained(model, str(target), save_tokenizer=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-dir", required=True, help="the training step's output path (holds step-N/)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", default=None, help="comma-separated steps (default: all checkpoints)")
    parser.add_argument("--seed", type=int, default=None, help="also export the step-0 initial model of this seed")
    args = parser.parse_args()

    model = delphi_model(model_spec(*parse_variant(args.variant)))
    run_dir, out_dir = Path(args.run_dir), Path(args.out_dir)
    available = {int(m.group(1)): p for p in run_dir.glob("step-*") if (m := re.fullmatch(r"step-(\d+)", p.name))}
    steps = sorted(available) if args.steps is None else [int(s) for s in args.steps.split(",")]
    if args.seed is not None and not (out_dir / "step-0" / "config.json").exists():
        export_init(model, args.seed, out_dir / "step-0")
        fix_architecture(out_dir / "step-0")
        print(f"exported initial model (seed {args.seed}) -> {out_dir / 'step-0'}")
    for step in steps:
        target = out_dir / f"step-{step}"
        if (target / "config.json").exists():
            print(f"skip step {step}: exported")
            continue
        if step not in available:
            print(f"skip step {step}: no checkpoint")
            continue
        export_main(
            ConvertLmConfig(
                trainer=TrainerConfig(),
                checkpoint_path=str(available[step]),
                output_dir=str(target),
                model=model,
                tokenizer=TOKENIZER,
                use_cpu=True,
            )
        )
        fix_architecture(target)
        print(f"exported step {step} -> {target}")


if __name__ == "__main__":
    main()
