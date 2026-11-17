# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Modifications copyright 2026 Qinghua Zhou

r"""Reproduce the smallest Delphi run, and downscaled variants of it, on a single local GPU.

Copied from ``experiments/references/train_delphi_seeds.py`` (main) with minimal changes. The model
block, optimizer, schedule, precision, z-loss and data mixture are Delphi's. Recipes come from the
Delphi heuristic itself (``experiments/references/completed_adamh.py``: model shape from the width,
then batch size, steps, LR, Adam LR, epsilon and beta2 from the FLOP budget), so a downscaled
variant is the recipe evaluated at a smaller budget and/or width, with no hand-tuned values.

Changes from ``train_delphi_seeds.py`` (see README.md for the reasons):

1. **Current main conventions.** Block shuffle (``DEFAULT_LM_DATA_SHUFFLE``) instead of Delphi's full
   feistel shuffle, and main's float32 cross-entropy logits (#8217, no code change needed).
2. **Data.** Local subsets of the same sources (``mixture.py``), tokenized with a byte-identical,
   ungated copy of the Llama 3.1 tokenizer. Validation is the 16 Paloma sets (no Uncheatable Eval).
3. **Hardware.** One local GPU (no Iris/TPU): the step runs in-process, the mesh is a single data
   axis, and ``--per-device-batch`` sets the microbatch for gradient accumulation (identical math).
   Hooks that need a cluster (W&B online, HF upload, region checks) are off: W&B runs offline and
   metrics are replicated to ``<output>/tracker_metrics.jsonl``.
4. **Checkpoints.** Permanent checkpoints at log2 steps, every ~2% of the run, and at steps matching
   the Pythia/OLMo checkpoints of the arc-res effective-rank study (``checkpoint_steps.py``).
5. **Downscaling.** Widths below Delphi's smallest (512) are allowed (multiples of 128; the depth
   rule and head size stay Delphi's), and the FLOP budget is free.

Variants are named ``{budget}-d{width}`` (e.g. ``3e18-d512`` is the original smallest Delphi run,
``isoflop-3e+18-d512-L6-B32-adamh_scaling_v6``; ``1e18-d512`` is the same model at a third of the
compute). Usage, from the repository root with ``MARIN_PREFIX`` set to a local directory::

    python experiments/nano-delphi/train_nano_delphi.py --variant 1e18-d512 --seed 0 --print-only
    python experiments/nano-delphi/train_nano_delphi.py --variant 1e18-d512 --seed 0

``run_local.template.sh`` wraps this with the GPU lock, environment and tmux session.
"""

import argparse
import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jmp
from fray.types import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.callbacks.watch import WatchConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.adamh import AdamHConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, run_levanter_train_lm

from checkpoint_steps import keep_policy, schedule_manifest
from experiments.references.completed_adamh import completed_adamh_heuristic
from mixture import TOKENIZER, train_weights, validation_datasets

VERSION = "2026.11.15"
EXPERIMENT_NAME = "nano-delphi"
SEQUENCE_LENGTH = 4096
# Same vocabulary (128,256) and special tokens as the heuristic's marin-community/marin-tokenizer.
HEURISTIC = dataclasses.replace(completed_adamh_heuristic, tokenizer=TOKENIZER)


@dataclass(frozen=True)
class ModelSpec:
    budget: float
    hidden_dim: int
    num_layers: int
    num_heads: int
    intermediate_dim: int
    batch_size: int
    num_train_steps: int
    learning_rate: float
    adam_lr: float
    beta2: float
    epsilon: float
    params: int
    non_embed_params: int
    tokens: int
    tags: list[str]


def model_spec(budget: float, hidden: int) -> ModelSpec:
    """Evaluate the Delphi heuristic at (budget, width), as ``candidates_for_budget`` does per width."""
    model = HEURISTIC._build_model_config(hidden, SEQUENCE_LENGTH)
    fpt = model.flops_per_token(HEURISTIC.vocab_size, SEQUENCE_LENGTH)
    tokens = budget / (3 * fpt)
    candidate = HEURISTIC._build_candidate_config(model, tokens, budget, SEQUENCE_LENGTH)
    if candidate is None and tokens < HEURISTIC.min_batch_size * 2**16 * SEQUENCE_LENGTH:
        # Downscaling extension: the heuristic targets 2^16 steps and rejects batches below its
        # minimum of 8, so budgets under ~2.1B tokens at batch 8 have no candidate. Keep the minimum
        # batch and shorten the run instead; LR, Adam LR, eps and beta2 follow the same formulas.
        batch = HEURISTIC.min_batch_size
        steps = round(tokens / (batch * SEQUENCE_LENGTH))
        candidate = dataclasses.replace(
            HEURISTIC._build_candidate_config(model, batch * 2**16 * SEQUENCE_LENGTH, budget, SEQUENCE_LENGTH),
            batch_size=batch,
            train_steps=steps,
            tokens=batch * steps * SEQUENCE_LENGTH,
            optimizer_config=HEURISTIC.build_optimizer_config(batch, tokens),
        )
    if candidate is None:
        raise ValueError(f"The Delphi heuristic has no valid batch size for budget={budget:g}, hidden={hidden}")
    opt = candidate.optimizer_config
    params = model.total_trainable_params(HEURISTIC.vocab_size)
    non_embed = params - 2 * HEURISTIC.vocab_size * hidden
    return ModelSpec(
        budget=budget,
        hidden_dim=hidden,
        num_layers=model.num_layers,
        num_heads=model.num_heads,
        intermediate_dim=model.intermediate_dim,
        batch_size=candidate.batch_size,
        num_train_steps=candidate.train_steps,
        learning_rate=opt.learning_rate,
        adam_lr=opt.adam_lr,
        beta2=opt.beta2,
        epsilon=opt.epsilon,
        params=params,
        non_embed_params=non_embed,
        tokens=candidate.tokens,
        tags=[
            f"FLOPs={budget:.1e}",
            f"N={params:.1e}",
            f"B={candidate.batch_size}",
            f"steps={candidate.train_steps}",
            f"tokens={candidate.tokens:.1e}",
            "optimizer=completed-adamh",
        ],
    )


def delphi_model(spec: ModelSpec) -> Qwen3Config:
    """The Delphi Qwen3 block at the spec's width (verbatim from train_delphi_seeds.py)."""
    return Qwen3Config(
        hidden_dim=spec.hidden_dim,
        intermediate_dim=spec.intermediate_dim,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        num_kv_heads=spec.num_heads,
        head_dim=None,
        max_seq_len=SEQUENCE_LENGTH,
        activation_function=ActivationFunctionEnum.silu,
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
        hybrid_norm=False,
        input_embedding_norm=False,
        use_qk_norm=False,  # Qwen3 enables QK norm in attention_config regardless of this inherited field.
        use_bias=False,
        use_layer_norm_weight=True,
        upcast_attn=False,
        scan_layers=True,
        gradient_checkpointing=True,
        attn_backend=None,
        flash_attention_block_size=None,
        reference_checkpoint="NousResearch/Llama-2-7b-hf",
        tokenizer=None,
        use_sliding_window=False,
        sliding_window=4096,
        rope=Llama3RotaryEmbeddingsConfig(
            theta=500000,
            factor=8,
            low_freq_factor=1,
            high_freq_factor=4,
            original_max_position_embeddings=8192,
        ),
    )


def parse_variant(variant: str) -> tuple[float, int]:
    budget, width = variant.split("-d")
    return float(budget), int(width)


def _training_step(
    variant: str, seed: int, per_device_batch: int, steps_per_eval: int, max_eval_batches: int | None, keep_schedule: bool,
    only_components: tuple[str, ...] | None = None,
    eval_batch: int = -1,
) -> ArtifactStep[LevanterCheckpoint]:
    spec = model_spec(*parse_variant(variant))
    name = f"{EXPERIMENT_NAME}-{variant}-seed{seed}"
    run_id = f"{name}-{VERSION}"
    resources = ResourceConfig.with_gpu("local", count=1)  # label only; the run is in-process
    schedule = schedule_manifest(
        tokens_per_step=spec.batch_size * SEQUENCE_LENGTH,
        non_embed_params=spec.non_embed_params,
        num_train_steps=spec.num_train_steps,
    )
    train = train_weights()
    if only_components:  # smoke tests only: train on a subset of the mixture
        train = {h: w for h, w in train.items() if h.name.removeprefix("nano-delphi/") in only_components}
    validation = validation_datasets()

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        model = delphi_model(spec)
        optimizer = AdamHConfig(
            learning_rate=spec.learning_rate,
            adam_lr=spec.adam_lr,
            beta1=0.9,
            beta2=spec.beta2,
            epsilon=spec.epsilon,
            max_grad_norm=0.1,
            weight_decay=0.1,
            lr_schedule="linear",
            warmup=0.1,
            decay=0.2,
            min_lr_ratio=0.0,
            rewarmup=0.0,
        )
        trainer = TrainerConfig(
            seed=seed,
            id=run_id,
            tracker=WandbConfig(
                project="nano-delphi",
                id=run_id,
                name=run_id,
                mode="offline",
                replicate_path=ctx.output_path,
                resume="allow",
                tags=list(spec.tags),
                save_code=False,
            ),
            mp=jmp.get_policy("p=f32,c=bf16,o=bf16"),
            train_batch_size=spec.batch_size,
            per_device_parallelism=per_device_batch,
            per_device_eval_parallelism=eval_batch,  # eval-only; does not affect training
            num_train_steps=spec.num_train_steps,
            allow_nondivisible_batch_size=True,
            mesh=MeshConfig(
                axes={"data": -1, "model": 1, "replica": 1},
                batch_axis_name="batch",
                compute_mapping={
                    "token": ["replica_dcn", "replica", "data"],
                    "token_repeat": ["replica_dcn", "replica", "data"],
                },
                param_mapping={"embed": "data"},
            ),
            use_explicit_mesh_axes=False,
            jax_config={"jax_softmax_custom_jvp": True, "jax_threefry_partitionable": True},
            require_accelerator=True,
            steps_per_eval=steps_per_eval,
            max_eval_batches=max_eval_batches,
            watch=WatchConfig(
                watch_targets=["grads", "params"],
                interval=10,
                include_norms=True,
                include_per_parameter_norms=True,
                include_histograms=False,
                split_scan_layers=True,
            ),
            profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=25, perfetto_link=False),
            checkpointer=CheckpointerConfig(
                base_path=ctx.output_path,
                keep=keep_policy(schedule["all_steps"], spec.num_train_steps) if keep_schedule else None,
                save_interval=timedelta(minutes=10),
                append_run_id_to_base_path=False,
                delete_old_temp_checkpoints=True,
                keep_last_temporary_checkpoints=1,
            ),
            initialize_from=None,
            load_checkpoint=None,
            load_checkpoint_path=None,
            allow_partial_checkpoint=False,
        )
        train_config = TrainLmConfig(
            data=mixture(ctx, train, validation=validation),
            trainer=trainer,
            model=model,
            optimizer=optimizer,
            train_seq_len=SEQUENCE_LENGTH,
            data_seed=None,
            z_loss_weight=1e-7,
            initialize_from_hf=False,
            initialize_from_checkpoint_path=None,
            initialize_model_from_checkpoint_path=None,
            use_hf_model_config=False,
            pad_tokenizer_to_match_model=False,
            hf_save_steps=spec.num_train_steps,
            hf_upload=None,
            merged_hf_upload=None,
            eval_harness=None,
            eval_harness_steps=10000,
            log_entropy=False,
        )
        return TrainLmOnPodConfig(
            train_config=train_config,
            resources=resources,
            output_path=ctx.output_path,
            auto_build_caches=False,
        )

    return ArtifactStep(
        name=name,
        version=VERSION,
        artifact_type=LevanterCheckpoint,
        run=run_levanter_train_lm,  # in-process on the local GPU (no Fray dispatch)
        build_config=build_config,
        deps=(*train, *validation),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", required=True, help="{budget}-d{width}, e.g. 3e18-d512 or 1e18-d512")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-device-batch", type=int, default=-1, help="microbatch size (-1: whole batch)")
    parser.add_argument("--steps-per-eval", type=int, default=1000)
    parser.add_argument("--max-eval-batches", type=int, default=None, help="cap Paloma eval batches (smoke tests)")
    parser.add_argument("--final-only", action="store_true", help="keep only the final checkpoint (smoke tests)")
    parser.add_argument("--only-components", default=None, help="comma list of mixture components (smoke tests)")
    parser.add_argument("--eval-batch", type=int, default=-1, help="eval microbatch size (-1: the training batch)")
    parser.add_argument("--print-only", action="store_true", help="print the recipe and checkpoint schedule")
    args = parser.parse_args()

    spec = model_spec(*parse_variant(args.variant))
    schedule = schedule_manifest(
        tokens_per_step=spec.batch_size * SEQUENCE_LENGTH,
        non_embed_params=spec.non_embed_params,
        num_train_steps=spec.num_train_steps,
    )
    print(json.dumps(dataclasses.asdict(spec), indent=1))
    print(f"{len(schedule['all_steps'])} permanent checkpoints: {schedule['all_steps']}")
    if args.print_only:
        return
    step = _training_step(
        args.variant, args.seed, args.per_device_batch, args.steps_per_eval, args.max_eval_batches, not args.final_only,
        tuple(args.only_components.split(",")) if args.only_components else None,
        args.eval_batch,
    )
    StepRunner().run([step.lower()], max_concurrent=1)


if __name__ == "__main__":
    main()
