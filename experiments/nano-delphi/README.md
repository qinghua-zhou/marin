# nano-delphi: reproduce the smallest Delphi run and scale it down, on one local GPU

Started 2026-11-15 (UTC). Branch `nano-delphi`, cut from `main` at `6e09d9043e`.

## Goal

1. Reproduce the smallest Delphi model (`isoflop-3e+18-d512-L6-B32-adamh_scaling_v6`, HF
   `marin-community/delphi-3e18-157Mparams-4.3Btokens`) with current-main conventions, at the same
   architecture, parameter count and token count.
2. Run smaller variants of the same recipe on single local GPUs and check that (a) the final loss
   lands where the Delphi scaling law puts it and (b) the training trajectory shows the internal and
   downstream behaviour seen in small Pythia/OLMo checkpoints (arc-res weight/activation
   effective-rank study). Every run keeps permanent checkpoints for that comparison.

## What the original suite used

### Data

Marin's tokenized caches (Llama 3.1 tokenizer, private `gs://marin-us-central2/tokenized/...`), mixed
with these weights, in this order (`experiments/references/train_delphi_seeds.py`, `DATASETS`):

| component | weight | share | raw source |
|---|---:|---:|---|
| nemotron_cc/hq_actual | 0.91351 | 8.3% | Nemotron-CC `quality=high/kind=actual` |
| nemotron_cc/hq_synth | 2.72 | 24.7% | Nemotron-CC `quality=high/kind=synthetic` (5 sub-kinds) |
| nemotron_cc/medium_high | 0.82471 | 7.5% | Nemotron-CC `quality=medium-high` |
| nemotron_cc/medium | 3.38 | 30.7% | Nemotron-CC `quality=medium` |
| nemotron_cc/medium_low | 1.54 | 14.0% | Nemotron-CC `quality=medium-low` |
| nemotron_cc/low_actual | 0.70123 | 6.4% | Nemotron-CC `quality=low/kind=actual` |
| nemotron_cc/low_synth | 0.62771 | 5.7% | Nemotron-CC `quality=low/kind=synthetic` |
| starcoderdata | 0.25 | 2.3% | `bigcode/starcoderdata` (text key `content`) |
| proofpile_2 | 0.055 | 0.5% | `EleutherAI/proof-pile-2` @ `901a927` |

Validation (weight 0): 16 Paloma sets (`allenai/paloma` @ `65cd6fc`) and 7 Uncheatable Eval sets
(snapshot `latest-364401`). Data settings: feistel permutation with full shuffle, mixture block size
2048, `stop_strategy=restart`, EOS enforced, attention blocked across documents.

### Code on current `main`

Delphi never had its own branch; its scripts were deleted from `main` in June 2026 (the blog commit
`78ae89a5` still has them). Their current-main equivalents:

| purpose | file on main |
|---|---|
| recipe (compute budget to model, batch, steps, LR, Adam LR, eps, beta2) | `experiments/references/completed_adamh.py` (`CompletedAdamHHeuristic`; same math as `scaling_law_sweeps/completed_adamh.py` at `78ae89a5`) |
| 3e18-3e20 IsoFLOP runs and seed repeats, data mixture, model block | `experiments/references/train_delphi_seeds.py` |
| 1e21-1e23 compute-optimal runs and scaling-law fit | `experiments/references/reference_scaling_suite.py`, `lib/marin/src/marin/scaling_laws/` |
| dataset catalogs | `experiments/datasets/{nemotron,starcoder,proofpile,paloma,uncheatable}.py` |
| optimizer | `lib/levanter/src/levanter/optim/adamh.py` (AdamH) |
| model | `lib/levanter/src/levanter/models/qwen.py` (`Qwen3Config`) |

`train_delphi_seeds.py` does not list the 157M run (its smallest is 3e18/d1024); the 157M run is the
heuristic at 3e18 FLOPs and width 512. Evaluating `completed_adamh.py` at that point reproduces the
released run exactly: d512, 6 layers, 4 heads, batch 32 x 4096, 32,844 steps (4,304,928,768 tokens;
the model card's 32,843 is the last 0-indexed step), LR 3.785e-3, Adam LR 3.535e-4, eps 3.433e-8,
beta2 0.9999. The fuller audit of main vs the blog commit is in the local note
`output/delphi_reproducibility.md` (not committed).

## What this folder changes

`train_nano_delphi.py` is `train_delphi_seeds.py` with minimal edits:

| change | why |
|---|---|
| recipe values computed by `completed_adamh.py` at (budget, width) instead of hardcoded | the same code path serves the 157M reproduction and every downscaled variant |
| block shuffle (`DEFAULT_LM_DATA_SHUFFLE`) instead of `shuffle=True` | current-main convention (requested) |
| float32 cross-entropy logits | current-main behaviour (#8217); no code change |
| local subsets of the same raw sources, tokenized here (`prepare_data.py`, `mixture.py`) | the original caches are private; a run needs at most ~6B tokens |
| tokenizer `NousResearch/Meta-Llama-3.1-8B` | byte-identical tokenizer files to `meta-llama/Llama-3.1-8B`, not gated |
| validation: 16 Paloma sets only | Delphi's Uncheatable snapshot is not public |
| one local GPU: in-process step, `per_device_parallelism` microbatching, W&B offline (metrics replicated to `tracker_metrics.jsonl`) | gpuhub/aether are not a cluster |
| permanent checkpoints at log2 steps, every ~2% of the run, and Pythia/OLMo-matched steps (`checkpoint_steps.py`) | trajectory analysis; Delphi kept only the final checkpoint |
| widths below 512 allowed | downscaling; depth rule, head size, MLP ratio stay Delphi's |

The data subset is a seeded draw of whole shards (Nemotron-CC: shuffled within each split across all
crawls; ProofPile-2: one train shard each of arxiv, open-web-math, algebraic-stack; StarCoder: 12
parquet files across languages, 25% of rows each), sized for 5.6B training tokens with a 1.3x
margin so no component repeats in a 4.3B-token run (`data_manifest.json`). Smaller runs read the same
caches.

## Expected loss (Delphi scaling law)

`scaling_law.py` fits the 81 Delphi IsoFLOP finals from HF `marin-community/delphi-blog-data`
(`delphi-ladder`; `loss` is the Paloma macro loss). Output: `expected_loss.json`.

| run | params (non-emb) | tokens | Chinchilla, N=total | Chinchilla, N=non-emb | same-width fit |
|---|---|---:|---:|---:|---:|
| 3e18-d512 (original; measured **3.711**) | 156.5M (25.2M) | 4.30B | 3.780 | 3.790 | 3.712 |
| 1e18-d512 | 156.5M (25.2M) | 1.43B | 4.025 | 4.032 | 3.873 |
| 1e18-d384 | 107.9M (9.4M) | 2.34B | 4.131 | 4.248 | - |
| 1e18-d256 | 68.8M (3.1M) | 3.94B | 4.394 | 4.673 | - |
| 3e17-d512 | 156.5M (25.2M) | 0.43B | 4.467 | 4.456 | 4.164 |

The global Chinchilla fits (RMSE 0.021 over the 81 points) miss the 3e18/d512 point by +0.07 and
under-predict the held-out 1e21-1e23 runs by 0.07-0.30, so they bracket rather than forecast. The
same-width fit (d512's three Delphi runs, beta from the global fit) only extrapolates in tokens and
is the primary target for d512 variants. Below d512 the two global fits disagree by 0.1-0.3: the run
itself tells which parameter count governs these embedding-dominated models.

## Runs

See `RUNS.md` (status, hardware, timings, results).

## Reproducing

```bash
# 1. plan and stage raw data (gated sources on a machine with an HF token that can read them)
python experiments/nano-delphi/prepare_data.py plan --train-tokens 5.6e9 --out experiments/nano-delphi/data_manifest.json
python experiments/nano-delphi/prepare_data.py download --manifest experiments/nano-delphi/data_manifest.json --prefix $MARIN_PREFIX --sources nemotron,proofpile
python experiments/nano-delphi/prepare_data.py download --manifest experiments/nano-delphi/data_manifest.json --prefix STAGING --sources paloma,starcoder
# 2. tokenize (the training step would also do this as a dependency)
python experiments/nano-delphi/mixture.py --run
# 3. inspect a recipe and its checkpoint schedule, then launch
python experiments/nano-delphi/train_nano_delphi.py --variant 1e18-d512 --print-only
experiments/nano-delphi/run_local.template.sh 1e18-d512 0 0 -1
```
