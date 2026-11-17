# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Modifications copyright 2026 Qinghua Zhou

"""Stage raw subsets of the Delphi training mixture and its Paloma validation sets under a local prefix.

Delphi trained on Marin's tokenized caches of the full corpora (private GCS). A nano-delphi run needs
only a few billion tokens, so this script downloads a fixed, seeded subset of the same public raw
sources and lays them out the way Marin's raw prefix does. ``mixture.py`` then tokenizes them with
``marin.experiment.data.tokenized``.

    plan      choose the shards (deterministic, seed 0) for a target number of training tokens
    download  fetch the chosen shards into ``<prefix>/raw/...``
    split     split each staged Nemotron-CC shard into smaller ``.jsonl.zstd`` pieces in place, so local
              tokenization runs one worker per piece (a 375 MB shard takes ~2 h on one worker)

Sources (the same ones as ``experiments/references/train_delphi_seeds.py``):

* Nemotron-CC, 7 quality splits, from ``https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC``.
  Shards are shuffled within each split (all crawls and ``kind2`` sub-kinds) and taken in order, so a
  larger plan is a superset of a smaller one.
* ProofPile-2 at revision ``901a927`` (the revision Marin downloaded): train shards of arxiv,
  open-web-math and algebraic-stack.
* StarCoderData (gated): parquet files sampled across languages in proportion to their file counts,
  each subsampled by rows.
* Paloma at revision ``65cd6fc`` (gated): the 16 ``val`` sets, laid out at
  ``raw/paloma-fc6827/65cd6fc/<subset>/val/`` so ``experiments.datasets.paloma`` works unchanged.

Gated sources need a Hugging Face token with access (``HF_TOKEN`` or ``~/.cache/huggingface/token``).
Download them on a trusted machine and copy the staged ``raw/`` tree to the training host.
"""

import argparse
import concurrent.futures
import gzip
import json
import math
import os
import random
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

NEMOTRON_BASE = "https://data.commoncrawl.org/"
NEMOTRON_INDEX = "https://data.commoncrawl.org/contrib/Nemotron/Nemotron-CC/data-jsonl.paths.gz"
NEMOTRON_SPLITS = {  # split -> path fragment (as in experiments/datasets/nemotron.py)
    "hq_actual": "quality=high/kind=actual/",
    "hq_synth": "quality=high/kind=synthetic/",
    "medium_high": "quality=medium-high/",
    "medium": "quality=medium/",
    "medium_low": "quality=medium-low/",
    "low_actual": "quality=low/kind=actual/",
    "low_synth": "quality=low/kind=synthetic/",
}
# Delphi train weights (train_delphi_seeds.py DATASETS).
WEIGHTS = {
    "nemotron_cc/hq_actual": 0.91351,
    "nemotron_cc/hq_synth": 2.72,
    "nemotron_cc/medium_high": 0.82471,
    "nemotron_cc/medium": 3.38,
    "nemotron_cc/medium_low": 1.54,
    "nemotron_cc/low_actual": 0.70123,
    "nemotron_cc/low_synth": 0.62771,
    "starcoderdata": 0.25,
    "proofpile_2": 0.055,
}
# Measured 2026-11-15: a quality=medium shard (379.5 MB zstd) holds 198.5M Llama-3 tokens.
NEMOTRON_TOKENS_PER_BYTE = 0.52
NEMOTRON_SHARD_BYTES = 375e6

PROOFPILE_REPO = "EleutherAI/proof-pile-2"
PROOFPILE_REVISION = "901a9273a770e9d4138c5ddd91802f9c5c6cdc4b"
PROOFPILE_SUBSETS = ("arxiv", "open-web-math", "algebraic-stack")
STARCODER_REPO = "bigcode/starcoderdata"
PALOMA_REPO = "allenai/paloma"
PALOMA_REVISION = "65cd6fc59dba021b21db414fa5e8d7765ffbe5e6"

# Layout under <prefix>/raw (Marin's raw prefix convention).
NEMOTRON_RAW = "raw/nano-delphi/nemotron-cc"  # + /contrib/Nemotron/Nemotron-CC/data-jsonl/...
PROOFPILE_RAW = "raw/nano-delphi/proof-pile-2"
STARCODER_RAW = "raw/nano-delphi/starcoderdata"
PALOMA_RAW = "raw/paloma-fc6827/65cd6fc"


def _hf_token() -> str | None:
    if token := os.environ.get("HF_TOKEN"):
        return token
    path = Path.home() / ".cache" / "huggingface" / "token"
    return path.read_text().strip() if path.exists() else None


def _get(url: str, *, auth: bool = False) -> bytes:
    request = urllib.request.Request(url)
    if auth and (token := _hf_token()):
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _hf_files(repo: str, revision: str, *, auth: bool) -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=true"
    files, cursor = [], url
    while cursor:
        request = urllib.request.Request(cursor)
        if auth and (token := _hf_token()):
            request.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(request, timeout=120) as response:
            files += [f for f in json.loads(response.read()) if f["type"] == "file"]
            link = response.headers.get("Link", "")
        cursor = link.split(";")[0].strip("<>") if 'rel="next"' in link else None
    return files


def plan(train_tokens: float, margin: float, seed: int) -> dict:
    total_weight = sum(WEIGHTS.values())
    need = {name: train_tokens * margin * w / total_weight for name, w in WEIGHTS.items()}

    index = gzip.decompress(_get(NEMOTRON_INDEX)).decode().split()
    nemotron = {}
    for split, fragment in NEMOTRON_SPLITS.items():
        paths = sorted(p for p in index if f"/data-jsonl/{fragment}" in p)
        random.Random(f"{seed}-{split}").shuffle(paths)
        count = math.ceil(need[f"nemotron_cc/{split}"] / (NEMOTRON_TOKENS_PER_BYTE * NEMOTRON_SHARD_BYTES))
        nemotron[split] = {"available": len(paths), "est_tokens_needed": need[f"nemotron_cc/{split}"], "paths": paths[:count]}

    # ProofPile-2: one train shard per subset already exceeds the need (~0.5% of the mix).
    pp_files = _hf_files(PROOFPILE_REPO, PROOFPILE_REVISION, auth=False)
    proofpile = []
    for subset in PROOFPILE_SUBSETS:
        shards = sorted(f["path"] for f in pp_files if f["path"].startswith(f"{subset}/train/"))
        proofpile.append(random.Random(f"{seed}-{subset}").choice(shards))

    return {
        "train_tokens": train_tokens,
        "margin": margin,
        "seed": seed,
        "weights": WEIGHTS,
        "tokens_needed": need,
        "nemotron": nemotron,
        "proofpile_2": {"revision": PROOFPILE_REVISION, "paths": proofpile},
        # StarCoder and Paloma are chosen at download time (their listings need gated access).
        "starcoderdata": {"est_tokens_needed": need["starcoderdata"], "num_files": 12, "row_fraction": 0.25},
        "paloma": {"revision": PALOMA_REVISION},
    }


def _curl(url: str, dest: Path, *, auth: bool = False) -> str:
    if dest.exists() and dest.stat().st_size > 0:
        return f"exists {dest}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    cmd = ["curl", "-sSfL", "--retry", "5", "--retry-delay", "5", "-o", str(tmp), url]
    config = None
    if auth and (token := _hf_token()):
        # Pass the header through a config on stdin so the token never appears in argv.
        cmd = ["curl", "-K", "-", *cmd[1:]]
        config = f'header = "Authorization: Bearer {token}"\n'
    for attempt in range(8):
        if subprocess.run(cmd, input=config, text=True).returncode == 0:
            break
        time.sleep(30 * (attempt + 1))
    else:
        raise RuntimeError(f"failed to download {url}")
    tmp.rename(dest)
    return f"fetched {dest} ({dest.stat().st_size / 1e6:.0f} MB)"


def download_nemotron(manifest: dict, prefix: Path, jobs: int) -> None:
    root = prefix / NEMOTRON_RAW
    tasks = [p for split in manifest["nemotron"].values() for p in split["paths"]]
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        for msg in pool.map(lambda p: _curl(NEMOTRON_BASE + p, root / p), tasks):
            print(msg, flush=True)


def download_proofpile(manifest: dict, prefix: Path, jobs: int) -> None:
    rev = manifest["proofpile_2"]["revision"]
    root = prefix / PROOFPILE_RAW
    base = f"https://huggingface.co/datasets/{PROOFPILE_REPO}/resolve/{rev}/"
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        for msg in pool.map(lambda p: _curl(base + p, root / p), manifest["proofpile_2"]["paths"]):
            print(msg, flush=True)


def download_paloma(manifest: dict, prefix: Path, jobs: int) -> None:
    rev = manifest["paloma"]["revision"]
    files = [f["path"] for f in _hf_files(PALOMA_REPO, rev, auth=True) if "/val/" in f["path"]]
    root = prefix / PALOMA_RAW
    base = f"https://huggingface.co/datasets/{PALOMA_REPO}/resolve/{rev}/"
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        for msg in pool.map(lambda p: _curl(base + urllib.parse.quote(p), root / p, auth=True), files):
            print(msg, flush=True)


def download_starcoder(manifest: dict, prefix: Path, jobs: int, seed: int) -> None:
    """Sample parquet files across languages (probability proportional to each language's file count),
    then keep a seeded row fraction of each file."""
    import pyarrow.parquet as pq

    spec = manifest["starcoderdata"]
    files = sorted(f["path"] for f in _hf_files(STARCODER_REPO, "main", auth=True) if f["path"].endswith(".parquet"))
    chosen = random.Random(f"{seed}-starcoder").sample(files, spec["num_files"])
    staging = prefix / STARCODER_RAW / ".full"
    base = f"https://huggingface.co/datasets/{STARCODER_REPO}/resolve/main/"
    with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
        for msg in pool.map(lambda p: _curl(base + urllib.parse.quote(p), staging / p, auth=True), chosen):
            print(msg, flush=True)
    for path in chosen:
        table = pq.read_table(staging / path)
        rng = random.Random(f"{seed}-{path}")
        keep = sorted(rng.sample(range(table.num_rows), max(1, int(table.num_rows * spec["row_fraction"]))))
        out = prefix / STARCODER_RAW / path
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.take(keep).select(["content"]), out)
        print(f"subsampled {path}: {len(keep)}/{table.num_rows} rows", flush=True)
        (staging / path).unlink()
    shutil.rmtree(staging)


def _split_one(path: Path, pieces: int) -> str:
    import zstandard

    # No marker files next to the data: anything matching the split globs (``*.jsonl.*``) is read as
    # input. The original shard is deleted after its pieces are written, so a rerun skips it.
    with path.open("rb") as raw:
        lines = zstandard.ZstdDecompressor().stream_reader(raw, read_across_frames=True).read().splitlines(keepends=True)
    stem = path.name.removesuffix(".jsonl.zstd")
    per = math.ceil(len(lines) / pieces)
    for i in range(pieces):
        out = path.with_name(f"{stem}-split{i:02d}.jsonl.zstd")
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_bytes(zstandard.ZstdCompressor(level=3).compress(b"".join(lines[i * per : (i + 1) * per])))
        tmp.rename(out)
    path.unlink()  # the pieces replace the shard (and still match the split globs)
    return f"split {path.name}: {len(lines)} docs"


def split_nemotron(prefix: Path, pieces: int, jobs: int) -> None:
    shards = sorted(p for p in (prefix / NEMOTRON_RAW).rglob("*.jsonl.zstd") if "-split" not in p.name)
    with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
        for msg in pool.map(_split_one, shards, [pieces] * len(shards)):
            print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--train-tokens", type=float, required=True)
    p_plan.add_argument("--margin", type=float, default=1.3, help="over-provisioning so no component repeats")
    p_plan.add_argument("--seed", type=int, default=0)
    p_plan.add_argument("--out", required=True)
    p_dl = sub.add_parser("download")
    p_dl.add_argument("--manifest", required=True)
    p_dl.add_argument("--prefix", required=True)
    p_dl.add_argument("--sources", default="nemotron,proofpile", help="comma list: nemotron,proofpile,starcoder,paloma")
    p_dl.add_argument("--jobs", type=int, default=8)
    p_split = sub.add_parser("split")
    p_split.add_argument("--prefix", required=True)
    p_split.add_argument("--pieces", type=int, default=8)
    p_split.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()

    if args.cmd == "split":
        split_nemotron(Path(args.prefix), args.pieces, args.jobs)
        return
    if args.cmd == "plan":
        manifest = plan(args.train_tokens, args.margin, args.seed)
        Path(args.out).write_text(json.dumps(manifest, indent=1) + "\n")
        for split, info in manifest["nemotron"].items():
            print(f"{split:12s} {len(info['paths']):3d} shards of {info['available']:5d} (need {info['est_tokens_needed']/1e6:7.0f}M tokens)")
        print("proofpile_2", manifest["proofpile_2"]["paths"])
        return

    manifest = json.loads(Path(args.manifest).read_text())
    prefix = Path(args.prefix)
    sources = set(args.sources.split(","))
    if "nemotron" in sources:
        download_nemotron(manifest, prefix, args.jobs)
    if "proofpile" in sources:
        download_proofpile(manifest, prefix, args.jobs)
    if "paloma" in sources:
        download_paloma(manifest, prefix, args.jobs)
    if "starcoder" in sources:
        download_starcoder(manifest, prefix, args.jobs, manifest["seed"])


if __name__ == "__main__":
    main()
