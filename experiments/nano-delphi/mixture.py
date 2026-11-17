# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Modifications copyright 2026 Qinghua Zhou

"""The Delphi training mixture and validation sets as tokenized-dataset handles over the local subsets.

Components, order and weights are those of ``experiments/references/train_delphi_seeds.py``
(``DATASETS``). The raw inputs are the subsets staged by ``prepare_data.py`` under ``MARIN_PREFIX``.

Tokenizer: ``NousResearch/Meta-Llama-3.1-8B``. Its ``tokenizer.json``, ``tokenizer_config.json`` and
``special_tokens_map.json`` are byte-identical to ``meta-llama/Llama-3.1-8B`` (the repo that
``meta-llama/Meta-Llama-3.1-8B``, Delphi's tokenizer, redirects to): LFS/git oids ``f916e710``,
``cb9ec255`` and ``d8cd5076``, checked 2026-11-15. The Nous mirror is not gated, so the training host
needs no Hugging Face token.

Paloma reuses ``experiments.datasets.paloma`` unchanged; its raw files sit at the path that module
expects (``raw/paloma-fc6827/65cd6fc``).

Run ``python experiments/nano-delphi/mixture.py`` to print the tokenize plan and ``--run`` to build it.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.datasets.nemotron import NEMOTRON_SPLIT_GLOBS
from experiments.datasets.paloma import paloma_datasets
from prepare_data import NEMOTRON_RAW, PROOFPILE_RAW, STARCODER_RAW, WEIGHTS

TOKENIZER = "NousResearch/Meta-Llama-3.1-8B"
# Calendar version of the local subset caches (bump if the staged raw subset changes).
DATA_VERSION = "2026.11.15"
_NEMOTRON_DATA = f"{NEMOTRON_RAW}/contrib/Nemotron/Nemotron-CC/data-jsonl"


def train_datasets() -> dict[str, ArtifactStep[TokenizedCache]]:
    """Training components keyed by their Delphi component name, in Delphi's order."""
    handles = {
        f"nemotron_cc/{split}": tokenized(
            f"nano-delphi/nemotron_cc/{split}",
            tokenizer=TOKENIZER,
            version=DATA_VERSION,
            paths=[f"{_NEMOTRON_DATA}/{glob}"],
        )
        for split, glob in NEMOTRON_SPLIT_GLOBS.items()
    }
    handles["starcoderdata"] = tokenized(
        "nano-delphi/starcoderdata",
        tokenizer=TOKENIZER,
        version=DATA_VERSION,
        paths=[f"{STARCODER_RAW}/*/*.parquet"],
        text_key="content",
    )
    handles["proofpile_2"] = tokenized(
        "nano-delphi/proofpile_2",
        tokenizer=TOKENIZER,
        version=DATA_VERSION,
        paths=[f"{PROOFPILE_RAW}/*/train/*.jsonl.zst"],
    )
    assert list(handles) == list(WEIGHTS), "component order must match the Delphi mixture"
    return handles


def train_weights() -> dict[ArtifactStep[TokenizedCache], float]:
    handles = train_datasets()
    return {handles[name]: weight for name, weight in WEIGHTS.items()}


def validation_datasets() -> list[ArtifactStep[TokenizedCache]]:
    """The 16 Paloma sets Delphi validated on (Uncheatable Eval is omitted: Delphi's pinned
    ``latest-364401`` snapshot is not public)."""
    return list(paloma_datasets(tokenizer=TOKENIZER, tag="llama3").values())


if __name__ == "__main__":
    handles = dict(train_datasets())
    handles.update({h.name: h for h in validation_datasets()})
    dataset_main(handles)
