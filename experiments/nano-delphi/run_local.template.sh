#!/usr/bin/env bash
# Launch one nano-delphi variant on a single local GPU, inside tmux, optionally under a GPU flock.
#
#   experiments/nano-delphi/run_local.template.sh VARIANT [SEED] [GPU] [PER_DEVICE_BATCH] [extra train args...]
#   experiments/nano-delphi/run_local.template.sh 1e18-d512 0 0 -1
#
# Paths are relative to this checkout: by default everything lives next to it, in its parent directory
# (WORKSPACE). Override any of these in the environment, or copy this file to *.local.sh (git-ignored) for
# machine-specific settings:
#   WORKSPACE  parent of the checkout                     MARIN_DIR  this checkout
#   ND_ROOT    run outputs, data store and logs           PYTHON     the checkout's venv python
#   HF_HOME    Hugging Face cache                         LOCK       GPU flock file, or "none"
set -euo pipefail
VARIANT=${1:?variant, e.g. 1e18-d512}
SEED=${2:-0}
GPU=${3:-0}
PDB=${4:--1}
shift $(( $# < 4 ? $# : 4 ))
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MARIN_DIR=${MARIN_DIR:-$(cd "${HERE}/../.." && pwd)}
WORKSPACE=${WORKSPACE:-$(dirname "${MARIN_DIR}")}
ND_ROOT=${ND_ROOT:-${WORKSPACE}/nano-delphi}
PYTHON=${PYTHON:-${MARIN_DIR}/.venv/bin/python}
HF_HOME=${HF_HOME:-${WORKSPACE}/hf}
if [[ -z "${LOCK:-}" ]]; then
  if [[ -d "${WORKSPACE}/gpu-coordination" ]]; then LOCK="${WORKSPACE}/gpu-coordination/gpu${GPU}.lock"; else LOCK=none; fi
fi
SESSION="nd-${VARIANT}-s${SEED}"
LOG="${ND_ROOT}/logs/${SESSION}.log"
JOB="${ND_ROOT}/logs/${SESSION}.sh"
mkdir -p "${ND_ROOT}/logs" "${ND_ROOT}/store" "${ND_ROOT}/tmp" "${ND_ROOT}/wandb"

if [[ "${LOCK}" == none ]]; then RUN_PREFIX=""; else RUN_PREFIX="flock -w 600 ${LOCK}"; fi

cat > "${JOB}" <<JOBEOF
#!/usr/bin/env bash
cd ${ND_ROOT}  # keep stray outputs (e.g. Levanter's logs/) out of the checkout
export CUDA_VISIBLE_DEVICES=${GPU}
export MARIN_PREFIX=${ND_ROOT}/store
export HF_HOME=${HF_HOME} HF_HUB_DISABLE_XET=1
export WANDB_MODE=offline WANDB_DIR=${ND_ROOT}/wandb
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export TMPDIR=${ND_ROOT}/tmp
echo "start \$(date -u +%FT%TZ) host \$(hostname) gpu ${GPU} commit \$(git -C ${MARIN_DIR} rev-parse --short HEAD) variant ${VARIANT} seed ${SEED}"
${RUN_PREFIX} ${PYTHON} ${MARIN_DIR}/experiments/nano-delphi/train_nano_delphi.py \\
  --variant ${VARIANT} --seed ${SEED} --per-device-batch ${PDB} $*
echo "exit \$? \$(date -u +%FT%TZ)"
JOBEOF
chmod +x "${JOB}"
tmux new-session -d -s "${SESSION}" "bash ${JOB} 2>&1 | tee -a ${LOG}"
echo "started tmux session ${SESSION}; log ${LOG}"
