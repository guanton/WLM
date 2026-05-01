#!/usr/bin/env bash
#SBATCH --job-name=WLM_run
#SBATCH --output=logs/run_%A_%a.out
#SBATCH --error=logs/run_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --partition=long
#SBATCH --requeue
#
# Generic runlist-driven sbatch.
#
# Usage:
#   RUNS_DIR=<top_level_out_dir> sbatch --array=0-N scripts/sbatch_run.sbatch <runlist.tsv>
#
# Example:
#   RUNS_DIR=oceans_interpolate \
#     sbatch --array=0-11 scripts/sbatch_run.sbatch runlists/oceans_interpolate.tsv
#
# Runlist format (tab-separated, # and blank lines skipped):
#   master_tag  data_cfg  train_cfg  run_tag  seed  [extra args for train.py...]
#
# All env knobs (RUNS_DIR, DEVICE, WANDB_ON, REGEN_DATA, SKIP_DONE) flow through
# to scripts/run_from_configs.sh.

set -euo pipefail

source ~/.bashrc
micromamba activate gradient-flow-sdes

cd ~/repos/WLM
mkdir -p logs

RUNLIST="${1:-}"
[[ -n "${RUNLIST}" && -f "${RUNLIST}" ]] || { echo "ERROR: runlist not found: ${RUNLIST:-<missing>}" >&2; exit 1; }

# Pick the SLURM_ARRAY_TASK_ID-th data line (0-indexed); skip comment/blank lines.
line="$(
  awk 'BEGIN{i=0}
       $0 ~ /^[[:space:]]*#/ {next}
       NF==0 {next}
       { if (i==ENVIRON["SLURM_ARRAY_TASK_ID"]) {print; exit} i++ }' \
    "${RUNLIST}"
)"
[[ -n "${line}" ]] || { echo "ERROR: no data line for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }

# Tab-split into the 5 fixed positional fields + everything else as one string.
master_tag="$(awk -F'\t' '{print $1}' <<< "${line}")"
data_cfg="$(  awk -F'\t' '{print $2}' <<< "${line}")"
train_cfg="$( awk -F'\t' '{print $3}' <<< "${line}")"
run_tag="$(   awk -F'\t' '{print $4}' <<< "${line}")"
seed="$(      awk -F'\t' '{print $5}' <<< "${line}")"
extra_args="$(awk -F'\t' '{for(i=6;i<=NF;i++) printf $i" "; print ""}' <<< "${line}")"

# Defaults — RUNS_DIR is the only one you usually set on the sbatch invocation.
export RUNS_DIR="${RUNS_DIR:-runs}"
export DEVICE="${DEVICE:-cuda}"
export WANDB_ON="${WANDB_ON:-1}"
export REGEN_DATA="${REGEN_DATA:-0}"
export SKIP_DONE="${SKIP_DONE:-1}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

# Local-disk wandb cache on the compute node (avoid hammering NFS)
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    export WANDB_DIR="${SLURM_TMPDIR}"
    export WANDB_CACHE_DIR="${SLURM_TMPDIR}/.cache/wandb"
    export WANDB_CONFIG_DIR="${SLURM_TMPDIR}/.config/wandb"
    mkdir -p "${WANDB_DIR}"
fi

echo "==================================================="
echo "[run] runlist    = ${RUNLIST}"
echo "[run] task       = ${SLURM_ARRAY_TASK_ID}  job=${SLURM_JOB_ID}"
echo "[run] RUNS_DIR   = ${RUNS_DIR}"
echo "[run] master_tag = ${master_tag}"
echo "[run] data_cfg   = ${data_cfg}"
echo "[run] train_cfg  = ${train_cfg}"
echo "[run] run_tag    = ${run_tag}"
echo "[run] seed       = ${seed}"
echo "[run] extra_args = ${extra_args}"
echo "[run] start at $(date)"
echo "==================================================="

# `${extra_args}` is intentionally unquoted so word-splitting forwards each
# token (e.g. `--holdout-marginals 1 3 5 7` becomes 5 separate argv tokens).
bash scripts/run_from_configs.sh \
    "${master_tag}" \
    "${data_cfg}" \
    "${train_cfg}" \
    "${run_tag}" \
    "${seed}" \
    ${extra_args}

echo "[run] done at $(date)"
