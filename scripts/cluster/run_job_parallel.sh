#!/bin/bash
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

#
# Modern SLURM job script for running single reasoning budget experiments
# Arguments:
#   $1: experiment config (e.g., "main_tasks/qwq_32b")
#   $2: reasoning budget (e.g., "4096")
#

set -e

# Parse arguments
EXPERIMENT=${1:-"main_tasks/qwq_32b"}
REASONING_BUDGET=${2:-"0"}

# Job info logging
echo "🚀 Job started at $(date)"
echo "🖥️  Running on host: $(hostname)"
echo "🎮 Available GPUs: $CUDA_VISIBLE_DEVICES"
echo "📋 Experiment: $EXPERIMENT"
echo "🧠 Reasoning Budget: $REASONING_BUDGET"
echo "🆔 SLURM Job ID: $SLURM_JOB_ID"
echo ""

# Activate virtual environment
if [[ -f ".venv/bin/activate" ]]; then
    echo "🐍 Activating virtual environment..."
    source .venv/bin/activate
else
    echo "⚠️  No virtual environment found, using system Python"
fi

# Verify GPU availability
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "✅ GPU(s) allocated: $CUDA_VISIBLE_DEVICES"
else
    echo "⚠️  No GPUs allocated"
fi

echo ""
echo "▶️  Starting experiment..."

# Create logs directory for Hydra if it doesn't exist
mkdir -p .logs

# Run the experiment with specific reasoning budget override
# The key insight: override reasoning_budgets to contain only the single budget for this job
python run_experiment.py \
    experiment="$EXPERIMENT" \
    evaluation.reasoning_budgets="[$REASONING_BUDGET]" \
    experiment.name="${EXPERIMENT//\//_}_budget_${REASONING_BUDGET}" \
    wandb.tags="[${EXPERIMENT//\//_},budget_${REASONING_BUDGET},parallel]"

EXIT_CODE=$?

echo ""
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ Job completed successfully at $(date)"
else
    echo "❌ Job failed with exit code $EXIT_CODE at $(date)"
fi

echo "📊 Final GPU memory usage:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
else
    echo "nvidia-smi not available"
fi

exit $EXIT_CODE