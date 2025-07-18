#!/bin/bash
#
# Modern job submission script for parallelizing reasoning budgets across separate SLURM jobs
# Usage: ./submit_jobs_parallel.sh [experiment_config] [num_gpus] [nodelist] [priority]
# Example: ./submit_jobs_parallel.sh main_tasks/qwq_32b 2 node-4 normal
# Example: ./submit_jobs_parallel.sh main_tasks/qwq_32b 2 node-4 low
# Example: ./submit_jobs_parallel.sh main_tasks/qwq_32b 2 node-4 high
#

set -e

# Default experiment or use first argument
EXPERIMENT=${1:-"main_tasks/qwq_32b"}

# Number of GPUs to request (default: 2)
NUM_GPUS=${2:-2}

# Nodes to use (default: any available)
NODELIST=${3:-""}

# Priority level (default: normal)
PRIORITY=${4:-"normal"}

# Build node selection parameter
if [[ -n "$NODELIST" ]]; then
    NODE_PARAM="--nodelist=$NODELIST"
else
    NODE_PARAM=""
fi

# Build priority parameter
case "$PRIORITY" in
    "high")
        PRIORITY_PARAM="--nice=-100"
        PRIORITY_DESC="high priority (nice=-100)"
        ;;
    "low")
        PRIORITY_PARAM="--nice=100"
        PRIORITY_DESC="low priority (nice=100)"
        ;;
    "normal"|"")
        PRIORITY_PARAM=""
        PRIORITY_DESC="normal priority (nice=0, default)"
        ;;
    *)
        echo "❌ Invalid priority: $PRIORITY. Use: high, normal, or low"
        exit 1
        ;;
esac

# Extract reasoning budgets from the experiment config
CONFIG_FILE="config/experiment/${EXPERIMENT}.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    echo "Available experiments:"
    find config/experiment -name "*.yaml" -type f | sed 's|config/experiment/||' | sed 's|\.yaml||' | sort
    exit 1
fi

echo "🚀 Submitting parallel jobs for experiment: $EXPERIMENT"
echo "📁 Using config: $CONFIG_FILE"
echo "🎮 Requesting $NUM_GPUS GPU(s) per job"
echo "⭐ Priority: $PRIORITY_DESC"
if [[ -n "$NODELIST" ]]; then
    echo "🖥️  Using nodes: $NODELIST"
else
    echo "🖥️  Using any available nodes"
fi

# Parse reasoning budgets from YAML (handles both list and scalar formats)
REASONING_BUDGETS=$(python3 -c "
import yaml
with open('$CONFIG_FILE', 'r') as f:
    config = yaml.safe_load(f)
budgets = config.get('evaluation', {}).get('reasoning_budgets', [])
if isinstance(budgets, list):
    print(' '.join(map(str, budgets)))
else:
    print(str(budgets))
")

if [[ -z "$REASONING_BUDGETS" ]]; then
    echo "❌ Error: No reasoning_budgets found in config file"
    exit 1
fi

echo "🧠 Found reasoning budgets: $REASONING_BUDGETS"

# Count total jobs to submit
BUDGET_ARRAY=($REASONING_BUDGETS)
TOTAL_JOBS=${#BUDGET_ARRAY[@]}
echo "📊 Total jobs to submit: $TOTAL_JOBS"

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit one job per reasoning budget
SUBMITTED_JOBS=()
for budget in $REASONING_BUDGETS; do
    # Create unique job name
    EXPERIMENT_CLEAN=$(echo "$EXPERIMENT" | tr '/' '_')
    JOB_NAME="${EXPERIMENT_CLEAN}_budget_${budget}"
    
    echo "📤 Submitting job: $JOB_NAME (budget: $budget)"
    
    # Submit the job and capture job ID
    JOB_ID=$(sbatch --job-name="$JOB_NAME" --gres=gpu:$NUM_GPUS $NODE_PARAM $PRIORITY_PARAM --parsable scripts/cluster/run_job_parallel.sh "$EXPERIMENT" "$budget")
    SUBMITTED_JOBS+=("$JOB_ID")
    
    echo "✅ Job submitted with ID: $JOB_ID"
done

echo ""
echo "🎉 Successfully submitted $TOTAL_JOBS parallel jobs!"
echo "📋 Job IDs: ${SUBMITTED_JOBS[*]}"
echo ""
echo "📊 Monitor progress with:"
echo "  squeue -u \$USER --format=\"%.18i %.12j %.8T %.10M %.6D %R\""
echo ""
echo "📁 View logs in:"
echo "  logs/job_${EXPERIMENT_CLEAN}_budget_*"
echo ""
echo "🔄 Results will be automatically merged when all jobs complete"