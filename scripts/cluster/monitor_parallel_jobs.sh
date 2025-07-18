#!/bin/bash
#
# Monitor and manage parallel reasoning budget jobs
# Usage: ./monitor_parallel_jobs.sh [command] [experiment]
#

set -e

COMMAND=${1:-"status"}
EXPERIMENT=${2:-"main_tasks_qwq_32b"}

case "$COMMAND" in
    "status"|"monitor")
        echo "📊 Monitoring jobs for experiment: $EXPERIMENT"
        echo ""
        
        # Show relevant jobs
        echo "🔍 Current job status:"
        squeue -u $USER --format="%.18i %.40j %.8T %.10M %.6D %R" | grep -E "(JOBID|$EXPERIMENT)" || {
            echo "No jobs found matching: $EXPERIMENT"
        }
        
        echo ""
        echo "📁 Recent log files:"
        find logs -name "*${EXPERIMENT}*" -type f -mmin -60 | head -10
        ;;
        
    "cancel")
        echo "🛑 Canceling all jobs for experiment: $EXPERIMENT"
        JOB_IDS=$(squeue -u $USER --format="%.18i %.40j" | grep "$EXPERIMENT" | awk '{print $1}' | grep -v JOBID)
        
        if [[ -z "$JOB_IDS" ]]; then
            echo "No running jobs found for: $EXPERIMENT"
        else
            echo "Canceling jobs: $JOB_IDS"
            for job_id in $JOB_IDS; do
                scancel "$job_id"
                echo "✅ Canceled job: $job_id"
            done
        fi
        ;;
        
    "logs")
        echo "📋 Showing recent logs for experiment: $EXPERIMENT"
        echo ""
        
        # Find and display recent log files
        LOG_FILES=$(find logs -name "*${EXPERIMENT}*" -type f -mmin -120 | sort -t_ -k3 -n | head -5)
        
        if [[ -z "$LOG_FILES" ]]; then
            echo "No recent log files found for: $EXPERIMENT"
        else
            for log_file in $LOG_FILES; do
                echo "📄 === $log_file (last 20 lines) ==="
                tail -20 "$log_file"
                echo ""
            done
        fi
        ;;
        
    "results")
        echo "📈 Checking results for experiment: $EXPERIMENT"
        
        # Look for result files
        RESULT_DIRS=$(find results -name "*${EXPERIMENT}*" -type d | head -5)
        
        if [[ -z "$RESULT_DIRS" ]]; then
            echo "No result directories found for: $EXPERIMENT"
        else
            for result_dir in $RESULT_DIRS; do
                echo ""
                echo "📁 Results in: $result_dir"
                
                # Check main results file
                if [[ -f "$result_dir/results_df.csv" ]]; then
                    echo "✅ Main results file exists"
                    LINES=$(wc -l < "$result_dir/results_df.csv")
                    echo "   Lines: $LINES"
                    
                    # Quick statistical robustness check using Python if available
                    if command -v python3 >/dev/null 2>&1; then
                        echo "🔍 Statistical robustness check:"
                        python3 -c "
import pandas as pd
from collections import Counter
try:
    df = pd.read_csv('$result_dir/results_df.csv')
    if 'instance_id' in df.columns and 'reasoning_budget' in df.columns:
        # Check for multiple runs per instance-budget
        instance_budget_counts = Counter()
        for _, row in df.iterrows():
            key = (row['instance_id'], row['reasoning_budget'])
            instance_budget_counts[key] += 1
        multi_run_counts = Counter(instance_budget_counts.values())
        print(f'   Instances with 1 run: {multi_run_counts.get(1, 0)}')
        print(f'   Instances with 2 runs: {multi_run_counts.get(2, 0)}')
        print(f'   Instances with 3 runs: {multi_run_counts.get(3, 0)}')
        print(f'   Instances with 4+ runs: {sum(count for runs, count in multi_run_counts.items() if runs >= 4)}')
        
        # Check budgets
        budgets = sorted(df['reasoning_budget'].unique())
        print(f'   Reasoning budgets: {budgets}')
    else:
        print('   ⚠️  Missing instance_id or reasoning_budget columns')
except Exception as e:
    print(f'   ❌ Error reading CSV: {e}')
" 2>/dev/null || echo "   ⚠️  Could not run statistical check"
                    fi
                else
                    echo "❌ No results_df.csv found"
                fi
                
                # Count prediction files
                PRED_COUNT=$(find "$result_dir" -name "*.jsonl" -type f | wc -l)
                echo "📋 Prediction files: $PRED_COUNT"
                
                # Check for budget-specific directories (for VLLM runs)
                BUDGET_DIRS=$(find "$result_dir" -maxdepth 1 -type d -name "*budget*" -o -name "*_[0-9]*" | wc -l)
                if [[ $BUDGET_DIRS -gt 0 ]]; then
                    echo "🔄 Budget-specific directories found: $BUDGET_DIRS"
                    echo "   Consider running: python scripts/analyze_results.py $result_dir"
                fi
                
                # Check completion status
                if [[ -f "$result_dir/analysis/summary_*.json" ]]; then
                    echo "✅ Analysis completed"
                elif [[ -d "$result_dir/analysis" ]]; then
                    echo "🔄 Analysis in progress"
                else
                    echo "⏳ Analysis not started"
                fi
            done
        fi
        ;;
        
    "help"|*)
        echo "🔧 Parallel Job Management Tool"
        echo ""
        echo "Usage: $0 [command] [experiment_pattern]"
        echo ""
        echo "Commands:"
        echo "  status    - Show current job status (default)"
        echo "  cancel    - Cancel all jobs matching pattern"
        echo "  logs      - Show recent log files"
        echo "  results   - Check experiment results"
        echo "  help      - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 status main_tasks_qwq_32b"
        echo "  $0 logs main_tasks"
        echo "  $0 cancel main_tasks_qwq_32b"
        echo ""
        ;;
esac