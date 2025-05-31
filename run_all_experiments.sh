#!/bin/bash

# Script to run all experiments except otherAI_cot for both GPT-4o and GPT-4o-mini
# This will run experiments across all supported datasets and splits

set -e  # Exit on any error

# Define arrays for the parameters
models=("gpt-4o" "gpt-4o-mini")
experiments=("cot_exp" "zs_exp" "verbalized" "verbalized_cot" "otherAI")

# Define dataset/split combinations (based on allowed_combinations in the code)
declare -A dataset_splits
dataset_splits["gsm8k"]="test"
dataset_splits["medmcqa"]="validation" 
dataset_splits["mmlu"]="test"
dataset_splits["simpleqa"]="test"
dataset_splits["truthfulqa"]="validation"

# Create logs directory
mkdir -p logs

# Function to run a single experiment
run_experiment() {
    local model=$1
    local dataset=$2
    local split=$3
    local exp=$4
    
    echo "Running: $model | $dataset/$split | $exp"
    
    # Create log file name
    log_file="logs/${model}_${dataset}_${split}_${exp}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run the experiment and log output
    if python openai_unified_calib.py \
        --model "$model" \
        --dataset "$dataset" \
        --split "$split" \
        --exp "$exp" \
        --temp 0.1 \
        --max_tokens 1024 \
        --workers 40 > "$log_file" 2>&1; then
        echo "✓ Completed: $model | $dataset/$split | $exp"
    else
        echo "✗ Failed: $model | $dataset/$split | $exp (see $log_file)"
        return 1
    fi
}

# Main execution
echo "Starting all experiments..."
echo "Models: ${models[*]}"
echo "Experiments: ${experiments[*]}"
echo "Dataset/Split combinations:"
for dataset in "${!dataset_splits[@]}"; do
    echo "  $dataset/${dataset_splits[$dataset]}"
done
echo ""

total_experiments=0
completed_experiments=0
failed_experiments=0

# Calculate total number of experiments
for model in "${models[@]}"; do
    for dataset in "${!dataset_splits[@]}"; do
        for exp in "${experiments[@]}"; do
            ((total_experiments++))
        done
    done
done

echo "Total experiments to run: $total_experiments"
echo "Starting execution..."
echo ""

# Run all experiments
for model in "${models[@]}"; do
    for dataset in "${!dataset_splits[@]}"; do
        split=${dataset_splits[$dataset]}
        for exp in "${experiments[@]}"; do
            echo "Progress: $((completed_experiments + failed_experiments + 1))/$total_experiments"
            
            if run_experiment "$model" "$dataset" "$split" "$exp"; then
                ((completed_experiments++))
            else
                ((failed_experiments++))
            fi
            
            echo ""
        done
    done
done

# Summary
echo "========================================"
echo "EXPERIMENT SUMMARY"
echo "========================================"
echo "Total experiments: $total_experiments"
echo "Completed: $completed_experiments"
echo "Failed: $failed_experiments"
echo ""

if [ $failed_experiments -eq 0 ]; then
    echo "🎉 All experiments completed successfully!"
else
    echo "⚠️  Some experiments failed. Check the log files in the logs/ directory."
    echo "Failed experiment logs can be found in logs/"
fi

echo "All logs saved in logs/ directory" 