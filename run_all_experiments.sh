#!/bin/bash

# Script to run all experiments except otherAI_cot for both GPT-4o and GPT-4o-mini
# This will run experiments across all supported datasets and splits

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
    
    echo "Running: $model - $dataset - $split - $exp"
    echo "============================================"
    
    # Create log file name
    log_file="logs/${model}_${dataset}_${split}_${exp}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run the experiment and log output
    python openai_unified_calib.py \
        --model "$model" \
        --dataset "$dataset" \
        --split "$split" \
        --exp "$exp" \
        --temp 0.1 \
        --max_tokens 1024 \
        --workers 40 > "$log_file" 2>&1
        
    echo "Completed: $model - $dataset - $split - $exp"
    echo "============================================"
    echo ""
}

# Main execution
echo "=========================================="
echo "Starting all experiments..."
echo "Models: ${models[*]}"
echo "Experiments: ${experiments[*]}"
echo "Dataset/Split combinations:"
for dataset in "${!dataset_splits[@]}"; do
    echo "  $dataset/${dataset_splits[$dataset]}"
done
echo "=========================================="
echo ""

# Run all experiments
for model in "${models[@]}"; do
    echo "=========================================="
    echo "Starting experiments for model: $model"
    echo "=========================================="
    
    for dataset in "${!dataset_splits[@]}"; do
        split=${dataset_splits[$dataset]}
        
        for exp in "${experiments[@]}"; do
            run_experiment "$model" "$dataset" "$split" "$exp"
        done
    done
    
    echo "Completed all experiments for model: $model"
    echo ""
done

echo "All experiments completed!"
echo "All logs saved in logs/ directory" 