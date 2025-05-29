#!/bin/bash

# Define permitted combinations based on local_unified_calib_vllm.py
declare -A permitted_combinations
permitted_combinations["gsm8k"]="test"
permitted_combinations["mmlu"]="test"
permitted_combinations["medmcqa"]="validation"
permitted_combinations["simpleqa"]="test"

datasets=("gsm8k" "mmlu" "medmcqa" "simpleqa")
exps=("verbalized" "verbalized_cot")

# Loop through datasets and their permitted splits
for dataset in "${datasets[@]}"; do
  split=${permitted_combinations[$dataset]}
  
  for exp in "${exps[@]}"; do
    echo "Running: $dataset - $split - $exp"
    echo "============================================"
    
    # Run the vLLM Python script with the current combination
    python local_unified_calib_vllm.py --dataset $dataset --split $split --exp $exp --port 8000 --workers 32
    
    echo "Completed: $dataset - $split - $exp"
    echo "============================================"
    echo ""
  done
done

echo "First vLLM calibration experiment completed!" 