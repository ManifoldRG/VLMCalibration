#!/bin/bash
datasets=("gsm8k" "mmlu" "medmcqa" "simpleqa")
splits=("train" "test" "validation")
exps=("cot_exp" "zs_exp")

# Loop through all combinations
for dataset in "${datasets[@]}"; do
  for split in "${splits[@]}"; do
    # Skip medmcqa test split as it doesn't contain answers
    if [ "$dataset" == "medmcqa" ] && [ "$split" == "test" ]; then
      echo "Skipping $dataset $split (no answers available)"
      continue
    fi
    
    for exp in "${exps[@]}"; do
      echo "Running: $dataset - $split - $exp"
      echo "============================================"
      
      # Run the Python script with the current combination
      python local_unified_calib.py --dataset $dataset --split $split --exp $exp
      
      echo "Completed: $dataset - $split - $exp"
      echo "============================================"
      echo ""
    done
  done
done

echo "All calibration experiments completed!" 