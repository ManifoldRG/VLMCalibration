#!/bin/bash

# Script for running calibration experiments specifically on deepseek-ai/deepseek-math-7b-instruct
# Runs all allowed dataset/experiment/split combinations

model="deepseek-ai/deepseek-math-7b-instruct"

# Define permitted combinations based on local_unified_calib_vllm.py
declare -A permitted_combinations
permitted_combinations["gsm8k"]="test"
permitted_combinations["mmlu"]="test"
permitted_combinations["medmcqa"]="validation"
permitted_combinations["simpleqa"]="test"

datasets=("gsm8k" "mmlu" "medmcqa" "simpleqa")
# All allowed experiment types from local_unified_calib_vllm.py
exps=("cot_exp" "zs_exp" "few_shot" "verbalized" "verbalized_cot")

# Workers for 7B model (>=7B gets 20 workers)
workers=20

# Function to check if vLLM server is ready
check_server_ready() {
    local port=$1
    local max_attempts=30
    local attempt=0
    
    echo "Waiting for vLLM server to be ready on port $port..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo "Server is ready!"
            return 0
        fi
        echo "Attempt $((attempt + 1))/$max_attempts: Server not ready yet..."
        sleep 30
        attempt=$((attempt + 1))
    done
    
    echo "Server failed to start within expected time"
    return 1
}

# Function to stop vLLM server
stop_vllm_server() {
    echo "Stopping vLLM server..."
    # Kill any running vLLM processes
    pkill -f "vllm serve" || true
    # Wait a bit for cleanup
    sleep 10
    echo "vLLM server stopped"
}

echo "=========================================="
echo "Starting calibration experiments for: $model"
echo "Using $workers workers"
echo "=========================================="

# Stop any existing vLLM servers
stop_vllm_server

# Start vLLM server for deepseek-math model
echo "Starting vLLM server for $model..."
vllm serve "$model" \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.94 \
    --trust-remote-code \
    --max-logprobs 25 \
    --tensor-parallel-size 4 \
    --port 8000 > /dev/null 2>&1 &

sleep 120 # wait for the server to start
# Store the PID of the vLLM server
VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for server to be ready
if ! check_server_ready 8000; then
    echo "Failed to start vLLM server for $model, exiting..."
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Run experiments for all dataset/experiment combinations
total_experiments=0
completed_experiments=0

# Count total experiments
for dataset in "${datasets[@]}"; do
    for exp in "${exps[@]}"; do
        ((total_experiments++))
    done
done

echo "Will run $total_experiments experiments total"
echo ""

for dataset in "${datasets[@]}"; do
    split=${permitted_combinations[$dataset]}
    
    for exp in "${exps[@]}"; do
        ((completed_experiments++))
        echo "[$completed_experiments/$total_experiments] Running: $model - $dataset - $split - $exp"
        echo "============================================"
        
        # Run the vLLM Python script with the current combination
        python local_unified_calib_vllm.py --dataset $dataset --split $split --exp $exp --port 8000 --workers $workers
        
        if [ $? -eq 0 ]; then
            echo "✅ Completed: $model - $dataset - $split - $exp"
        else
            echo "❌ Failed: $model - $dataset - $split - $exp"
        fi
        echo "============================================"
        echo ""
    done
done

# Stop the vLLM server
echo "Stopping vLLM server for $model..."
kill $VLLM_PID 2>/dev/null || true
stop_vllm_server

echo "🎉 All calibration experiments completed for $model!"
echo "Completed $completed_experiments/$total_experiments experiments" 