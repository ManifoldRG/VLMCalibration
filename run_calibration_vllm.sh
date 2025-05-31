#!/bin/bash

# Define models to test (placeholders for now)
models=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "google/gemma-2-2b-it"
    "Qwen/Qwen2.5-14B-Instruct"
    "allenai/OLMo-2-1124-7B-Instruct"
    # GEMMA NEEDS NO SYSTEM PROMPT - eval later
    "google/gemma-2-9b-it"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "deepseek-ai/deepseek-math-7b-rl"
    "deepseek-ai/deepseek-math-7b-instruct"

    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.2-1B-Instruct"
    "meta-llama/Llama-3.2-3B-Instruct" 
    "allenai/OLMo-2-1124-13B-Instruct"
)

# Define permitted combinations based on local_unified_calib_vllm.py
declare -A permitted_combinations
permitted_combinations["gsm8k"]="test"
permitted_combinations["mmlu"]="test"
permitted_combinations["medmcqa"]="validation"
permitted_combinations["simpleqa"]="test"
permitted_combinations["truthfulqa"]="validation"
# "gsm8k" "mmlu" "medmcqa" "simpleqa" 
datasets=("truthfulqa")
# "otherAI_cot"
exps=("cot_exp" "zs_exp" "verbalized" "verbalized_cot" "otherAI")

# Function to extract parameter size from model name and determine workers
get_workers_for_model() {
    local model_name=$1
    
    # Extract parameter size using regex (handles formats like 1B, 3B, 7B, 14B, 0.5B, etc.)
    # Look for patterns like "XB" or "X.YB" where X and Y are digits
    if [[ $model_name =~ ([0-9]+\.?[0-9]*)([Bb])[^0-9] ]] || [[ $model_name =~ ([0-9]+\.?[0-9]*)([Bb])$ ]]; then
        local param_size=${BASH_REMATCH[1]}
        
        # Convert to float comparison (bash doesn't handle float comparison natively)
        # Use awk for floating point comparison
        if awk "BEGIN {exit !($param_size >= 7)}"; then
            echo 28
        else
            echo 40
        fi
    else
        # Default to 20 workers if we can't parse the model size
        echo 20
    fi
}

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

# Main loop through models
for model in "${models[@]}"; do
    echo "=========================================="
    echo "Starting experiments for model: $model"
    echo "=========================================="
    
    # Determine number of workers for this model
    workers=$(get_workers_for_model "$model")
    echo "Using $workers workers for model: $model"
    
    # Stop any existing vLLM servers
    stop_vllm_server
    
    # Start vLLM server for current model
    echo "Starting vLLM server for $model..."
    vllm serve "$model" \
        --max-model-len 4096 \
        --dtype bfloat16 \
        --gpu-memory-utilization 0.94 \
        --trust-remote-code \
        --max-logprobs 25 \
        --port 8000 > test.log 2>&1 &
        # --tensor-parallel-size 4 \
    
    sleep 120 # wait for the server to start
    # Store the PID of the vLLM server
    VLLM_PID=$!
    echo "vLLM server started with PID: $VLLM_PID"
    # Wait for server to be ready
    if ! check_server_ready 8000; then
        echo "Failed to start vLLM server for $model, skipping..."
        kill $VLLM_PID 2>/dev/null || true
        continue
    fi
    
    # Run experiments for all dataset/experiment combinations
    for dataset in "${datasets[@]}"; do
        split=${permitted_combinations[$dataset]}
        
        for exp in "${exps[@]}"; do
            echo "Running: $model - $dataset - $split - $exp"
            echo "============================================"
            
            # Run the vLLM Python script with the current combination
            python local_unified_calib_vllm.py --dataset $dataset --split $split --exp $exp --port 8000 --workers $workers
            
            echo "Completed: $model - $dataset - $split - $exp"
            echo "============================================"
            echo ""
        done
    done
    
    # Stop the vLLM server for this model
    echo "Stopping vLLM server for $model..."
    kill $VLLM_PID 2>/dev/null || true
    stop_vllm_server
    
    echo "Completed all experiments for model: $model"
    echo ""
done

echo "vLLM calibration experiment completed for all models!" 