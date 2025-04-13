# VLMCalibration

## Experiment Tracking for Calibration

| **Model**                     | **gsm8k-zs** | **gsm8k-cot** | **mmlu-zs** | **mmlu-cot** | **medmcqa-zs** | **medmcqa-cot** | **simpleqa-zs** | **simpleqa-cot** |
|:----------------------------------:|:---------:|:-------------:|:--------:|:------------:|:-------------:|:---------------:|:-------------:|:---------------:|
| **Meta-Llama-3.1-8B-Instruct-Q8_0**|     ✅    |       ✅      |     ✅   |      ❌      |       ✅      |        ❌       |       ✅      |        ✅       |
| **gemma-2-9b-it-Q8_0**             |     ✅    |       ✅      |     ✅   |      ❌      |       ✅      |        ❌       |       ✅      |        ✅       |
| **Qwen2.5-7B-Instruct-Q8_0**       |     ✅    |       ✅      |     ✅   |      ❌      |       ✅      |        ❌       |       ✅      |        ✅       |
| **gpt 4o**                         |     ✅    |       ✅      |     ✅   |      ❌      |       ✅      |        ✅       |       ✅      |        ✅       |
| **gpt 4o mini**                    |     ✅    |       ✅      |     ✅   |      ❌      |       ✅      |        ✅       |       ✅      |        ✅       |


cot_exp = Chain of Thought Experiments

zs_exp = Zero Shot Experiments

### Calibration Experiments conducted:

1. Meta-Llama-3.1-8B-Instruct-Q8_0
2. gemma-2-9b-it-Q8_0
3. Qwen2.5-7B-Instruct-Q8_0

Fit on the GSM8k train set

Switched to llama.cpp C++ server for faster parallel inference instead of sequential python

Example commands to download model (example):
```bash
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF --include "Qwen2.5-7B-Instruct-Q8_0.gguf" --local-dir ./models
```

Command to run server (example):
```bash
# cd to the build/bin directory under llama.cpp folder where there is the llama-server binary, see setup below
./llama-server -m /opt/dlami/nvme/VLMCalibration/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf -c 16384 -np 8 -t 8 -tb 8 -b 4096 -ub 2048 -cb --gpu-layers 300
```

300 is just a large number to put all layers on GPU. -1 didn't seem to work for some reason, but a large number puts all on GPU. Some notes on params:
| Argument | Explanation |
|----------|-------------|
| `-m` | Path to the model file |
| `-c` | Context window size in tokens (divided among -np processes, so if 16384 ctx_size and divided amongst 8 processes, then each has ctx_size of 2048) |
| `-np` | Number of parallel sequences to decode |
| `-t` | Number of threads to use during generation |
| `-tb` | Number of threads to use during batch and prompt processing |
| `-b` | Logical maximum batch size |
| `-ub` | Physical maximum batch size |
| `-cb` | Enable continuous batching |
| `--gpu-layers` | Number of layers to offload to GPU |

### All model running commands:
All must be run from directory of `llama-server` binary

```bash
./llama-server -m /opt/dlami/nvme/models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf -c 32768 -np 16 -t 10 -tb 10 -b 4096 -ub 2048 -cb --gpu-layers 300 # Llama

./llama-server -m /opt/dlami/nvme/models/Qwen2.5-7B-Instruct-Q8_0.gguf -c 49152 -np 24 -t 10 -tb 10 -b 4096 -ub 2048 -cb --gpu-layers 300 # Qwen

./llama-server -m /opt/dlami/nvme/models/gemma-2-9b-it-Q8_0.gguf -c 24576 -np 12 -t 10 -tb 10 -b 4096 -ub 2048 -cb --gpu-layers 300 # Gemma
```



## Steps to setup llama.cpp C++ server
```bash
sudo apt update
sudo apt install -y build-essential cmake git curl libcurl4-openssl-dev
nvcc --version # check CUDA installed
```

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
mkdir build
cd build
cmake -DGGML_CUDA=on -DLLAMA_CURL=on -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release --parallel $(nproc)
cd bin # there should be a llama-server there that can now be run
```

## Utils
UPDATE: No longer needed, watch works fine on 22.04 (now default in EC2)
`nvidia.py` can be used to keep checking GPU usage. Currently watch segfaults on Ubuntu 20.04, so this can be run in place of that to monitor:

```bash
chmod +x nvidia.py
sudo ./nvidia.py
```

## vLLM Info
This is still in development but might help increase inference speed. Details yet to be finalized, but also might not work optimally with GGUFs.
```bash
vllm serve ./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf --tokenizer meta-llama/Llama-3.1-8B-Instruct --trust-remote-code --max-model-len 4096 --host localhost --port 8080 --max-num-batched-tokens 8192
```


