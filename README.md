# Llama 3.2 3B Instruct LangChain Launcher

Quick and dirty launcher script for running Llama-3.2-3B-Instruct GGUF models with LangChain.

## Description

Simple Python script using LangChain to load and chat with Llama 3.2 3B Instruct GGUF quantized models. Minimal setup with LangChain abstractions for easy integration into larger applications.

## Requirements

- Python 3.13
- LangChain and llama-cpp-python
- 4-5GB VRAM for Q8_0 quantization
- GGUF model file

## Setup

1. Create conda environment:
```bash
conda env create -f environment.yml
conda activate llama_32_3b_instruct_launcher_langchain
```

2. Install llama-cpp-python wheel:
```bash
# Option A: Download pre-built wheel from https://github.com/dougeeai/llama-cpp-python-wheels
pip install llama_cpp_python-X.X.X-cpXXX-cpXXX-win_amd64.whl

# Option B: Build from source
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/13.0
```

3. **IMPORTANT: Update model path in the script**
```python
MODEL_PATH = r"E:\ai\models\llama_32_3b_instruct_gguf\llama_32_3b_instruct_q8_0.gguf"  # Change to your model location
```

4. Run:
```bash
python llama_32_3b_instruct_launcher_langchain.py
```

## Files

- `llama_32_3b_instruct_launcher_langchain.py` - LangChain-based launcher with streaming output
- `environment.yml` - Conda environment specification with LangChain dependencies

## Configuration

Key settings in the script:
- `GPU_LAYERS` - Number of layers to offload to GPU (28 = all layers)
- `CONTEXT_LENGTH` - Context window (131072 = full 128K context)
- `VERBOSE` - Set to True for debugging output

## Usage

Type messages at the prompt. Commands:
- `quit` - Exit the chat

## Notes

- Uses LangChain's LlamaCpp wrapper for cleaner abstraction
- Streaming output enabled by default
- Conversation history managed automatically
- Can be extended with LangChain chains, agents, and tools
