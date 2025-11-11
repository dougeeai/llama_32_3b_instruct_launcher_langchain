# %% [0.0] Launcher Script Info
# Script metadata and documentation for version tracking
# Llama 3.2 3B Instruct GGUF Launcher - LangChain Version
# Description: LangChain launcher for Llama-3.2-3B-Instruct GGUF models
# Author: dougeeai
# Created: 2025-11-09
# Last Updated: 2025-11-11

# %% [0.1] Model Card & Summary
# Quick reference for model capabilities and requirements
# MODEL: Llama-3.2-3B-Instruct
# Architecture: Llama 3.2 (3.21B parameters)
# Framework: LangChain with llama-cpp-python backend

# %% [1.0] Core Imports
# Essential Python libraries required for LangChain GGUF operation
import os
import sys
from pathlib import Path

from langchain_community.llms import LlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

# %% [1.1] Utility Imports
# Bare Version: Utility imports skipped

# %% [2.0] Base Directory Configuration
# Set base AI directory - all paths will be relative to this location

# Set your base directory (change this for your system)
BASE_DIR = r"E:\ai"  # <-- CHANGE THIS to your AI folder location

# Directory structure (automatically created from BASE_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
HF_DOWNLOADS_DIR = os.path.join(MODELS_DIR, "huggingface_downloads")  # For HF downloads
HF_CACHE_DIR = os.path.join(HF_DOWNLOADS_DIR, "cache")  # HF cache

# Create directories if they don't exist
for dir_path in [MODELS_DIR, HF_DOWNLOADS_DIR, HF_CACHE_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# %% [2.1] Model Source Configuration
# Configure where to load the model from - local file or HuggingFace download

# Choose model source: "local" or "huggingface"
MODEL_SOURCE = "local"  # Options: "local" or "huggingface"

# Model identification
MODEL_NAME = "llama_32_3b_instruct_gguf"  # Folder name for this model
MODEL_FILENAME = "llama_32_3b_instruct_q8_0.gguf"  # Actual GGUF filename

# Local file configuration (for manually downloaded models)
# Local models go directly in: BASE_DIR/models/MODEL_NAME/
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME, MODEL_FILENAME)

# HuggingFace configuration
HF_REPO_ID = "bartowski/Llama-3.2-3B-Instruct-GGUF"
HF_FILENAME = "Llama-3.2-3B-Instruct-Q8_0.gguf"  # Filename on HuggingFace
# HuggingFace models will be saved to: models/huggingface_downloads/MODEL_NAME/
HF_MODEL_DIR = os.path.join(HF_DOWNLOADS_DIR, MODEL_NAME)

# Resolve actual model path based on source
if MODEL_SOURCE == "huggingface":
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading model from HuggingFace: {HF_REPO_ID}/{HF_FILENAME}")
        print(f"Download location: {HF_MODEL_DIR}")
        print(f"Cache location: {HF_CACHE_DIR}")
        
        MODEL_PATH = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            cache_dir=HF_CACHE_DIR,
            local_dir=HF_MODEL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded to: {MODEL_PATH}")
    except ImportError:
        print("ERROR: huggingface-hub not installed. Run: pip install huggingface-hub")
        print("Falling back to local path...")
        MODEL_PATH = LOCAL_MODEL_PATH
    except Exception as e:
        print(f"ERROR downloading from HuggingFace: {e}")
        print("Falling back to local path...")
        MODEL_PATH = LOCAL_MODEL_PATH
else:
    MODEL_PATH = LOCAL_MODEL_PATH
    if not Path(MODEL_PATH).exists():
        print(f"WARNING: Local model not found at: {MODEL_PATH}")
        print(f"Expected location: {LOCAL_MODEL_PATH}")
        print(f"To download from HuggingFace, set MODEL_SOURCE = 'huggingface'")

# %% [2.2] User Configuration - All Settings
# Central location for all user-modifiable model and generation settings

GPU_LAYERS = 28
CONTEXT_LENGTH = 131072
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 40
REPEAT_PENALTY = 1.1
MAX_TOKENS = 2048
SYSTEM_MESSAGE = "You are a helpful AI assistant."
VERBOSE = False  # Set True for debugging

# %% [2.3] Model Configuration Dataclass
# Bare Version: Dataclass skipped - using direct variables

# %% [3.0] Hardware Auto-Detection
# Bare Version: Auto-detection skipped

# %% [3.1] Hardware Detection
# Bare Version: Hardware detection skipped

# %% [3.2] Environment Validation
# Bare Version: Environment validation skipped

# %% [4.0] Model Loader
# Direct model loading function with LangChain
def load_model():
    """Load the GGUF model with LangChain"""
    # Callback for streaming
    streaming_callback = StreamingStdOutCallbackHandler()
    
    # Initialize LlamaCpp with LangChain
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=GPU_LAYERS,
        n_ctx=CONTEXT_LENGTH,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        repeat_penalty=REPEAT_PENALTY,
        max_tokens=MAX_TOKENS,
        verbose=VERBOSE,
        streaming=True,
        callbacks=[streaming_callback],  # Use callbacks instead of callback_manager
    )
    
    return llm

# %% [4.1] Model Validation
# Bare Version: Model validation handled by LangChain

# %% [5.0] Model Initialization
# Bare Version: Using direct load_model() instead

# %% [6.0] Inference Test
# Bare Version: Inference test skipped

# %% [6.1] Terminal Chat Interface
# Minimal chat loop with LangChain streaming
def chat_loop(llm):
    """Minimal chat interface with LangChain"""
    print("Chat Interface - Type 'quit' to exit")
    print("-" * 40)
    
    # Initialize conversation with system message
    conversation_history = SYSTEM_MESSAGE + "\n\n"
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif not user_input:
            continue
        
        # Build prompt with history
        prompt = conversation_history + f"Human: {user_input}\nAssistant: "
        
        print("\nAssistant: ", end="", flush=True)
        
        # Generate response with LangChain
        response = llm.invoke(prompt)
        
        print()  # New line after streaming output
        
        # Update conversation history
        conversation_history += f"Human: {user_input}\nAssistant: {response}\n\n"
        
        # Keep history manageable (trim if too long)
        if len(conversation_history) > 10000:  # Rough character limit
            # Keep system message and last few exchanges
            lines = conversation_history.split('\n')
            conversation_history = SYSTEM_MESSAGE + "\n\n" + "\n".join(lines[-50:])

# %% [7.0] Optional Features
# Bare Version: Optional features skipped
# Note: LangChain offers chains, agents, tools, etc. for advanced use

# %% [8.0] Main Entry Point
# Simple main function - load and chat
def main():
    """Bare bones main - just load and chat"""
    print("Loading model with LangChain...")
    llm = load_model()
    print("Model loaded. Starting chat...")
    chat_loop(llm)

if __name__ == "__main__":
    main()
