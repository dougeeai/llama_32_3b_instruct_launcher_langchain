# %% [0.0] Launcher Script Info
r"""
Llama 3.2 3B Instruct GGUF Launcher - LangChain Version
Filename: llama_32_3b_instruct_launcher_langchain.py
Description: LangChain launcher for Llama-3.2-3B-Instruct GGUF models
Author: dougeeai
Created: 2025-11-09
Last Updated: 2025-11-09
"""

# %% [0.1] Model Card & Summary
"""
MODEL: Llama-3.2-3B-Instruct-Q8_0
LangChain version - uses LangChain abstractions for GGUF
"""

# %% [1.0] Core Imports
from langchain_community.llms import LlamaCpp
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

# %% [1.1] Utility Imports
# Minimal imports for bare bones version
import sys

# %% [2.0] User Configuration - All Settings
MODEL_PATH = r"E:\ai\models\llama_32_3b_instruct_gguf\llama_32_3b_instruct_q8_0.gguf" #Update model location
GPU_LAYERS = 28
CONTEXT_LENGTH = 131072
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 40
REPEAT_PENALTY = 1.1
MAX_TOKENS = 2048
SYSTEM_MESSAGE = "You are a helpful AI assistant."
VERBOSE = False  # Set True for debugging

# %% [2.1] Model Configuration Dataclass
# Bare Version: Using direct variables for LangChain config

# %% [2.2] Model Path Validation
# Bare Version: Path validation skipped - LangChain will error if missing

# %% [2.3] Model Paths - HF Download (Optional)
# Bare Version: HF download option skipped

# %% [3.0] Hardware Auto-Detection
# Bare Version: Auto-detection skipped

# %% [3.1] Hardware Detection
# Bare Version: Hardware detection skipped

# %% [3.2] Environment Validation
# Bare Version: Environment validation skipped

# %% [4.0] Model Loader
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
def main():
    """Bare bones main - just load and chat"""
    print("Loading model with LangChain...")
    llm = load_model()
    print("Model loaded. Starting chat...")
    chat_loop(llm)

if __name__ == "__main__":
    main()