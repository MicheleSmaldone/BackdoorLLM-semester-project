import os

# DefenseBox root, resolved from this file so the scripts work from any checkout location and working directory
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# base models from the HF hub (override with a local path via env var)
LLAMA2_7B = os.environ.get("BASE_MODEL", "meta-llama/Llama-2-7b-chat-hf")
LLAMA2_13B = os.environ.get("BASE_MODEL_13B", "meta-llama/Llama-2-13b-chat-hf")
