MODEL_PROVIDER = "ollama"
MODEL_NAME = "qwen2.5:7b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434"

# Context window budget — set explicitly so we know the real limit.
# qwen2.5:7b default is 4096; bump to 8192 if your Ollama build supports it.
NUM_CTX = 4096
