MODEL_PROVIDER = "ollama"
MODEL_NAME = "qwen2.5:7b-instruct"
OLLAMA_BASE_URL = "http://localhost:11434"

# Context window budget — set explicitly so we know the real limit.
# qwen2.5:7b default is 4096; bump to 8192 if your Ollama build supports it.
NUM_CTX = 4096

# Extra delay per streamed chunk, in milliseconds. Set to 0 to stream at the
# model's native rate. A small delay (~8ms) slows output ~30% so it's easier
# to read as it arrives; tune to taste.
STREAM_DELAY_MS = 100

# Cap on tool result size (in characters) before it's appended to the message
# history. Anything bigger gets sliced and a marker is appended. Prevents a
# single `read_file` on a 5MB log from blowing the context window.
TOOL_RESULT_MAX_BYTES = 10_000
