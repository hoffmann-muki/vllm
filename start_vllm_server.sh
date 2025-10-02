#!/bin/bash
# Simple script to start vLLM server for InkubaLM-0.4B

echo "🚀 Starting vLLM server for InkubaLM-0.4B..."
echo "📡 Server will be available at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
echo "🌐 Your browser can call: http://localhost:8000/v1/chat/completions"
echo "⏹️  Press Ctrl+C to stop the server"
echo

python -m vllm.entrypoints.openai.api_server \
    --model ./models/InkubaLM-0.4B \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name InkubaLM-0.4B \
    --max-model-len 2048 \
    --dtype auto \
    --disable-log-requests
