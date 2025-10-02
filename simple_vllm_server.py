#!/usr/bin/env python3
"""
Simple vLLM Server for InkubaLM-0.4B
Run this script to serve the model via OpenAI-compatible API
"""

def serve_inkuba():
    """
    Serve InkubaLM-0.4B using vLLM command line
    """
    import subprocess
    import sys
    
    model_path = "./dist/models/InkubaLM-0.4B"
    
    print("🚀 Starting vLLM server for InkubaLM-0.4B...")
    print("📡 Server will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("🌐 Your browser can call: http://localhost:8000/v1/chat/completions")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    # Command to run vLLM server
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--trust-remote-code",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--served-model-name", "InkubaLM-0.4B",
        "--max-model-len", "2048",
        "--dtype", "auto",  # Let vLLM decide the best dtype
        "--disable-log-requests"  # Reduce terminal noise
        # No --api-key means no authentication required
    ]
    
    try:
        # Run the server
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    serve_inkuba()
