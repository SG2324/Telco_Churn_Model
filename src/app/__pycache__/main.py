import uvicorn
import os
import sys
import socket

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def find_available_port(preferred_port: int = 8000) -> int:
    for port in [preferred_port, 8080, 8001, 8002, 8501]:
        if not is_port_in_use(port):
            return port
    return preferred_port

def start():
    """Launch Uvicorn server for FastAPI and Gradio UI."""
    port = find_available_port(8000)
    print("==================================================")
    print("Starting Telco Churn Prediction Serving Server...")
    print(f"REST API Health Check: http://127.0.0.1:{port}/")
    print(f"REST API Swagger Docs: http://127.0.0.1:{port}/docs")
    print(f"Gradio Web UI:         http://127.0.0.1:{port}/ui")
    print("==================================================")
    uvicorn.run("src.app.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    start()
