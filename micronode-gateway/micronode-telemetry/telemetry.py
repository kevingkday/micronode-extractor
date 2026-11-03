import time
import json
import subprocess
import requests

DASHBOARD_URL = "https://api.oceanfalls.com/telemetry"
NODE_ID = "micronode-alpha-01"
API_KEY = "micronode-telemetry-secret"

def get_gpu_telemetry():
    try:
        # Query nvidia-smi for specific telemetry data (CSV format)
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the CSV output
        data = result.stdout.strip().split(", ")
        return {
            "temp_c": int(data[0]),
            "gpu_util_percent": int(data[1]),
            "vram_used_mb": int(data[2]),
            "vram_total_mb": int(data[3])
        }
    except Exception as e:
        print(f"Failed to fetch GPU telemetry: {e}")
        return None

def push_telemetry():
    while True:
        telemetry = get_gpu_telemetry()
        if telemetry:
            payload = {
                "node_id": NODE_ID,
                "timestamp": int(time.time()),
                "metrics": telemetry
            }
            try:
                requests.post(
                    DASHBOARD_URL,
                    json=payload,
                    headers={"Authorization": f"Bearer {API_KEY}"},
                    timeout=5
                )
                print(f"Pushed: {json.dumps(payload)}")
            except requests.exceptions.RequestException:
                print("Failed to reach telemetry dashboard.")
                
        # Poll every 60 seconds
        time.sleep(60)

if __name__ == "__main__":
    print(f"Starting Zero-Trust Telemetry Daemon for {NODE_ID}...")
    push_telemetry()