FROM python:3.11-slim

# Install system dependencies needed for polling GPU stats
RUN apt-get update && apt-get install -y pciutils procps && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir requests

COPY telemetry.py .

CMD ["python", "telemetry.py"]
