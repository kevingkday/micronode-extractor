FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY console_server.py .
COPY static ./static

EXPOSE 8080

CMD ["uvicorn", "console_server:app", "--host", "0.0.0.0", "--port", "8080"]
