FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

EXPOSE 5000
EXPOSE 8501