FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install pip-tools

COPY requirements.in .    
COPY . .

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

RUN pip-compile requirements.in -v
RUN pip install -r requirements.txt -v

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["sleep", "infinity"]
