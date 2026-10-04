FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    freetds-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY graph_analytics ./graph_analytics
COPY main.py .

ENV INPUT_DIR=/data/input
ENV OUTPUT_DIR=/data/output

VOLUME ["/data/input", "/data/output"]

ENTRYPOINT ["python", "main.py"]
CMD ["pipeline", "excel"]
