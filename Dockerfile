FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY run.sh /app/run.sh

RUN mkdir -p /app/output /app/data /app/log
RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]
