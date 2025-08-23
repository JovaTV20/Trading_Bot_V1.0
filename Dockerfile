# TradingBot Docker Image
# Multi-stage build für kleineres finales Image

# Build Stage
FROM python:3.10-slim as builder

# Build-Dependencies installieren
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production Stage
FROM python:3.10-slim

# Metadata
LABEL maintainer="TradingBot Team <contact@tradingbot.com>"
LABEL version="1.0.0"
LABEL description="Professional Trading Bot for Alpaca Markets"

# System-Dependencies für Production
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Non-root User erstellen
RUN groupadd -r trader && useradd -r -g trader trader

# Working Directory
WORKDIR /app

# Python Dependencies von Builder kopieren
COPY --from=builder /root/.local /home/trader/.local
ENV PATH=/home/trader/.local/bin:$PATH

# Application Code kopieren
COPY . .

# Verzeichnisse erstellen und Permissions setzen
RUN mkdir -p logs data models \
    && chown -R trader:trader /app \
    && chmod +x main.py run_validation.py

# Switch zu non-root user
USER trader

# Environment Variables (Defaults)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app'); \
    from core.controller import TradingController; \
    import json; config=json.load(open('config.json')); \
    controller=TradingController(config); \
    status=controller.get_status(); \
    exit(0 if status.get('is_running') else 1)" || exit 1

# Expose Dashboard Port
EXPOSE 5000

# Volume für persistente Daten
VOLUME ["/app/logs", "/app/data", "/app/models"]

# Default Command - kann überschrieben werden
CMD ["python", "main.py", "--mode", "live", "--symbol", "AAPL"]

# Alternative Commands:
# docker run tradingbot python main.py --mode backtest --symbol TSLA
# docker run -p 5000:5000 tradingbot python dashboard/app.py