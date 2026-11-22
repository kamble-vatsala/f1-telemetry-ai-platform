# 1. Start from an official, lightweight Linux image with Python 3.12 pre-installed
FROM python:3.12-slim

# 2. Prevent Python from writing .pyc files and make print/log messages show up immediately in Cloud Run logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Create and move into the /app folder inside the Linux container
WORKDIR /app

# 4. Install libgomp1 (required by XGBoost on slim Linux for multi-threaded CPU math)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy ONLY the requirements list first and install libraries
#    (Docker caches this layer so future code tweaks rebuild in seconds!)
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# 6. Copy your application code, modular frontend, and DuckDB lakehouse
#    (Everything listed in .dockerignore is automatically skipped)
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY data/f1_lakehouse.duckdb ./data/f1_lakehouse.duckdb

# 7. Tell Cloud Run our app listens on the port Google assigns (default 8080)
#    0.0.0.0 means "accept traffic from outside the container" (unlike 127.0.0.1)
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]