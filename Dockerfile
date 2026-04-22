# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="GROUP1 MIT807"
LABEL description="Telecom Customer Churn Predictor — Streamlit App"

# ── System deps (slim image needs these for some Python packages) ──────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker caches this layer unless deps change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY . .

# ── Train & save the model at build time ──────────────────────────────────────
# This bakes the model/rf_pipeline.pkl into the image so no training
# is needed at runtime on Render.
RUN python train_model.py

# ── Expose Render's default port ─────────────────────────────────────────────
EXPOSE 10000

# ── Streamlit config for headless / containerised environments ────────────────
ENV STREAMLIT_SERVER_PORT=10000 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# ── Launch ────────────────────────────────────────────────────────────────────
CMD ["streamlit", "run", "app.py", \
     "--server.port=10000", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
