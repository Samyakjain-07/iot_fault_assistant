# Multi-stage build. The builder installs dependencies and generates the
# artifacts that are gitignored (trained model, vector store, embedding model),
# so the final image is self-contained and runs on a host with nothing but
# Docker and a .env file.

# ---- builder ---------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# CPU-only torch first: the default PyPI wheel drags in ~2 GB of CUDA libraries
# that are useless on a CPU-only host. sentence-transformers then finds torch
# already satisfied.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model into the Hugging Face cache. Kept as its own
# layer so editing code doesn't trigger a re-download. Name must match
# rag.embedding_model in config/project.yaml.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY config config
COPY data data
COPY src src
COPY scripts scripts
RUN mkdir -p models database \
    && python scripts/train_model.py \
    && python scripts/ingest_manuals.py

# ---- runtime ---------------------------------------------------------------
FROM python:3.12-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # The embedding model is baked in below; never go to the network for it.
    HF_HUB_OFFLINE=1

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY --from=builder /app/models models
COPY --from=builder /app/rag_store rag_store

COPY .streamlit .streamlit
COPY config config
COPY data data
COPY src src
COPY scripts scripts
COPY dashboard dashboard
# database/ is a volume at runtime (shared with Grafana); init_db creates app.db.
RUN mkdir -p database

# Streamlit's port comes from .streamlit/config.toml.
EXPOSE 8502

# Default process is the dashboard; docker-compose overrides `command` to run
# the subscriber and publisher from this same image.
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0"]
