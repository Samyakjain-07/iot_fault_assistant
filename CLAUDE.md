# CLAUDE.md

This file is read automatically at the start of every Claude Code session in this
repository. It is the single source of truth for how this project is built —
read `docs/` for the reasoning behind these decisions, but treat this file as
binding.

## Project

**Knowledge-Driven IoT Fault Diagnosis Assistant** — an AI copilot that watches
simulated factory equipment telemetry, detects abnormal readings, and explains
the likely cause in plain English by retrieving the relevant passage from the
equipment's fault-reference documentation (RAG), not by guessing.

Full context: `docs/01_PROJECT_BRIEF.md`, `docs/02_ARCHITECTURE.md`.

## Current phase

Check `docs/03_ROADMAP.md` for the phase list and its "Definition of done" per
phase. **Implement only the phase you are told to work on** — do not jump
ahead to later phases (e.g. do not wire the LLM layer while working on the
anomaly detector) unless explicitly asked. Each phase should leave the repo in
a state where `pytest` and `ruff check .` both pass clean.

## Repository layout

```
src/
  config.py              # loads config/project.yaml + .env
  mqtt_publisher.py       # synthetic telemetry generator -> EMQX Cloud
  mqtt_subscriber.py      # ingests telemetry, orchestrates the pipeline
  database.py              # SQLite access: telemetry, anomaly_events, diagnosis_log
  feature_engineering.py   # raw telemetry dict -> ML feature vector
  anomaly_detector.py      # Isolation Forest train/predict
  prediction_service.py    # app-level interface over anomaly_detector
  alert_manager.py          # publishes anomaly alerts back over MQTT
  rag/
    ingest.py              # chunk + embed data/manuals/*.md into Chroma
    retriever.py            # similarity search over the vector store
  llm_service.py            # wraps the LLM backend (see ADR below)
  diagnosis_service.py      # anomaly -> retrieve -> prompt -> grounded answer
  utils.py
dashboard/
  app.py                   # Streamlit: live telemetry + diagnosis chat
scripts/
  generate_telemetry.py     # standalone run of the synthetic generator
  train_model.py             # trains and serializes the Isolation Forest
  ingest_manuals.py          # one-off / re-run to (re)build the vector store
  evaluate_model.py          # precision/recall/F1 against injected faults
tests/                      # mirrors src/, one test file per module
grafana/provisioning/        # datasource + dashboard JSON for the Grafana container
data/manuals/                 # seed knowledge-base content (markdown, committed)
config/project.yaml           # non-secret configuration
.env                           # secrets — gitignored, never commit
models/                        # serialized model artifacts — gitignored
```

## Tech stack (do not substitute without asking)

- **MQTT broker:** EMQX Cloud, TLS. No local Mosquitto — only EMQX is
  available for this project.
- **Database:** SQLite (`database/app.db`) for the student-scale demo. Code
  that touches the DB should go through `src/database.py` so swapping to
  PostgreSQL/InfluxDB later doesn't ripple elsewhere.
- **Anomaly detection:** `scikit-learn` Isolation Forest, unsupervised,
  5-feature set (see `docs/05_DATA_SCHEMA.md`).
- **Knowledge base / RAG:** ChromaDB (persistent, local) +
  `sentence-transformers/all-MiniLM-L6-v2` embeddings.
- **LLM backend:** Ollama, local, default model `llama3.2:3b` — see
  "ADR: LLM backend" below before changing this.
- **Dashboards:** Streamlit (`dashboard/app.py`) for the diagnosis chat +
  live view; Grafana (via Docker) reading the same SQLite file for the
  always-on sensor dashboards the brief asks for.
- **Testing / lint:** `pytest`, `ruff` (config in `pyproject.toml`).
- **CI/CD:** GitHub Actions — runs `ruff check .` then `pytest` on every push
  and PR to `main`.
- **Containerization:** Docker; `docker-compose.yml` runs the app + Grafana
  together for local dev.
- **Cloud deployment:** AWS (ECS or a single EC2 instance running the Docker
  image) — see `docs/04_TECH_STACK.md`.

### ADR: LLM backend

Default is **Ollama, local** (`ollama pull llama3.2:3b`), because it is free,
needs no per-call API key, and keeps the whole pipeline runnable offline for
a demo. `LLM_BACKEND` in `.env` controls this (`ollama` is the only
implemented value right now). If the laptop running the demo can't handle a
local model, the documented fallback is AWS Bedrock (there is already an AWS
account and IAM user set up from earlier RAG work) — do not implement that
path unless asked; just keep `llm_service.py`'s interface backend-agnostic
(a `generate(prompt) -> str` function) so swapping later is a contained
change, not a rewrite.

## MQTT topics

```
factory/iot/telemetry   # every simulated reading, JSON, QoS 1
factory/iot/status       # publisher/subscriber heartbeat
factory/iot/anomaly      # published when the detector flags a reading
factory/iot/diagnosis    # published once the RAG+LLM layer has an answer
```

## Conventions

- Config split: non-secret values in `config/project.yaml`, secrets in
  `.env` (never committed — `.env.example` is the template). Load both
  through `src/config.py`; nothing else should read `os.environ` directly.
- Every new `src/` module gets a matching `tests/test_<module>.py` in the
  same change — do not leave a phase "done" with untested code.
- Prefer the simplest thing that passes the phase's definition of done. This
  is a student project on a deadline, not a production system — don't add
  abstractions (plugin systems, generic adapters, config-driven everything)
  that nothing in `docs/03_ROADMAP.md` asks for.
- Use type hints and docstrings on public functions; keep functions small
  enough to unit test without mocking half the module.
- Before marking a phase complete: run `ruff check .` and `pytest -v`, fix
  what they flag, and summarize what changed against that phase's
  definition of done.
