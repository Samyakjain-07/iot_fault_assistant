# Knowledge-Driven IoT Fault Diagnosis Assistant

An AI copilot that watches simulated factory equipment telemetry, detects
abnormal readings, and explains the likely cause in plain English — grounded
in the equipment's own fault-reference documentation via RAG, with a citation,
not a guess.

Built for a college "Startup" subject project. See `CLAUDE.md` for how this
repository is built with Claude Code, and `docs/` for the full brief,
architecture, roadmap and tech-stack decisions.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp .env.example .env              # then fill in your EMQX Cloud credentials
```

See `docs/03_ROADMAP.md` for the build order — this repo is meant to be built
phase by phase with Claude Code, not all at once.

## Running the dashboards

Needs Docker, a filled-in `.env`, a trained model (`models/isolation_forest.joblib`),
a built vector store (`rag_store/`) and Ollama running on the host with
`llama3.2:3b` pulled.

```bash
docker compose up --build
```

This starts the simulated motor (publisher), the subscriber that stores and
scores its readings, and two dashboards:

- Streamlit (live view, anomaly list, diagnosis chat): http://localhost:8502
  (set `APP_PORT` to publish it on a different host port)
- Grafana (Motor Telemetry dashboard, login `admin` / `admin`): http://localhost:3000

Without Docker: `streamlit run dashboard/app.py`, with the subscriber
(`python -m src.mqtt_subscriber`) and publisher
(`python scripts/generate_telemetry.py`) running in other terminals.

## Status

Phases 1-9 are built (data foundation through the dashboards). Phase 10
(containerize and deploy to AWS) is next.
