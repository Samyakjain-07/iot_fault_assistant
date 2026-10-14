# 04 — Tech Stack Decisions

| Layer | Choice | Why |
|---|---|---|
| MQTT broker | EMQX Cloud (TLS) | Only MQTT platform currently available for this project |
| Database | SQLite | No server to manage, fine at student-project scale; code goes through `src/database.py` so a later swap to PostgreSQL/InfluxDB doesn't ripple |
| Anomaly detection | Isolation Forest (scikit-learn) | Unsupervised — no large labelled fault dataset required |
| Knowledge base | ChromaDB (persistent, local) | Simple, no server to run, same pattern as prior RAG work |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Small, fast, well-proven for this scale |
| LLM | Ollama, local, default `llama3.2:3b` | Free, no per-call API cost, works offline for a demo. Swappable later — see ADR in `CLAUDE.md` |
| Diagnosis interface | Streamlit | Fast to build a conversational chat + operator view |
| Live visualization | Grafana (Docker) | Purpose-built for always-on sensor dashboards |
| Testing | pytest | Standard, simple |
| Linting | ruff | Fast, one consistent style (`pyproject.toml`) |
| CI/CD | GitHub Actions | Runs ruff + pytest on every push/PR before merge |
| Containerization | Docker (+ docker-compose for local dev) | Packages model, vector store and dashboard into one reproducible image |
| Cloud deployment | AWS (ECS or a single EC2 instance) | Reuses the AWS account/IAM already set up from earlier RAG work |

## Deliberately deferred

These are real production concerns, named here so they're not forgotten —
but out of scope for the first build:

- Device certificate management / mutual TLS for a real ESP32 fleet
- A production time-series database (InfluxDB/TimescaleDB) in place of SQLite
- Model monitoring, drift detection, scheduled retraining
- Centralized logging, metrics, alerting on the deployed service itself
- High availability / disaster recovery for the deployed container
