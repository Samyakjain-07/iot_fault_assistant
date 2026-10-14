# 03 — Roadmap

Seven phases over ten weeks. Build in order — each phase's definition of
done should hold before starting the next one. `pytest` and `ruff check .`
should pass clean at the end of every phase.

## Phase 1 — Project & DevOps setup (Week 1)

**Build:** `src/config.py` (loads `config/project.yaml` + `.env`),
`src/database.py` with the three tables from `docs/05_DATA_SCHEMA.md` and a
function to initialize the DB file, a minimal `Dockerfile` that installs
`requirements.txt` and can run `streamlit run dashboard/app.py` (dashboard
doesn't need to do anything yet — a placeholder page is fine), and
`.github/workflows/ci.yml` wired to actually run `ruff check .` then
`pytest`.

**Definition of done:** `python -c "from src.database import init_db;
init_db()"` creates the SQLite file with all three tables; CI goes green on
a push; `docker build` succeeds locally.

## Phase 2 — Data foundation (Week 2)

**Build:** `src/mqtt_publisher.py` (connects to EMQX Cloud over TLS,
publishes synthetic telemetry on `factory/iot/telemetry` at a configurable
interval, can inject the fault scenarios named in
`docs/05_DATA_SCHEMA.md`/the AI-ML design), `src/mqtt_subscriber.py`
(subscribes, validates, writes to the `telemetry` table), `scripts/
generate_telemetry.py` as a runnable entry point.

**Definition of done:** running the publisher and subscriber in two
terminals results in rows accumulating in the `telemetry` table;
`tests/test_database.py` covers insert/read.

## Phase 3-4 — Anomaly detection (Weeks 3-4)

**Build:** `src/feature_engineering.py` (telemetry dict → 5-feature
vector), `src/anomaly_detector.py` (train + predict, serializes to
`models/isolation_forest.joblib`), `src/prediction_service.py`
(app-level NORMAL/ANOMALY interface), `scripts/train_model.py`,
`scripts/evaluate_model.py` (precision/recall/F1 against a held-out set of
injected-fault telemetry).

**Definition of done:** `scripts/evaluate_model.py` prints real
precision/recall numbers (not just "it ran"); `tests/
test_feature_engineering.py` and `tests/test_anomaly_detector.py` pass;
the subscriber pipeline now writes to `anomaly_events` when it flags a
reading.

## Phase 5-6 — Knowledge base (Weeks 5-6)

**Build:** `src/rag/ingest.py` (chunks `data/manuals/*.md`, embeds with
MiniLM, writes to a persistent Chroma store), `src/rag/retriever.py`
(similarity search, returns chunks + metadata), `scripts/
ingest_manuals.py` as the runnable entry point.

**Definition of done:** a test query like "motor running hot" returns the
correct section of `data/manuals/motor_fault_reference.md` in the top 3
results; retrieval is unit-tested with a known query/expected-chunk pair.

## Phase 7-8 — Reasoning layer (Weeks 7-8)

**Build:** `src/llm_service.py` (Ollama backend, see the ADR in
`CLAUDE.md`), `src/diagnosis_service.py` (anomaly event → retrieve →
prompt template that forces the model to answer only from retrieved
context and cite the section it used → write to `diagnosis_log`), wire
this into the subscriber so an `ANOMALY` prediction triggers it
automatically.

**Definition of done:** an injected high-temperature fault produces a
`diagnosis_log` row whose `llm_answer` references the correct fault type
and whose `retrieved_chunk_ids` match what `rag/retriever.py` actually
returned (i.e. no ungrounded free-text answers); `tests/
test_prediction_service.py`-style coverage for the diagnosis service with
a stubbed/mocked LLM call (don't hit a real model in CI).

## Phase 9 — Interface (Week 9)

**Build:** `dashboard/app.py` (Streamlit — live telemetry view, anomaly
list, and a chat-style box to ask about a given anomaly, showing the
cited source passage), `grafana/provisioning/datasources/` +
`grafana/provisioning/dashboards/` (a datasource pointing at the SQLite
file via the SQLite plugin, and a starter dashboard with voltage/current/
power/temperature panels over time).

**Definition of done:** `streamlit run dashboard/app.py` shows live data
and a working diagnosis chat; `docker compose up` brings up the app +
Grafana together and the Grafana dashboard renders real telemetry.

## Phase 10 — Containerize & deploy (Week 10)

**Build:** finalize the `Dockerfile` (multi-stage if it helps image size),
confirm `docker-compose.yml` runs the full stack, add a deploy step/notes
for pushing the image and running it on AWS (ECS task definition or a
plain EC2 + `docker run`), finalize `README.md` with real run
instructions.

**Definition of done:** the containerized app runs end-to-end on AWS and is
reachable at a URL; CI still green; README accurately describes how to run
both locally and how it's deployed.
