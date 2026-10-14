# 02 — System Architecture

```
                 +----------------------------+
                 | Synthetic Telemetry         |
                 | Generator (Python)          |
                 +--------------+---------------+
                                | MQTT / TLS
                                v
                 +----------------------------+
                 | EMQX Cloud (MQTT Broker)    |
                 +--------------+---------------+
                                |
                                v
                 +----------------------------+
                 | MQTT Subscriber             |
                 +--------------+---------------+
                                |
                 +--------------+---------------+
                 |                              |
                 v                              v
        +----------------+           +------------------------+
        | SQLite          |           | Feature Engineering    |
        | telemetry        |           | (5 features)           |
        | anomaly_events   |           +-----------+-------------+
        | diagnosis_log    |                       |
        +----------------+                        v
                 ^                       +------------------------+
                 |                       | Isolation Forest        |
                 |                       | (anomaly_detector.py)   |
                 |                       +-----------+-------------+
                 |                                   |
                 |                      +------------+-------------+
                 |                      |                          |
                 |                   NORMAL                    ANOMALY
                 |                      |                          |
                 |                      v                          v
                 |             Dashboard Analytics          Alert Manager
                 |                      |                          |
                 |                      |                          v
                 |                      |                   MQTT Anomaly Alert
                 |                      |                          |
                 |                      |                          v
                 |                      |              +------------------------+
                 |                      |              | RAG Retriever           |
                 |                      |              | (Chroma + MiniLM)       |
                 |                      |              +-----------+-------------+
                 |                      |                          |
                 |                      |                          v
                 |                      |              +------------------------+
                 |                      |              | LLM (Ollama, local)     |
                 |                      |              | grounded, cited answer  |
                 |                      |              +-----------+-------------+
                 |                      |                          |
                 +----------------------+--------------------------+
                                        |
                             +----------+-----------+
                             |                       |
                             v                       v
                    Streamlit Dashboard      Grafana Dashboard
                    (diagnosis chat +        (live sensor
                     operator view)           visualization)
```

## Component responsibilities

| Component | Responsibility | Depends on |
|---|---|---|
| Telemetry generator | Create realistic + injectable-fault synthetic readings | `config/project.yaml` |
| MQTT publisher | Serialize + publish telemetry over TLS | EMQX Cloud |
| MQTT subscriber | Ingest, validate, persist, trigger the pipeline | Database, feature engineering |
| Feature engineering | Raw telemetry → 5-feature ML vector | pandas |
| Anomaly detector | Isolation Forest train/predict | scikit-learn |
| Alert manager | Publish anomaly events back over MQTT | MQTT |
| RAG ingest | Chunk + embed `data/manuals/*.md` into Chroma | sentence-transformers |
| RAG retriever | Similarity search for a given anomaly context | Chroma |
| LLM service | Backend-agnostic `generate(prompt) -> str` | Ollama (default) |
| Diagnosis service | Anomaly → retrieve → prompt → grounded, cited answer | Retriever, LLM service |
| Streamlit dashboard | Diagnosis chat + operator view | SQLite, diagnosis_service |
| Grafana | Live telemetry dashboards | SQLite (via the SQLite datasource plugin) |

## Why two dashboards, not one

Streamlit is built for interactive, conversational flows (the diagnosis
chat); Grafana is built for always-on, auto-refreshing time-series
dashboards. Using each for the job it's good at is a design decision worth
stating explicitly in the pitch, not an accident of the tool checklist.

## Hardware-ready note

Same principle as a typical reference IoT pipeline: nothing downstream of
the MQTT broker should care whether a reading came from the Python
simulator or a real ESP32 + sensor array. Keep the telemetry JSON schema
(`docs/05_DATA_SCHEMA.md`) as the only contract the generator has to honor.
