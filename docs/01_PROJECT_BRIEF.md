# 01 — Project Brief

## Problem

Unplanned downtime is expensive, and most of that cost is diagnosis time: a
technician sees an abnormal reading and has to manually cross-check manuals
and past fault logs before they can even start the fix.

## Solution

A system that does two things continuously:

1. **Detects** — watches live (simulated) sensor telemetry and flags
   abnormal readings using an unsupervised anomaly model.
2. **Explains** — when something is flagged, retrieves the matching passage
   from the equipment's fault-reference documentation and has an LLM turn it
   into a plain-English diagnosis, with a citation back to the source
   passage (RAG — retrieval before generation, not a hallucinated guess).

## Objectives

- Working MQTT telemetry pipeline (EMQX Cloud, TLS) carrying simulated
  factory equipment readings.
- Persistent telemetry + anomaly history in SQLite.
- An Isolation Forest anomaly detector evaluated against injected fault
  scenarios (precision/recall/F1, not just "it runs").
- A small but real knowledge base (seed content in `data/manuals/`) chunked
  and embedded into a vector store.
- A reasoning layer that only answers from retrieved passages and always
  cites what it retrieved.
- Two dashboards with different jobs: Streamlit for the diagnosis chat +
  operator view, Grafana for always-on live sensor visualization.
- The whole thing containerized (Docker) and deployed (AWS), with CI
  (GitHub Actions running pytest + ruff) and a test suite.

## Non-goals (for this project)

- No physical hardware/sensors — synthetic telemetry only, built so an
  ESP32 could later replace the generator without touching anything
  downstream of MQTT.
- Not a production-certified safety system — this is a decision-support
  tool, not something that should directly control machinery.
- Not trying to cover every possible fault type — depth on one equipment
  type (the induction motor / energy-meter domain in
  `data/manuals/motor_fault_reference.md`) beats shallow breadth across many.

## Why this is a reasonable "startup" pitch

Industrial predictive maintenance is a fast-growing software category, and
the specific pain this targets — hours of manual diagnosis time after an
anomaly is flagged — is the gap between "we detected something's wrong" (what
most student IoT projects stop at) and "here's what's wrong and how to fix
it" (what this project adds on top).
