# 05 — Data Schema

## `telemetry` table

| Field          | Type      | Notes                              |
|----------------|-----------|-------------------------------------|
| id             | INTEGER PK | autoincrement                      |
| device_id      | TEXT       | e.g. `MOTOR-001`                   |
| timestamp      | TEXT       | ISO 8601 UTC                        |
| voltage        | REAL       |                                      |
| current        | REAL       |                                      |
| power_kw       | REAL       |                                      |
| power_factor   | REAL       |                                      |
| temperature    | REAL       |                                      |

## `anomaly_events` table

| Field          | Type      | Notes                                        |
|----------------|-----------|------------------------------------------------|
| id             | INTEGER PK |                                                |
| telemetry_id   | INTEGER FK | -> telemetry.id                               |
| device_id      | TEXT       |                                                |
| timestamp      | TEXT       |                                                |
| prediction     | TEXT       | `NORMAL` \| `ANOMALY`                          |
| anomaly_score  | REAL       | raw Isolation Forest score                     |

## `diagnosis_log` table

| Field               | Type      | Notes                                                |
|---------------------|-----------|---------------------------------------------------------|
| id                  | INTEGER PK |                                                        |
| anomaly_event_id    | INTEGER FK | -> anomaly_events.id                                   |
| timestamp           | TEXT       |                                                          |
| query_text          | TEXT       | what was sent to the retriever                          |
| retrieved_chunk_ids | TEXT       | JSON list of chunk ids used as context                  |
| llm_answer          | TEXT       | the plain-English diagnosis returned to the technician  |
| confidence_flag     | TEXT       | `grounded` \| `low_confidence` (see diagnosis_service)   |

`confidence_flag` is set to `low_confidence` when retrieval similarity falls
below a configured threshold — the answer should say so rather than present
a weakly-grounded guess as settled fact.

## ML feature set

Five features, matching the `telemetry` columns used for detection:

```
voltage, current, power_kw, power_factor, temperature
```

Isolation Forest raw output: `-1` → `ANOMALY`, `1` → `NORMAL` (converted to
the application-level strings above before being stored or displayed).

## Knowledge-base chunks

`data/manuals/*.md` is split into chunks (target ~300–500 tokens, by
section heading first, then by size) and embedded with
`sentence-transformers/all-MiniLM-L6-v2`. Each chunk stored in Chroma keeps
metadata: `source_file`, `section_heading`, `fault_type` (when identifiable
from the heading) — this metadata is what lets the dashboard show "Source:
motor_fault_reference.md § High Temperature" next to an answer.
