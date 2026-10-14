"""Anomaly -> retrieve -> prompt -> grounded, cited answer -> ``diagnosis_log``.

The model is only ever shown the manual passages the retriever returned and is
told to answer from them alone and to cite the section it used. The service
then checks the answer really does cite one of those sections and compares the
retrieval similarity to ``rag.low_confidence_threshold``; if either check
fails the row is flagged ``low_confidence`` and the answer says so.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_config
from src.database import insert_diagnosis
from src.llm_service import generate
from src.rag.retriever import RetrievedChunk, retrieve

GROUNDED = "grounded"
LOW_CONFIDENCE = "low_confidence"

# Healthy bounds from the "Device Fact Sheet" / fault sections of the manual.
HIGH_VOLTAGE = 410.0
LOW_VOLTAGE = 380.0
RATED_CURRENT = 15.0
HIGH_TEMPERATURE = 65.0
LOW_POWER_FACTOR = 0.90

NO_CONTEXT_ANSWER = (
    "No relevant passage was found in the fault reference, so no diagnosis can "
    "be given. Inspect the equipment manually."
)

PROMPT_TEMPLATE = """\
You are a diagnostic assistant for an industrial induction motor. A monitoring \
system flagged the telemetry reading below as anomalous.

RULES
1. Answer ONLY from the REFERENCE PASSAGES below. Do not use outside knowledge \
and do not invent causes, numbers or procedures.
2. If the passages do not explain the readings, say exactly: "The fault \
reference does not cover this." and stop.
3. Name the readings that led to your conclusion.
4. End with one line per passage you used, copied exactly from the passage \
header, in the form: Source: <file> § <section>
5. Keep it short, in plain English, for a maintenance technician.

ANOMALOUS READING (device {device_id} at {timestamp})
{reading_lines}

REFERENCE PASSAGES
{passages}

Answer:"""


@dataclass(frozen=True)
class Diagnosis:
    """The outcome of one diagnosis, as stored in ``diagnosis_log``."""

    diagnosis_id: int
    anomaly_event_id: int
    query_text: str
    chunks: list[RetrievedChunk]
    answer: str
    confidence_flag: str
    cited_sections: list[str]


def build_query(reading: Mapping[str, Any]) -> str:
    """Describe the abnormal signals in the manual's own words, for retrieval.

    Raw numbers embed poorly, so each out-of-range signal becomes a short
    symptom phrase. If nothing is individually out of range (the detector
    flagged an unusual combination) a generic query is used.
    """
    symptoms: list[str] = []
    if reading["voltage"] > HIGH_VOLTAGE:
        symptoms.append("supply voltage above 400V, suspect the grid or transformer")
    elif reading["voltage"] < LOW_VOLTAGE:
        symptoms.append("voltage sag below 380V when other loads start on the circuit")
    if reading["current"] > RATED_CURRENT:
        symptoms.append("current above the 15A rating, bearing seized or jammed load")
    if reading["temperature"] > HIGH_TEMPERATURE:
        symptoms.append("motor running hot, temperature above 65°C")
    if reading["power_factor"] < LOW_POWER_FACTOR:
        symptoms.append("power factor dropped below 0.90, motor lightly loaded")
    if not symptoms:
        return "motor readings anomalous, multiple signals appear together"
    return "Motor fault: " + "; ".join(symptoms)


def _format_reading(reading: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"- voltage: {reading['voltage']} V",
            f"- current: {reading['current']} A",
            f"- power: {reading['power_kw']} kW",
            f"- power factor: {reading['power_factor']}",
            f"- temperature: {reading['temperature']} °C",
        ]
    )


def build_prompt(reading: Mapping[str, Any], chunks: Sequence[RetrievedChunk]) -> str:
    """Fill the grounding prompt with the reading and the retrieved passages."""
    passages = "\n\n".join(
        f"[{i}] Source: {c.source} § {c.section}\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )
    return PROMPT_TEMPLATE.format(
        device_id=reading["device_id"],
        timestamp=reading["timestamp"],
        reading_lines=_format_reading(reading),
        passages=passages,
    )


def cited_sections(answer: str, chunks: Sequence[RetrievedChunk]) -> list[str]:
    """Sections of the retrieved ``chunks`` that ``answer`` names, in rank order."""
    text = answer.lower()
    found: list[str] = []
    for chunk in chunks:
        if chunk.section.lower() in text and chunk.section not in found:
            found.append(chunk.section)
    return found


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def diagnose(
    reading: Mapping[str, Any],
    anomaly_event_id: int,
    db_path: str | Path | None = None,
    store_path: Path | None = None,
    generate_fn: Callable[[str], str] | None = None,
    retrieve_fn: Callable[..., list[RetrievedChunk]] | None = None,
) -> Diagnosis:
    """Diagnose one flagged reading and log it to ``diagnosis_log``.

    ``generate_fn`` / ``retrieve_fn`` default to the real LLM and retriever;
    pass stubs to test without a model or vector store. Raises
    ``LLMError`` if the LLM fails and ``RuntimeError`` if the vector store has
    not been built -- in both cases nothing is logged, because there is no
    grounded answer to record.
    """
    generate_fn = generate_fn or generate
    retrieve_fn = retrieve_fn or retrieve
    threshold = get_config().data["rag"]["low_confidence_threshold"]

    query = build_query(reading)
    chunks = retrieve_fn(query, store_path=store_path)

    if not chunks:
        answer, flag, cited = NO_CONTEXT_ANSWER, LOW_CONFIDENCE, []
    else:
        answer = generate_fn(build_prompt(reading, chunks)).strip()
        cited = cited_sections(answer, chunks)
        notes: list[str] = []
        best = max(c.score for c in chunks)
        if best < threshold:
            notes.append(
                f"the closest manual section matched weakly (similarity "
                f"{best:.2f}, below {threshold:.2f})"
            )
        if not cited:
            notes.append("the answer does not cite any retrieved manual section")
        flag = LOW_CONFIDENCE if notes else GROUNDED
        if notes:
            answer = (
                f"LOW CONFIDENCE: {'; '.join(notes)}. Treat this as a hypothesis "
                f"to verify on-site.\n\n{answer}"
            )

    diagnosis_id = insert_diagnosis(
        anomaly_event_id,
        _utc_now(),
        query,
        [c.chunk_id for c in chunks],
        answer,
        flag,
        db_path,
    )
    return Diagnosis(
        diagnosis_id=diagnosis_id,
        anomaly_event_id=anomaly_event_id,
        query_text=query,
        chunks=list(chunks),
        answer=answer,
        confidence_flag=flag,
        cited_sections=cited,
    )
