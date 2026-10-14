"""Streamlit dashboard: live telemetry, anomaly list, and the diagnosis chat.

Run with ``streamlit run dashboard/app.py``. It only reads what the MQTT
subscriber has written to SQLite; the one thing it computes itself is a new
diagnosis when the user asks for one.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts this file's folder on sys.path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import get_config  # noqa: E402
from src.database import (  # noqa: E402
    get_anomaly_events,
    get_diagnoses,
    get_telemetry,
    get_telemetry_by_id,
    init_db,
)

# (column, label) for the four time-series charts.
SIGNALS = (
    ("voltage", "Voltage (V)"),
    ("current", "Current (A)"),
    ("power_kw", "Power (kW)"),
    ("temperature", "Temperature (°C)"),
)
ANOMALY_LIMIT = 500
CONFIDENCE_LABELS = {"grounded": "grounded", "low_confidence": "LOW CONFIDENCE"}


def telemetry_frame(rows: list[dict]) -> pd.DataFrame:
    """Telemetry rows (newest first, as the database returns them) as a frame
    ordered oldest first with a datetime ``timestamp`` column."""
    frame = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    return frame


def signal_figure(
    frame: pd.DataFrame, column: str, label: str, anomaly_telemetry_ids: set[int]
) -> go.Figure:
    """Line chart of one signal over time, flagged readings marked in red."""
    figure = go.Figure(go.Scatter(x=frame["timestamp"], y=frame[column], name=label))
    flagged = frame[frame["id"].isin(anomaly_telemetry_ids)]
    if not flagged.empty:
        figure.add_scatter(
            x=flagged["timestamp"],
            y=flagged[column],
            mode="markers",
            name="anomaly",
            marker={"color": "red", "size": 9},
        )
    figure.update_layout(
        title=label,
        height=280,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        showlegend=False,
    )
    return figure


def anomaly_table(events: list[dict], diagnoses: list[dict]) -> pd.DataFrame:
    """One row per anomaly event, with the status of its latest diagnosis."""
    latest: dict[int, str] = {}
    for diagnosis in diagnoses:  # newest first, so the first one seen is latest
        latest.setdefault(
            diagnosis["anomaly_event_id"],
            CONFIDENCE_LABELS[diagnosis["confidence_flag"]],
        )
    return pd.DataFrame(
        {
            "event": [e["id"] for e in events],
            "time (UTC)": [e["timestamp"] for e in events],
            "device": [e["device_id"] for e in events],
            "score": [round(e["anomaly_score"], 3) for e in events],
            "diagnosis": [latest.get(e["id"], "pending") for e in events],
        }
    )


@st.fragment(run_every=get_config().data["dashboard"]["refresh_seconds"])
def live_section(device_id: str, points: int) -> None:
    """Latest readings, the four signal charts and the anomaly list.

    A fragment, so it refreshes on a timer without resetting the chat below.
    """
    rows = get_telemetry(limit=points, device_id=device_id)
    if not rows:
        st.info(
            "No telemetry yet. Start the subscriber "
            "(`python -m src.mqtt_subscriber`) and the publisher "
            "(`python scripts/generate_telemetry.py`)."
        )
        return

    frame = telemetry_frame(rows)
    events = get_anomaly_events(limit=ANOMALY_LIMIT)
    flagged_ids = {e["telemetry_id"] for e in events}

    latest, previous = frame.iloc[-1], frame.iloc[-2] if len(frame) > 1 else None
    st.caption(f"Latest reading: {latest['timestamp']:%Y-%m-%d %H:%M:%S} UTC")
    tiles = st.columns(5)
    for tile, (column, label) in zip(
        tiles, [*SIGNALS, ("power_factor", "Power factor")], strict=True
    ):
        delta = None
        if previous is not None:
            delta = round(latest[column] - previous[column], 2)
        tile.metric(label, round(latest[column], 2), delta)

    left, right = st.columns(2)
    for i, (column, label) in enumerate(SIGNALS):
        (left if i % 2 == 0 else right).plotly_chart(
            signal_figure(frame, column, label, flagged_ids),
            width="stretch",
            key=f"chart_{column}",
        )

    st.subheader("Anomalies")
    if events:
        st.dataframe(
            anomaly_table(events, get_diagnoses(limit=ANOMALY_LIMIT)),
            hide_index=True,
            width="stretch",
        )
    else:
        st.write("No anomalies detected yet.")


def show_diagnosis(diagnosis: dict) -> None:
    """Render one logged diagnosis as an assistant chat message, with the
    manual passages it was grounded on (the ones it cited start expanded)."""
    from src.rag.retriever import get_chunks

    answer = diagnosis["llm_answer"]
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.caption(
            f"{CONFIDENCE_LABELS[diagnosis['confidence_flag']]} · "
            f"{diagnosis['timestamp']} · retrieval query: {diagnosis['query_text']}"
        )
        try:
            passages = get_chunks(diagnosis["retrieved_chunk_ids"])
        except RuntimeError as exc:
            st.warning(f"Source passages unavailable: {exc}")
            return
        for passage in passages:
            cited = passage["section"].lower() in answer.lower()
            label = f"{passage['source']} § {passage['section']}"
            with st.expander(label + (" (cited)" if cited else ""), expanded=cited):
                st.markdown(passage["text"])


def run_diagnosis(reading: dict, event_id: int) -> bool:
    """Diagnose one anomaly and log it; show the error and return False if the
    LLM or the vector store is unavailable."""
    from src.diagnosis_service import diagnose

    try:
        with st.spinner("Retrieving manual passages and asking the model..."):
            diagnose(reading, event_id)
    except (RuntimeError, ValueError) as exc:
        st.error(f"Diagnosis failed: {exc}")
        return False
    return True


def chat_section() -> None:
    """Pick an anomaly, read its logged diagnoses, or request a new one."""
    st.subheader("Diagnosis chat")
    events = get_anomaly_events(limit=ANOMALY_LIMIT)
    if not events:
        st.write("Nothing to diagnose yet: no anomalies have been detected.")
        return

    by_id = {e["id"]: e for e in events}
    event_id = st.selectbox(
        "Anomaly",
        list(by_id),
        format_func=lambda i: f"#{i} · {by_id[i]['timestamp']}",
    )
    event = by_id[event_id]
    reading = get_telemetry_by_id(event["telemetry_id"])
    if reading is None:
        st.error(f"Telemetry row {event['telemetry_id']} for this anomaly is missing.")
        return

    with st.chat_message("user"):
        st.markdown(
            f"What caused anomaly #{event_id}?  \n"
            f"{reading['voltage']} V · {reading['current']} A · "
            f"{reading['power_kw']} kW · PF {reading['power_factor']} · "
            f"{reading['temperature']} °C"
        )

    diagnoses = get_diagnoses(anomaly_event_id=event_id)
    for diagnosis in reversed(diagnoses):  # oldest first, like a transcript
        show_diagnosis(diagnosis)
    if not diagnoses:
        st.caption("No diagnosis logged yet for this anomaly.")

    if st.button("Diagnose again" if diagnoses else "Diagnose this anomaly") and (
        run_diagnosis(reading, event_id)
    ):
        st.rerun()


def main() -> None:
    """Lay out the page."""
    config = get_config()
    init_db()
    title = config.data["dashboard"]["title"]
    st.set_page_config(page_title=title, layout="wide")
    st.title(title)

    device_id = config.data["project"]["device_id"]
    with st.sidebar:
        st.header(device_id)
        points = st.slider("Readings shown", 20, 500, 100, step=10)
        refresh = config.data["dashboard"]["refresh_seconds"]
        st.caption(f"Live view refreshes every {refresh} s.")

    live_section(device_id, points)
    chat_section()


main()
