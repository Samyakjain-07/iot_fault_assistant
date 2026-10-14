"""Train the Isolation Forest on synthetic healthy telemetry and save it.

The model learns what normal looks like from readings produced by the same
generator the publisher uses (no faults injected), so no MQTT connection or
database is needed. Output goes to ``ml.model_path`` in ``config/project.yaml``.

Examples:
    python scripts/train_model.py
    python scripts/train_model.py --samples 10000 --seed 1
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Allow `python scripts/train_model.py` (not just `python -m scripts...`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.anomaly_detector import save_model, train  # noqa: E402
from src.config import get_config  # noqa: E402
from src.feature_engineering import build_feature_matrix  # noqa: E402
from src.mqtt_publisher import generate_reading  # noqa: E402

DEFAULT_SAMPLES = 5000


def generate_training_readings(samples: int, seed: int) -> list[dict]:
    """Healthy (fault-free) telemetry readings for fitting the detector."""
    rng = random.Random(seed)
    device_id = get_config().data["project"]["device_id"]
    return [generate_reading(device_id, rng) for _ in range(samples)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
        help=f"healthy readings to train on (default: {DEFAULT_SAMPLES})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed for the training data and the forest (default: ml.random_state)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train and save the model; returns a process exit code."""
    args = parse_args(argv)
    ml = get_config().data["ml"]
    seed = args.seed if args.seed is not None else ml["random_state"]

    readings = generate_training_readings(args.samples, seed)
    model = train(build_feature_matrix(readings), random_state=seed)
    path = save_model(model)

    print(
        f"Trained IsolationForest on {args.samples} healthy readings "
        f"(contamination={ml['contamination']}, seed={seed})"
    )
    print(f"Saved model to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
