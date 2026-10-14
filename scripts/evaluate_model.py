"""Evaluate the trained detector against a held-out set of injected faults.

Builds a fresh labelled set -- healthy readings plus readings from every fault
scenario in ``src/mqtt_publisher.py`` -- using a seed different from the
training seed, runs it through :class:`PredictionService`, and prints
precision / recall / F1 with ANOMALY as the positive class.

Note that precision depends on how many healthy readings are in the set
relative to faulty ones; the composition is printed alongside the metrics.

Examples:
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --healthy 5000 --per-fault 250 --seed 99
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# Allow `python scripts/evaluate_model.py` (not just `python -m scripts...`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config  # noqa: E402
from src.mqtt_publisher import FAULT_SCENARIOS, generate_reading  # noqa: E402
from src.prediction_service import ANOMALY, PredictionService  # noqa: E402

DEFAULT_HEALTHY = 2000
DEFAULT_PER_FAULT = 100
# Differs from the training default (ml.random_state) so the set is held out.
DEFAULT_SEED = 2024


def build_eval_set(
    healthy: int, per_fault: int, seed: int
) -> list[tuple[dict, str | None]]:
    """Return ``(reading, fault)`` pairs; ``fault`` is ``None`` for healthy ones."""
    rng = random.Random(seed)
    device_id = get_config().data["project"]["device_id"]
    cases: list[tuple[dict, str | None]] = [
        (generate_reading(device_id, rng), None) for _ in range(healthy)
    ]
    for fault in FAULT_SCENARIOS:
        cases.extend(
            (generate_reading(device_id, rng, fault), fault)
            for _ in range(per_fault)
        )
    return cases


def compute_metrics(actual: list[bool], predicted: list[bool]) -> dict[str, float]:
    """Confusion counts plus precision/recall/F1, ANOMALY (True) as positive.

    Ratios with an empty denominator are reported as 0.0.
    """
    pairs = list(zip(actual, predicted, strict=True))
    tp = sum(a and p for a, p in pairs)
    fp = sum(not a and p for a, p in pairs)
    fn = sum(a and not p for a, p in pairs)
    tn = sum(not a and not p for a, p in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(
    service: PredictionService, healthy: int, per_fault: int, seed: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Run the held-out set; return ``(overall metrics, recall per fault)``."""
    cases = build_eval_set(healthy, per_fault, seed)
    predictions = service.predict_many([reading for reading, _ in cases])
    flagged = [p.label == ANOMALY for p in predictions]
    actual = [fault is not None for _, fault in cases]

    per_fault_recall = {}
    for fault in FAULT_SCENARIOS:
        hits = [
            was_flagged
            for (_, case_fault), was_flagged in zip(cases, flagged, strict=True)
            if case_fault == fault
        ]
        per_fault_recall[fault] = sum(hits) / len(hits) if hits else 0.0
    return compute_metrics(actual, flagged), per_fault_recall


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--healthy",
        type=int,
        default=DEFAULT_HEALTHY,
        help=f"healthy readings in the eval set (default: {DEFAULT_HEALTHY})",
    )
    parser.add_argument(
        "--per-fault",
        type=int,
        default=DEFAULT_PER_FAULT,
        help=f"readings per fault scenario (default: {DEFAULT_PER_FAULT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed for the eval set (default: {DEFAULT_SEED})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Print the evaluation report; returns a process exit code."""
    args = parse_args(argv)
    try:
        service = PredictionService.load()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    metrics, per_fault_recall = evaluate(
        service, args.healthy, args.per_fault, args.seed
    )
    n_faulty = args.per_fault * len(FAULT_SCENARIOS)
    total = args.healthy + n_faulty

    print(
        f"Eval set (seed={args.seed}): {total} readings = {args.healthy} healthy "
        f"+ {n_faulty} faulty ({args.per_fault} x {len(FAULT_SCENARIOS)} scenarios)"
    )
    print("\nConfusion matrix (positive class = ANOMALY)")
    print(f"  TP={metrics['tp']:<5} FP={metrics['fp']:<5}")
    print(f"  FN={metrics['fn']:<5} TN={metrics['tn']:<5}")
    print(f"\nPrecision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    if args.healthy:
        print(f"False-positive rate on healthy: {metrics['fp'] / args.healthy:.3f}")
    print("\nRecall per fault scenario")
    for fault, recall in per_fault_recall.items():
        print(f"  {fault:<18} {recall:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
