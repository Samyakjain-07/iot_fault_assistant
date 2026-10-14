"""Run the synthetic telemetry publisher against EMQX Cloud.

Examples:
    python scripts/generate_telemetry.py
    python scripts/generate_telemetry.py --count 50 --interval 1
    python scripts/generate_telemetry.py --fault high_temperature --fault-after 20
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow `python scripts/generate_telemetry.py` (not just `python -m scripts...`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_config  # noqa: E402
from src.mqtt_publisher import FAULT_SCENARIOS, run_publisher  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--interval",
        type=float,
        help="seconds between readings (default: project.yaml value)",
    )
    parser.add_argument(
        "--count", type=int, help="stop after this many readings (default: run forever)"
    )
    parser.add_argument(
        "--fault",
        choices=FAULT_SCENARIOS,
        help="fault scenario to inject (default: none, healthy readings only)",
    )
    parser.add_argument(
        "--fault-after",
        type=int,
        default=0,
        help="number of healthy readings before the fault begins (default: 0)",
    )
    parser.add_argument(
        "--fault-count",
        type=int,
        help="how many readings the fault lasts (default: until the run ends)",
    )
    parser.add_argument(
        "--seed", type=int, help="seed the generator for repeatable data"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the publisher; returns a process exit code."""
    args = parse_args(argv)
    config = get_config()
    logging.basicConfig(
        level=config.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        sent = run_publisher(
            config,
            interval=args.interval,
            count=args.count,
            fault=args.fault,
            fault_after=args.fault_after,
            fault_count=args.fault_count,
            seed=args.seed,
        )
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("interrupted")
        return 0
    except ConnectionError as exc:
        logging.getLogger(__name__).error("%s", exc)  # noqa: TRY400
        return 1
    logging.getLogger(__name__).info("done, published %d readings", sent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
