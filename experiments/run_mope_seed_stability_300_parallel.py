"""Optional historical 300-epoch diagnostic, separate from the 16-epoch protocol."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve().with_name("experiment_mope_seed_stability.py")
RUN_ID = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
RUN_ROOT = ROOT / "results" / "diagnostic_300_epochs_fdg_pooled_reference_v1" / RUN_ID
SEEDS = [42, 43, 44, 45, 46]


def run_seed(seed: int) -> tuple[int, int]:
    output_dir = RUN_ROOT / f"seed_{seed}"
    # The child creates its output directory exclusively. Keep logs outside it.
    log_dir = RUN_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT),
        "--seeds",
        str(seed),
        "--epochs",
        "300",
        "--batch-size",
        "128",
        "--output-dir",
        str(output_dir),
    ]
    with (log_dir / f"seed_{seed}_stdout.log").open("x", encoding="utf-8") as stdout, (
        log_dir / f"seed_{seed}_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return seed, completed.returncode


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=len(SEEDS)) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            seed, returncode = future.result()
            print(f"seed={seed} returncode={returncode}", flush=True)
            if returncode != 0:
                raise RuntimeError(
                    f"Seed {seed} failed; inspect {RUN_ROOT / 'logs' / f'seed_{seed}_stderr.log'}"
                )


if __name__ == "__main__":
    main()
