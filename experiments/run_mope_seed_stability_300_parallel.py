from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiment_mope_seed_stability.py"
RUN_ROOT = ROOT / "results" / "result7_mope_seed_stability_300_runs"
SEEDS = [42, 43, 44, 45, 46]


def run_seed(seed: int) -> tuple[int, int]:
    output_dir = RUN_ROOT / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
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
    with (output_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (
        output_dir / "stderr.log"
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
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(SEEDS)) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            seed, returncode = future.result()
            print(f"seed={seed} returncode={returncode}", flush=True)
            if returncode != 0:
                raise RuntimeError(
                    f"Seed {seed} failed; inspect {RUN_ROOT / f'seed_{seed}' / 'stderr.log'}"
                )


if __name__ == "__main__":
    main()
