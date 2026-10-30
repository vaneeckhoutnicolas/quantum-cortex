"""tools/adr010_launcher.py -- the launcher of every run of ADR-010 (the amendment of 2026-10-15 on parts B and C, B.3 and
4 (b)): outside the package, so that the environment is set before the interpreter that runs the measurement imports
anything. It sets PYTHONHASHSEED to 0 for every run of the ADR and, with --pin-threads (the cost harness, which refuses
to run otherwise), the four thread variables to 1, and PYTHONUNBUFFERED so that a log piped through Tee-Object receives
every line as it is printed; then it starts `python -m <module> <args>` and exits with its code.

    python tools/adr010_launcher.py [--pin-threads] <module> [arguments of the module]

The command is recorded by the measurement in its file (4 (b)).
"""
from __future__ import annotations

import os
import subprocess
import sys

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def main(argv: list[str]) -> int:
    pin = False
    if argv and argv[0] == "--pin-threads":
        pin = True; argv = argv[1:]
    if not argv:
        print("usage: python tools/adr010_launcher.py [--pin-threads] <module> [arguments]", file=sys.stderr)
        return 2
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    if pin:
        for k in THREAD_VARIABLES:
            env[k] = "1"
    env["ADR010_LAUNCHER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"                       # the run's lines reach a log piped through Tee-Object as they are printed
    cmd = [sys.executable, "-m", argv[0], *argv[1:]]
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
