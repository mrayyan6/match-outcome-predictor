"""Pull in the latest results and fixtures, rebuild, push.

Runs twice a week from a Windows scheduled task (Sofascore blocks GitHub's
servers, so GitHub Actions was out). Each run also fetches possession for
up to 400 older games we're still missing, so the history fills itself in
over a few runs without hammering anyone.

    python refresh.py             fetch, rebuild, commit and push if anything changed
    python refresh.py --no-push   same, but leave the commit local
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "refresh.log"
OUTPUTS = ["data/processed", "reports", "web/data"]
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # quiet under pythonw


def log(msg: str) -> None:
    line = f"{datetime.now():%Y-%m-%d %H:%M} {msg}"
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if sys.stdout:
        print(line)


def run(*args: str) -> str:
    out = subprocess.run(
        args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        creationflags=NO_WINDOW, env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    if out.returncode:
        log(f"failed: {' '.join(args)}\n{out.stdout[-2000:]}{out.stderr[-2000:]}")
        raise SystemExit(1)
    return out.stdout.strip()


def python(*args: str) -> str:
    exe = Path(sys.executable).with_name("python.exe")
    return run(str(exe if exe.exists() else sys.executable), *args)


def changed_files() -> list[str]:
    status = run("git", "status", "--porcelain", "--", *OUTPUTS)
    return [line[3:] for line in status.splitlines()]


def main() -> None:
    push = "--no-push" not in sys.argv
    log("refresh started")
    run("git", "pull", "--ff-only")

    for line in python("data_loader.py", "--max-stats", "400").splitlines():
        if "matches" in line or "possession" in line or "said no" in line:
            log(line.strip())
    python("model.py")
    python("export_web.py")
    python("-m", "pytest", "-q")

    changed = changed_files()
    # the export date alone isn't news
    if all(f.endswith("meta.json") for f in changed):
        run("git", "checkout", "--", "web/data/meta.json")
        log("nothing new")
        return

    run("git", "add", *OUTPUTS)
    run("git", "commit", "-m", f"Refresh results and fixtures {datetime.now():%Y-%m-%d}", "--", *OUTPUTS)
    if push:
        run("git", "push")
    log(f"committed {len(changed)} changed files" + (" and pushed" if push else ""))


if __name__ == "__main__":
    main()
