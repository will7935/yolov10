"""Run the five configured slot-group regression videos.

Examples (from the repository root)::

    python industrial_inspection/tools/run_regression_suite.py
    python industrial_inspection/tools/run_regression_suite.py --max-frames 300
    python industrial_inspection/tools/run_regression_suite.py --with-video

The default run skips annotated MP4 output and writes one data directory per
video plus a ``regression_suite_report.json`` index under
``OUTPUT_ROOT/regression_runs``.  ``--with-video`` enables annotated MP4s.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from industrial_inspection.tools.predict_slot_groups import run_regression_suite


def main() -> int:
    extra_args = list(sys.argv[1:])
    write_video = "--with-video" in extra_args
    extra_args = [argument for argument in extra_args if argument != "--with-video"]
    report = run_regression_suite(extra_args, write_video=write_video)
    print(f"回归套件完成：{report['status']}，报告：{report['report_path']}", flush=True)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
