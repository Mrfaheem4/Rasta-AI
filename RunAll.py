"""Reproduce every experiment and figure: python RunAll.py [--quick]

Modules run in order; a failing module is reported and the rest still run. The exit code is 1 if any failed.
"""

import argparse
import importlib
import importlib.util
import inspect
import sys
import time
import traceback

from core.Config import stampLine
from experiments.Stamp import printTable

modules = [
    "experiments.WorldExperiments",
    "experiments.RouteExperiments",
    "experiments.ScenarioExperiments",
    "experiments.GameExperiments",
    "experiments.LlmExperiments",
    "experiments.Plots",
]


def runModule(name, quick):
    main = importlib.import_module(name).main
    if "quick" in inspect.signature(main).parameters:
        main(quick=quick)
    else:
        main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="passed to modules that accept quick=True")
    args = parser.parse_args()
    print(stampLine)
    summary = []
    for name in modules:
        if importlib.util.find_spec(name) is None:
            print(f"\n[skip] {name}: no such module")
            summary.append((name, "skipped", 0.0))
            continue
        print(f"\n[run] {name}")
        started = time.perf_counter()
        try:
            runModule(name, args.quick)
            status = "ok"
        except Exception:
            traceback.print_exc()
            status = "FAILED"
        summary.append((name, status, time.perf_counter() - started))
    printTable("RunAll summary", ["module", "status", "seconds"], summary)
    failed = [name for name, status, _ in summary if status == "FAILED"]
    if failed:
        print(f"\n{len(failed)} module(s) failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
