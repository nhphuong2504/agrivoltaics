"""
Dependency-free test runner for the saturation regression suite.

The project venv does not ship pytest (and requirements.txt does not list it), so this
runs the same `test_*` functions using only the standard library. If pytest is ever
installed, `pytest sat_work/tests` works too and honours the same `known_failure`
markers via `_helpers.known_failure`.

Usage
-----
    ./venv/Scripts/python.exe sat_work/tests/run_tests.py            # everything
    ./venv/Scripts/python.exe sat_work/tests/run_tests.py -k export  # name filter
    ./venv/Scripts/python.exe sat_work/tests/run_tests.py -v         # per-test timing
    ./venv/Scripts/python.exe sat_work/tests/run_tests.py --list     # list only

Exit code is 0 when nothing fails. Tests marked `known_failure` that fail as expected
report as XFAIL and do not affect the exit code; if one of them *passes*, it reports as
XPASS, which is a signal that a documented defect has been fixed and the artifact (and
SATURATION_REVIEW.md) needs updating.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _helpers  # noqa: E402
from _helpers import KnownFailure  # noqa: E402

TEST_MODULES = ["test_detectors", "test_exports", "test_topology"]


def discover(filters=()):
    """Yield (module_name, test_name, callable) for every test_* function."""
    found = []
    for mod_name in TEST_MODULES:
        mod = importlib.import_module(mod_name)
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            if filters and not any(f.lower() in name.lower() for f in filters):
                continue
            found.append((mod_name, name, fn))
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", "--filter", action="append", default=[],
                    help="only run tests whose name contains this substring (repeatable)")
    ap.add_argument("-v", "--verbose", action="store_true", help="print per-test timing")
    ap.add_argument("--list", action="store_true", help="list tests without running them")
    args = ap.parse_args(argv)

    tests = discover(args.filter)
    if args.list:
        for mod, name, _ in tests:
            print(f"{mod}.{name}")
        print(f"\n{len(tests)} tests")
        return 0

    print("=" * 88)
    print("SATURATION DETECTOR REGRESSION SUITE")
    print("=" * 88)

    n_pass = n_fail = n_xfail = n_xpass = 0
    failures, xpasses = [], []
    t_suite = time.time()

    for mod, name, fn in tests:
        t0 = time.time()
        try:
            fn()
            dt = time.time() - t0
            if getattr(fn, "_known_failure", None):
                n_xpass += 1
                xpasses.append((mod, name, fn._known_failure))
                status = "XPASS"
            else:
                n_pass += 1
                status = "PASS "
        except KnownFailure as exc:
            n_xfail += 1
            dt = time.time() - t0
            status = "XFAIL"
            print(f"{status} {mod}.{name}  ({dt:.1f}s)")
            if args.verbose:
                print(f"        {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - a test runner must catch everything
            n_fail += 1
            dt = time.time() - t0
            status = "FAIL "
            failures.append((mod, name, exc, traceback.format_exc()))
        print(f"{status} {mod}.{name}  ({dt:.1f}s)" if args.verbose
              else f"{status} {mod}.{name}")

    total = time.time() - t_suite
    print("-" * 88)
    print(f"{n_pass} passed, {n_fail} failed, {n_xfail} xfailed, {n_xpass} xpassed "
          f"in {total:.1f}s")

    if failures:
        print()
        print("=" * 88)
        print("FAILURES")
        print("=" * 88)
        for mod, name, exc, tb in failures:
            print(f"\n--- {mod}.{name} ---")
            print(tb if args.verbose else f"{type(exc).__name__}: {exc}")

    if xpasses:
        print()
        print("=" * 88)
        print("XPASS -- a documented defect appears FIXED")
        print("=" * 88)
        for mod, name, reason in xpasses:
            print(f"  {mod}.{name}")
            print(f"     documented as: {reason}")
        print("\n  Remove the `known_failure` marker and update SATURATION_REVIEW.md.")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
