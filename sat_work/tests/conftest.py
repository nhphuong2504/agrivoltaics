"""pytest configuration.

Only needed if pytest is installed; `run_tests.py` works without it. Ensures the tests
directory and the research directory are importable so `from _helpers import ...` and
`from bench import ...` resolve regardless of the invocation directory.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH = os.path.join(os.path.dirname(_HERE), "research")
_ROOT = os.path.dirname(os.path.dirname(_RESEARCH))

for _p in (_HERE, _RESEARCH, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
