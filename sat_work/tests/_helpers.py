"""
Test helpers for the saturation-detector regression suite.

Runs with pytest if it is installed, but nothing here requires it: `run_tests.py`
discovers and executes the same `test_*` functions with the standard library only,
because the project venv does not ship pytest.

`known_failure` marks a test that asserts a defect which is real but not yet fixed.
It reports as XFAIL rather than a failure, and flips to a hard pass once the underlying
issue is resolved -- which is what we want for a documented bug in a shipped artefact.
Write the test to assert the *fixed* condition; the decorator handles the rest.
"""
from __future__ import annotations

import functools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH = os.path.join(os.path.dirname(_HERE), "research")
_ROOT = os.path.dirname(os.path.dirname(_RESEARCH))   # the repo root
for _p in (_RESEARCH, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class KnownFailure(Exception):
    """A test that documents a known, unfixed defect and still fails as expected."""


def known_failure(reason):
    """Turn an AssertionError into an expected failure, reported as XFAIL.

    The test should assert the condition that will hold *once the defect is fixed*.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except KnownFailure:
                raise
            except AssertionError as exc:
                msg = f"{reason} :: {exc}"
                if "pytest" in sys.modules:          # pragma: no cover - pytest path
                    import pytest
                    pytest.xfail(msg)
                raise KnownFailure(msg) from None
        wrapper._known_failure = reason
        return wrapper
    return deco


# Convenience re-exports so tests import from one place.
import bench as bench  # noqa: E402

# THE CANONICAL FLAGS ARTEFACT. Since 2026-09 this is the vat-v1 export: it is written by
# research/recommended.py and is the file downstream consumers read.
CANONICAL_CSV = os.path.join(_ROOT, "saturation_flags.csv")

# The schema the ORIGINAL published v1 artefact shipped, frozen here as a literal.
#
# This has to be a constant rather than read back from the file on disk. When the canonical
# file started being produced by a different method, deriving "the published schema" from
# it became circular -- the contract would hold by construction and test nothing.
#
# MIGRATION (2026-09): the published v1 schema is no longer a subset of the canonical one.
# The three tier columns were collapsed into a single `{pair}_sat`, so a consumer of
# `sat_moderate` or `sat_severe` must switch to `sat`. That is a deliberate breaking change,
# not an oversight: the artefact has no consumers outside this repository, and retaining a
# `sat_moderate` alias would have kept alive exactly the tier vocabulary the simplification
# removed. `deficit` is unchanged and still carries severity. `test_exports.py` pins the
# migration one-for-one so it cannot drift.
PUBLISHED_V1_COLUMNS = (
    "time", "ref", "dq_site_outage",
    *(f"{a}_{b}_{suffix}" for a, b in bench.PAIRS
      for suffix in ("deficit", "sat_moderate", "sat_severe")),
)

# Published v1 columns that the canonical schema deliberately does NOT carry, each replaced
# by `{pair}_sat`. Asserted in test_exports.py.
RETIRED_V1_COLUMNS = tuple(
    f"{a}_{b}_{suffix}" for a, b in bench.PAIRS for suffix in ("sat_moderate", "sat_severe")
)


# --------------------------------------------------------------------------- #
# Scenario accessors. The expensive part is cached in bench, so these are cheap.
# --------------------------------------------------------------------------- #
def noclip(pair=("eu_1", "eu_3")):
    """An independent pair with NO constraint injected: every flag is false by definition."""
    return bench.scenario(pair, duty=1.00, q=None, pattern="chronic")


def intermittent(pair=("eu_1", "eu_3")):
    """The realistic failure mode: the limit binds on 35 % of days, not all of them."""
    return bench.scenario(pair, duty=0.35, q=0.75, pattern="episodic")


def chronic(pair=("eu_1", "eu_3")):
    """A limit that binds every single day. This is the regime the published validation
    implicitly assumed, and the one in which the published detector looks fine."""
    return bench.scenario(pair, duty=1.00, q=0.75, pattern="chronic")


EPISODIC_PAIRS = (("eu_1", "eu_3"), ("eu_15", "eu_23"))


def score(sc, detector, **kw):
    """Score one detector on one scenario, optionally with extra detector kwargs."""
    if kw:
        def wrapped(df, fe, a, b, _fn=detector, _kw=kw):
            return _fn(df, fe, a, b, **_kw)
        return bench.score_scenario(sc, wrapped)
    return bench.score_scenario(sc, detector)
