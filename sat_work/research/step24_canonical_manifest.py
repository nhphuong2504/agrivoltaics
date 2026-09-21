"""
STEP 24 - write and verify the canonical version lock.

`saturation_flags.csv` is a generated product and is gitignored (`*.csv`) on purpose: it is
a large artefact that `recommended.py` rebuilds deterministically, and committing it would
put a large file in every future diff. So reproducibility is locked by a small committed
manifest instead:

    sat_work/canonical/CANONICAL_vat-v1.json

which records

  * the method configuration, read from code rather than re-typed,
  * every headline metric with its definition and denominator (from canon_metrics),
  * the sha256 of the exact artefact those metrics were measured on.

`tests/test_manifest.py` re-derives the metrics from the live file and fails if they stop
matching, so the manifest cannot silently go stale. The hash is checked here rather than in
the test, because a byte-level hash legitimately depends on the pandas version that wrote
the file; the *metrics* are what must hold.

    python step24_canonical_manifest.py            # regenerate the manifest
    python step24_canonical_manifest.py --check    # verify, non-zero exit on drift
"""
import sys, io, os, json, datetime
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import canon_metrics as CM  # noqa: E402

OUT_DIR = os.path.join(_ROOT, "sat_work", "canonical")
OUT = os.path.join(OUT_DIR, "CANONICAL_vat-v1.json")


def build() -> dict:
    can = CM.load_canonical()
    s = CM.summarise(can)
    num = can.select_dtypes("number")
    n_inf = int((num == float("inf")).sum().sum() + (num == float("-inf")).sum().sum())
    return dict(
        version="vat-v1",
        generated=datetime.date.today().isoformat(),
        artefact=dict(
            path="saturation_flags.csv",
            tracked=False,
            tracked_reason="generated product, gitignored by *.csv; rebuild with "
                           "sat_work/research/recommended.py",
            sha256=CM.sha256(),
            rows=s["record_rows"],
            # Derived, never hardcoded. This was a literal `21` left over from the retired
            # six-columns-per-pair schema; the artefact has carried 9 columns since the
            # tier collapse, and no test asserted the field, so the lock misdescribed the
            # file it was locking for two revisions.
            columns=int(len(can.columns)),
            infinities=n_inf,
            deficit_values_outside_domain=s["deficit_values_outside_domain"],
        ),
        method=s["method"],
        metrics=dict(
            flagged=s["flagged"],
            controls=s["controls"],
            energy=s["energy"],
            domain_rows=s["domain_rows"],
            deficit_in_domain_range=s["deficit_in_domain_range"],
            benchmark=s["benchmark"],
        ),
        definitions=dict(
            domain=CM.DOMAIN_DEF,
            energy_gate=CM.GATE_DEF,
            hours="flagged rows / 12 (verified modal 5-min grid; sampled hours, not "
                  "wall-clock, because acquisition gaps contribute no rows)",
            flagged_rows="pooled over the three shared MPPT pairs",
            controls="the three legacy control pairs; the cross-inverter pair is the only "
                     "honest flag-level null",
            bias_floor="control-pair apparent loss / shared-pair apparent loss, same gate",
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the manifest against the live artefact instead of writing")
    args = ap.parse_args()
    fresh = build()

    if args.check:
        if not os.path.exists(OUT):
            print(f"MISSING manifest {OUT}")
            return 1
        locked = json.load(open(OUT, encoding="utf-8"))
        problems = []
        for key in ("metrics", "method"):
            if locked[key] != fresh[key]:
                problems.append(key)
                for k in sorted(set(locked[key]) | set(fresh[key])):
                    if locked[key].get(k) != fresh[key].get(k):
                        print(f"  {key}.{k}: locked {locked[key].get(k)} != "
                              f"live {fresh[key].get(k)}")
        same_hash = locked["artefact"]["sha256"] == fresh["artefact"]["sha256"]
        print(f"metrics/method match: {not problems}")
        print(f"sha256 match: {same_hash}")
        print(f"  locked {locked['artefact']['sha256']}")
        print(f"  live   {fresh['artefact']['sha256']}")
        if problems:
            print("\nDRIFT: regenerate with step24_canonical_manifest.py")
            return 1
        print("\nOK - the live artefact still satisfies the lock.")
        return 0

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(fresh, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {OUT}")
    print(f"  version      : {fresh['version']}")
    print(f"  sha256       : {fresh['artefact']['sha256']}")
    print(f"  flagged      : {fresh['metrics']['flagged']['published_rows']:,} rows -> "
          f"{fresh['metrics']['flagged']['canonical_rows']:,} rows "
          f"({fresh['metrics']['flagged']['canonical_hours']:,} sampled hours)")
    print(f"  bias floor   : {fresh['metrics']['energy']['bias_floor_published_pct']} % -> "
          f"{fresh['metrics']['energy']['bias_floor_canonical_pct']} %")
    print(f"  specificity  : {fresh['metrics']['benchmark']['specificity_gain_x']}x episodic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
