"""
STEP 22 - fill `power_production.pptx` from the canonical summary.

Why a script rather than hand-editing the slide: the numbers live in the canonical
`saturation_flags.csv` via `canon_metrics.summarise()`, the same source the review, the
canvas and the version lock use. Hand-copying them into a deck is what makes a deck drift.

Slide 3 is titled "Explain about saturated problem" and shipped with an empty figure
placeholder, so this fills a hole the deck's author left open rather than rewriting
anything. Slides 1-2 are untouched.

Slide 3 gets the validated four-panel figure (step25) because that is the slide whose job is
to *explain the problem*, and the figure carries its own four independent validations.
Slide 4 keeps the published-vs-corrected comparison.

Idempotent: every shape this script adds is named with the SATX_ prefix and removed before
re-adding, so re-running updates in place instead of stacking duplicate tables.

Run:  ./venv/Scripts/python.exe sat_work/research/step22_fill_deck.py
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

import canon_metrics as CM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "power_production.pptx")
FIGURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validated_evidence.png")

INK = RGBColor(0x1F, 0x24, 0x2B)
MUTED = RGBColor(0x5A, 0x63, 0x6E)
ACCENT = RGBColor(0x0B, 0x63, 0x8C)

# ---- every number in the deck comes from here, none is typed in ----------- #
S = CM.summarise()
F, E, BM, CT = S["flagged"], S["energy"], S["benchmark"], S["controls"]
PAIR_ROWS = []
for a, b in __import__("bench").PAIRS:
    k = f"{a}_{b}".replace("+", "_")
    p = F["per_pair"][k]
    PAIR_ROWS.append((f"{a} + {b}", p["canonical_rows"], p["canonical_severe_rows"]))
H = CM.SAMPLES_PER_HOUR


def hrs(rows):
    return f"{rows / H:,.1f} h"


def test_count():
    """Count the suite instead of hard-coding it into a slide."""
    sys.path.insert(0, os.path.join(ROOT, "sat_work", "tests"))
    import run_tests
    return len(run_tests.discover())


prs = Presentation(SRC)


def drop_tagged(slide, prefix="SATX_"):
    """Remove shapes this script added on a previous run."""
    for sh in list(slide.shapes):
        if sh.name.startswith(prefix):
            sh._element.getparent().remove(sh._element)


def style_run(run, size=12, bold=False, italic=False, color=INK):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def write_lines(tf, lines, size=12):
    """lines: (text, indent_level, bold, color) tuples."""
    tf.clear()
    for i, (text, level, bold, color) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = level
        p.space_after = Pt(4)
        style_run(p.add_run(), size=size, bold=bold, color=color)
        p.runs[0].text = text


def add_box(slide, name, left, top, width, height):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tb.name = name
    tb.text_frame.word_wrap = True
    return tb


def add_table(slide, name, left, top, width, height, header, rows, widths=None,
              size=11, header_size=11):
    gf = slide.shapes.add_table(len(rows) + 1, len(header), left, top, width, height)
    gf.name = name
    tbl = gf.table
    if widths:
        span = sum(widths)
        for i, w in enumerate(widths):
            tbl.columns[i].width = Emu(int(width * w / span))
    for j, htxt in enumerate(header):
        c = tbl.cell(0, j)
        c.text = htxt
        for r in c.text_frame.paragraphs[0].runs:
            style_run(r, size=header_size, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
        c.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
    for i, row in enumerate(rows, start=1):
        is_total = str(row[0]).lower().startswith("total")
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.text = str(val)
            for r in c.text_frame.paragraphs[0].runs:
                style_run(r, size=size, bold=is_total, color=INK)
            c.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.RIGHT
    return tbl


def clear_empty_placeholders(slide, keep=("Title 1",)):
    """The shipped slides carry empty placeholders; remove them so added shapes are not
    sitting on top of invisible-but-clickable boxes."""
    for sh in list(slide.shapes):
        if sh.is_placeholder and sh.name not in keep and (
                not sh.has_text_frame or not sh.text_frame.text.strip()):
            sh._element.getparent().remove(sh._element)


# --------------------------------------------------------------------------- #
# slide 3 - the explanation slide: the validated figure is the argument
# --------------------------------------------------------------------------- #
s3 = prs.slides[2]
drop_tagged(s3)
clear_empty_placeholders(s3)

# The Title placeholder ships 1.45 in tall for a single line of text, so its bounding box
# runs under the content. Tighten it to its actual text height so the layout geometry is
# honest and nothing sits inside another shape's box.
for sh in s3.shapes:
    if sh.name == "Title 1":
        sh.height = Inches(0.80)

add_table(
    s3, "SATX_S3_TABLE",
    left=Inches(0.40), top=Inches(1.55), width=Inches(3.75), height=Inches(1.55),
    header=["Shared MPPT pair", "Saturated", "Severe"],
    rows=[[r[0], hrs(r[1]), hrs(r[2])] for r in PAIR_ROWS]
         + [["Total", hrs(F["canonical_rows"]), hrs(F["canonical_severe_rows"])]],
    widths=[42, 30, 28], size=10.5, header_size=10.5,
)

s3_body = add_box(s3, "SATX_S3_BODY", Inches(0.40), Inches(3.30), Inches(3.75), Inches(3.7))
write_lines(s3_body.text_frame, [
    ("What saturation is", 0, True, ACCENT),
    ("The pair sum stops rising while irradiance keeps rising: the shared MPPT has run out "
     "of headroom and energy is lost.", 0, False, INK),
    ("Detection (vat-v1)", 0, True, ACCENT),
    ("expected = theta(t) * ref", 1, False, MUTED),
    ("deficit = 1 - measured / expected", 1, False, MUTED),
    ("moderate: > 15 % for 15 min", 1, False, MUTED),
    ("severe: > 30 % for 30 min", 1, False, MUTED),
    ("moderate tier vs the published method: "
     f"{F['published_hours']:,.1f} h -> {F['canonical_hours']:,.1f} h "
     f"({F['row_change_pct']:+.1f} %).", 0, False, INK),
], size=10)

# the figure slot the slide shipped with, now filled with the validated evidence.
# Sized so the bottom edge clears the caption: width 8.20 in on a 1690x1170 figure is
# 5.68 in tall, so 1.30 + 5.68 = 6.98 and the caption sits below it.
if os.path.exists(FIGURE):
    s3.shapes.add_picture(FIGURE, Inches(4.32), Inches(1.30), width=Inches(8.20)).name = \
        "SATX_S3_FIG"
    cap = add_box(s3, "SATX_S3_CAP", Inches(4.32), Inches(7.02), Inches(8.20), Inches(0.40))
    write_lines(cap.text_frame, [
        ("Four independent validations: (a) normalisation-free peer comparison on real data, "
         "(b) the inverter topology natural experiment, (c) the 9.0 kW nameplate cross-check, "
         "(d) injected ground truth.", 0, False, MUTED)], size=9)
else:
    print("  WARNING: figure missing, run step25_validated_evidence.py")

# --------------------------------------------------------------------------- #
# slide 4 - the corrected numbers
# --------------------------------------------------------------------------- #
s4 = prs.slides[3]
drop_tagged(s4)
clear_empty_placeholders(s4, keep=())

title = add_box(s4, "SATX_S4_TITLE",
                Inches(0.70), Inches(0.30), Inches(12.0), Inches(0.75))
write_lines(title.text_frame, [
    (f"Corrected numbers: {abs(F['row_change_pct']):.0f} % fewer flagged hours, severe tier "
     f"overstated {F['severe_inflation_x']:.2f}x, energy bias floor "
     f"{E['bias_floor_published_pct']} % -> {E['bias_floor_canonical_pct']} %", 0, True, INK),
], size=19)

PUB_ROW = {}
for a, b in __import__("bench").PAIRS:
    k = f"{a}_{b}".replace("+", "_")
    PUB_ROW[f"{a} + {b}"] = F["per_pair"][k]
add_table(
    s4, "SATX_S4_TABLE",
    left=Emu(643467), top=Inches(1.35), width=Emu(10905066), height=Inches(2.0),
    header=["Shared pair", "published", "corrected", "published severe", "corrected severe"],
    rows=[[r[0], hrs(PUB_ROW[r[0]]["published_rows"]), hrs(PUB_ROW[r[0]]["canonical_rows"]),
           hrs(PUB_ROW[r[0]]["published_severe_rows"]),
           hrs(PUB_ROW[r[0]]["canonical_severe_rows"])] for r in PAIR_ROWS]
         + [["Total", hrs(F["published_rows"]), hrs(F["canonical_rows"]),
             hrs(F["published_severe_rows"]), hrs(F["canonical_severe_rows"])]],
    widths=[26, 18, 18, 19, 19], size=13, header_size=12,
)

body = add_box(s4, "SATX_S4_BODY",
               Inches(0.70), Inches(3.75), Inches(12.0), Inches(3.0))
write_lines(body.text_frame, [
    (f"Apparent lost energy over the detector gate domain "
     f"(ref >= 0.70 and s > 0.6*expected) is {E['shared_canonical_kwh']:,} kWh. The "
     f"published figure was {E['shared_published_kwh']:,} kWh, but "
     f"{E['bias_floor_published_pct']} % of it was baseline bias -- independent pairs that "
     f"cannot saturate were 'losing' energy too. The corrected bias floor is "
     f"{E['bias_floor_canonical_pct']} %.", 0, False, INK),
    ("The seasonal shape is unchanged (May-Jul peak, plus the cold-February clear-day peak), "
     "and every published conclusion survives. Only the magnitudes were inflated.", 0, False, INK),
    (f"Canonical artefact: saturation_flags.csv ({S['record_rows']:,} x 21), regenerated from "
     f"vat-v1 and locked by sat_work/canonical/CANONICAL_vat-v1.json. Guarded by "
     f"{test_count()} regression tests.", 0, False, MUTED),
    ("Known limitation: the raw deficit column is only defined inside the decision domain "
     "(ref >= 0.70, both units active, no data-quality flag); it is NaN elsewhere by design.",
     0, False, MUTED),
], size=13)

prs.save(SRC)
print("wrote", SRC)
print(f"  slide 3: saturated hours {F['canonical_hours']:,.1f} h, severe "
      f"{F['canonical_severe_hours']:,.1f} h, figure {'placed' if os.path.exists(FIGURE) else 'MISSING'}")
print(f"  slide 4: {F['published_hours']:,.1f} h -> {F['canonical_hours']:,.1f} h, "
      f"severe {F['published_severe_hours']:,.1f} h -> {F['canonical_severe_hours']:,.1f} h")
print(f"  tests counted: {test_count()}")
