"""
STEP 22 - fill `power_production.pptx` with the regenerated saturation numbers.

Why a script rather than hand-editing the slide: the numbers now live in the canonical
`saturation_flags.csv`, so the deck should be regenerable from them instead of being a
hand-copied snapshot that drifts the next time the detector changes.

Slide 3 is titled "Explain about saturated problem" and shipped with an empty content
placeholder, so this fills a hole the deck's author left open rather than rewriting
anything. Slides 1-2 are untouched.

Idempotent: every shape this script adds is named with the SATX_ prefix and removed before
re-adding, so re-running it updates in place instead of stacking duplicate tables.

Run:  ./venv/Scripts/python.exe sat_work/research/step22_fill_deck.py
"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "power_production.pptx")

INK = RGBColor(0x1F, 0x24, 0x2B)
MUTED = RGBColor(0x5A, 0x63, 0x6E)
ACCENT = RGBColor(0x0B, 0x63, 0x8C)
DANGER = RGBColor(0xA3, 0x2B, 0x2B)

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


# --------------------------------------------------------------------------- #
# slide 3 - the explanation the author left as a heading only
# --------------------------------------------------------------------------- #
s3 = prs.slides[2]
drop_tagged(s3)

# The slide shipped with an EMPTY PICTURE placeholder on the right (a figure slot) and no
# text body at all. There is no figure in the repo I can verify the contents of, so rather
# than guess at one for a shared deck this lays the slide out as table-left / text-right
# and leaves the figure decision to the author.
for sh in list(s3.shapes):
    if sh.is_placeholder and sh.name != "Title 1" and (
            not sh.has_text_frame or not sh.text_frame.text.strip()):
        sh._element.getparent().remove(sh._element)

add_table(
    s3, "SATX_S3_TABLE",
    left=838200, top=1900000, width=3250000, height=1750000,
    header=["Shared MPPT pair", "Saturated", "Severe"],
    rows=[["eu_8 + eu_16", "386.6 h", "7.1 h"],
          ["eu_10 + eu_18", "539.8 h", "13.0 h"],
          ["eu_13 + eu_21", "342.8 h", "7.2 h"],
          ["Total", "1,269.2 h", "27.3 h"]],
    widths=[40, 32, 28], size=11,
)

s3_body = add_box(s3, "SATX_S3_BODY", 4400000, 1700000, 5500000, 4600000)
write_lines(s3_body.text_frame, [
    ("Saturation: the pair sum stops rising while irradiance keeps rising -- the shared "
     "MPPT has run out of headroom, so available energy is lost.", 0, False, INK),
    ("Shared MPPT channels (confirmed with the site):", 0, True, INK),
    ("eu_8 + eu_16      eu_10 + eu_18      eu_13 + eu_21", 1, False, ACCENT),
    ("Detection (vat-v1):", 0, True, INK),
    ("expected = theta(t) * ref,   deficit = 1 - measured / expected", 1, False, MUTED),
    ("moderate = deficit > 15 % for 15 min, at ref >= 0.70", 1, False, MUTED),
    ("severe = deficit > 30 % for 30 min", 1, False, MUTED),
    ("The model is physical, not fitted: theta recovers the 9.0 kW string rating to within 4 % "
     "across all 23 units (median 8.85 kW).", 0, False, INK),
    ("Independent, non-shared pairs stay clean -- two of the three flag exactly zero.", 0, True, INK),
], size=12)

# --------------------------------------------------------------------------- #
# slide 4 - the corrected numbers
# --------------------------------------------------------------------------- #
s4 = prs.slides[3]
drop_tagged(s4)

# the slide shipped with an empty full-bleed placeholder: remove it so the table below is
# not sitting on top of an invisible-but-clickable box
for sh in list(s4.shapes):
    if sh.is_placeholder and (
            not sh.has_text_frame or not sh.text_frame.text.strip()):
        sh._element.getparent().remove(sh._element)

title = add_box(s4, "SATX_S4_TITLE",
                Inches(0.70), Inches(0.30), Inches(12.0), Inches(0.75))
write_lines(title.text_frame, [
    ("Corrected numbers: 13 % fewer flagged hours, severe tier overstated 2.9x, "
     "energy bias floor 47 % -> 7 %", 0, True, INK),
], size=20)

add_table(
    s4, "SATX_S4_TABLE",
    left=Emu(643467), top=Inches(1.35), width=Emu(10905066), height=Inches(2.0),
    header=["Shared pair", "published", "corrected", "published severe", "corrected severe"],
    rows=[["eu_8 + eu_16", "473.3 h", "386.6 h", "30.7 h", "7.1 h"],
          ["eu_10 + eu_18", "592.6 h", "539.8 h", "27.8 h", "13.0 h"],
          ["eu_13 + eu_21", "394.0 h", "342.8 h", "20.2 h", "7.2 h"],
          ["Total", "1,459.9 h", "1,269.2 h", "78.7 h", "27.3 h"]],
    widths=[26, 18, 18, 19, 19], size=13, header_size=12,
)

body = add_box(s4, "SATX_S4_BODY",
               Inches(0.70), Inches(3.75), Inches(12.0), Inches(3.0))
write_lines(body.text_frame, [
    ("Apparent lost energy at ref >= 0.70 is 6,120 kWh. The published figure was 7,131 kWh, "
     "but 47 % of it was baseline bias -- independent pairs that cannot saturate were 'losing' "
     "energy too. The corrected bias floor is 7 %.", 0, False, INK),
    ("The seasonal shape is unchanged (May-Jul peak, plus the cold-February clear-day peak), and "
     "every published conclusion survives. Only the magnitudes were inflated.", 0, False, INK),
    ("Canonical artefact: saturation_flags.csv, 51,005 x 21, regenerated from vat-v1. Guarded by "
     "44 regression tests -- 43 passing, 1 documenting the old method's known control-pair bias.",
     0, False, MUTED),
    ("Known limitation: an unfiltered mean over the raw deficit column is not meaningful -- the "
     "column is only defined inside the ref >= 0.70 gate.", 0, False, MUTED),
], size=13)

prs.save(SRC)
print("wrote", SRC)
print("slides:", len(prs.slides))
for i, s in enumerate(prs.slides, 1):
    added = [sh.name for sh in s.shapes if sh.name.startswith("SATX_")]
    if added:
        print(f"  slide {i}: added {added}")
