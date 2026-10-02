"""Build the cover image and the output card for the DEV submission.

Every number rendered here is read from data/results.json or hard-coded from a line
in FINDINGS.md, so the image cannot drift from the measurements it claims.

Usage: python scripts/make_cover.py
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "assets"
OUT.mkdir(exist_ok=True)

FONT_BOLD = "C:/Windows/Fonts/arialbd.ttf"
FONT_REG = "C:/Windows/Fonts/arial.ttf"
FONT_MONO = "C:/Windows/Fonts/consola.ttf"

INK = (26, 28, 34)
PAPER = (247, 247, 242)
FOREST = (45, 71, 66)
ACCENT = (229, 57, 39)
MUTED = (120, 124, 132)


def font(path, size):
    return ImageFont.truetype(path, size)


def holdout():
    table = json.loads((REPO / "data" / "results.json").read_text(encoding="utf-8"))["holdout_table"]
    return {row["model"]: row for row in table}


def cover():
    rows = holdout()
    folk = rows["FOLK BELIEF: loudest-first (leaky features)"]
    trivial = rows["always-alive (trivial)"]
    tabpfn = rows["TabPFN v2, triage-time features only"]
    corpus = json.loads((REPO / "data" / "results.json").read_text(encoding="utf-8"))["corpus"]

    W, H = 1200, 630
    img = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(img)

    # Forest band on the left edge, matching the Hacktoberfest palette.
    d.rectangle([0, 0, 18, H], fill=FOREST)

    d.text((66, 58), "TRIAGELENS", font=font(FONT_MONO, 26), fill=(140, 178, 222))
    d.text((66, 100), "Offline issue triage on TabPFN v2", font=font(FONT_REG, 24), fill=MUTED)

    d.text((66, 168), "Loudest-first triage", font=font(FONT_BOLD, 62), fill=PAPER)
    d.text((66, 240), "is worse than random", font=font(FONT_BOLD, 62), fill=ACCENT)

    d.text(
        (66, 336),
        f"precision@100 of the top 100 issues, held-out future period, "
        f"{corpus['n_closed']:,} closed issues",
        font=font(FONT_REG, 21),
        fill=MUTED,
    )

    # The three-row comparison that is the whole argument.
    y = 392
    d.rectangle([66, y, 620, y + 148], fill=(34, 37, 45))
    rows_to_draw = [
        ("Sort by comments + reactions", folk["precision_at_100"], ACCENT),
        ("Pick at random", trivial["precision_at_100"], ACCENT),
        ("TabPFN v2", tabpfn["precision_at_100"], (140, 178, 222)),
    ]
    for i, (label, value, colour) in enumerate(rows_to_draw):
        ry = y + 18 + i * 44
        d.text((90, ry), label, font=font(FONT_REG, 21), fill=PAPER if i == 2 else MUTED)
        d.text((560, ry), f"{value:.3f}", font=font(FONT_MONO, 22), fill=colour)

    d.text((664, 400), "ROC-AUC", font=font(FONT_MONO, 20), fill=MUTED)
    for i, (label, value) in enumerate(
        [("loudest-first", folk["roc_auc"]), ("TabPFN v2", tabpfn["roc_auc"])]
    ):
        ry = 434 + i * 44
        d.text((664, ry), label, font=font(FONT_REG, 20), fill=MUTED)
        d.text((1150, ry), f"{value:.4f}", font=font(FONT_MONO, 22), fill=PAPER, anchor="ra")
        # Bar scaled from the 0.45 floor so a coin flip reads as a coin flip.
        bar_w = int((value - 0.45) / 0.40 * 486)
        if bar_w > 0:
            d.rectangle([664, ry + 28, 664 + bar_w, ry + 34], fill=(140, 178, 222))

    d.text(
        (66, 574),
        "MIT licensed  |  30 tests  |  no account, no API key, no network at inference",
        font=font(FONT_REG, 19),
        fill=MUTED,
    )

    path = OUT / "cover.png"
    img.save(path, "PNG", optimize=True)
    return path, img.size


def cli_card():
    """A card of the verbatim CLI output, so the post shows real output not a claim."""
    W, H = 1200, 760
    img = Image.new("RGB", (W, H), (18, 20, 25))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 56], fill=(30, 33, 40))
    for i, colour in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        d.ellipse([24 + i * 30, 20, 40 + i * 30, 36], fill=colour)
    d.text((130, 18), "triagelens — python -m triagelens.cli triage", font=font(FONT_MONO, 20), fill=MUTED)

    lines = [
        ("$ python -m triagelens.cli triage --top 5", (140, 178, 222)),
        ("", (0, 0, 0)),
        ("training on 6638 closed issues (898 closed without a fix), backend=logistic", PAPER),
        ("  fitted in 49.8s on 3000 rows / 44 features", MUTED),
        ("scoring 1716 open issues...", PAPER),
        ("  warning: 31% of the backlog predates the training window", (245, 183, 77)),
        ("           (issue numbers below 10874). Their scores are extrapolation.", (245, 183, 77)),
        ("wrote report.html", PAPER),
        ("", (0, 0, 0)),
        ("  88.2%  #14638  Allow (formatter?) config to never introduce implicitly concaten...", PAPER),
        ("  81.0%  #4368   Add config to disable S101 (assert detected) and a few others", PAPER),
        ("  80.3%  #20362  `RUF022` does not follow `lint.isort.classes` option", PAPER),
        ("  77.8%  #7568   Allow disabling sub-config required-versions", PAPER),
        ("  77.7%  #29073  B014 does not detect redundant built-in exception subclasses", PAPER),
        ("                 no triage labels yet", MUTED),
    ]
    mono = font(FONT_MONO, 19)
    y = 92
    for text, colour in lines:
        if text:
            d.text((40, y), text, font=mono, fill=colour)
        y += 42

    path = OUT / "cli-output.png"
    img.save(path, "PNG", optimize=True)
    return path, img.size


if __name__ == "__main__":
    for path, size in (cover(), cli_card()):
        kb = path.stat().st_size / 1024
        print(f"{path.name}  {size[0]}x{size[1]}  {kb:.0f} KB")