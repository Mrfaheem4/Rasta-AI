import csv
from pathlib import Path

import matplotlib

from core.Config import stampLine

matplotlib.use("Agg")

outDir = Path(__file__).resolve().parents[1] / "outputs"


def fmt(value):
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def printTable(title, header, rows):
    print(f"\n== {title} ==")
    print(stampLine)
    lines = [header] + [[fmt(v) for v in row] for row in rows]
    widths = [max(len(line[i]) for line in lines) for i in range(len(header))]
    for line in lines:
        print("  ".join(v.ljust(w) for v, w in zip(line, widths, strict=True)))


def writeCsv(name, header, rows):
    outDir.mkdir(exist_ok=True)
    with open(outDir / name, "w", newline="", encoding="utf-8") as f:
        f.write(f"# {stampLine}\n")
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def readCsv(name):
    """Returns (header, rows as lists of strings); skips the '#' stamp line."""
    with open(outDir / name, encoding="utf-8") as f:
        reader = csv.reader(line for line in f if not line.startswith("#"))
        header = next(reader)
        return header, list(reader)


def saveFigure(fig, name):
    figDir = outDir / "figures"
    figDir.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, 0.005, stampLine, fontsize=7, ha="left", va="bottom")
    fig.savefig(figDir / name, dpi=150, bbox_inches="tight")
    return figDir / name
