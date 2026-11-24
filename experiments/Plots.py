import matplotlib.pyplot as plt

from core.Config import stampLine
from experiments.Stamp import readCsv, saveFigure


def column(rows, header, name):
    index = header.index(name)
    return [float(row[index]) for row in rows]


def labels(rows, header, name):
    index = header.index(name)
    return [row[index] for row in rows]


def nodesByAlgorithm():
    header, rows = readCsv("comparison.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    names = labels(rows, header, "algorithm")
    nodes = column(rows, header, "nodesExpanded")
    ax.bar(names, nodes, color="steelblue")
    ax.set_yscale("log")
    ax.set_ylabel("nodes expanded (log scale)")
    ax.set_title("Nodes expanded per algorithm, full 6-stop problem")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for i, value in enumerate(nodes):
        ax.text(i, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
    return fig


def weightedAStar():
    header, rows = readCsv("inadmissible.csv")
    weights = column(rows, header, "weight")
    gaps = column(rows, header, "gapPercent")
    nodes = column(rows, header, "nodesExpanded")
    fig, (axGap, axNodes) = plt.subplots(1, 2, figsize=(10, 4.5))
    axGap.plot(weights, gaps, marker="o", color="firebrick")
    axGap.set_xlabel("weight w in f = g + w*h")
    axGap.set_ylabel("cost gap vs optimal (%)")
    axGap.set_title("Solution quality")
    axNodes.plot(weights, nodes, marker="o", color="steelblue")
    axNodes.set_yscale("log")
    axNodes.set_xlabel("weight w in f = g + w*h")
    axNodes.set_ylabel("nodes expanded (log scale)")
    axNodes.set_title("Search effort")
    fig.suptitle("Weighted A* with tunnelMst")
    return fig


def admissibilityExact():
    header, rows = readCsv("admissibility_exact.csv")
    names = labels(rows, header, "heuristic")
    over = column(rows, header, "overestimates")
    checks = column(rows, header, "checks")
    worst = column(rows, header, "maxOvershoot")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, over, color=["firebrick" if o else "seagreen" for o in over])
    for bar, o, total, w in zip(bars, over, checks, worst, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            o,
            f"{o:.0f}/{total:.0f}\nmax +{w:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("states with h > h* (of 7656 checks)")
    ax.set_ylim(0, max(over) * 1.3 + 1)
    ax.set_title("Exact admissibility, one pending carried stop")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    return fig


def idsVsAStar():
    header, rows = readCsv("ids_vs_astar.csv")
    stops = sorted({int(float(r[header.index("stops")])) for r in rows})
    fig, axes = plt.subplots(1, len(stops), figsize=(10, 4.5), squeeze=False)
    for ax, k in zip(axes[0], stops, strict=True):
        part = [r for r in rows if int(float(r[header.index("stops")])) == k]
        ids = [r for r in part if r[header.index("algorithm")] == "IDS"]
        astar = [r for r in part if r[header.index("algorithm")] != "IDS"]
        budgets = column(ids, header, "budget")
        ax.plot(budgets, column(ids, header, "nodesExpanded"), marker="o", label="IDS (node budget)")
        ax.axhline(
            column(astar, header, "nodesExpanded")[0],
            color="seagreen",
            linestyle="--",
            label="A*+tunnelMst (no budget)",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("maxNodes budget")
        ax.set_ylabel("nodes expanded")
        found = [r[header.index("found")] for r in ids]
        ax.set_title(f"{k} stop(s): IDS found = {found}", fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("IDS vs A* on subproblems")
    return fig


def main():
    print(stampLine)
    for name, make in [
        ("nodes_by_algorithm.png", nodesByAlgorithm),
        ("weighted_astar.png", weightedAStar),
        ("admissibility_exact.png", admissibilityExact),
        ("ids_vs_astar.png", idsVsAStar),
    ]:
        print("saved", saveFigure(make(), name))


if __name__ == "__main__":
    main()
