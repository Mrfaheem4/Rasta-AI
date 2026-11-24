import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from algo import aStar
from core.Config import stampLine
from core.Instance import Instance
from experiments.Stamp import printTable, saveFigure, writeCsv
from world import DeliveryProblem, World
from world.CostMap import benchmark
from world.Heuristics import StopDistances, tunnelMst

terrainNames = ["road", "grass", "mud", "ford"]
terrainColours = ["#e0d8b0", "#9ccc65", "#8d6e63", "#42a5f5"]


def benchmarkPart(instance):
    result = benchmark(instance, repeats=50)
    header = ["grid", "vectorisedSeconds", "loopSeconds", "speedup"]
    rows = [(name, r["vectorised"], r["loop"], r["speedup"]) for name, r in result.items()]
    printTable("A) cost-map benchmark", header, rows)
    writeCsv("costmap_benchmark.csv", header, rows)


def drawRoute(ax, path, actions):
    """Walked steps as solid lines; each teleport hop as a curved purple arrow with a 'T' label.
    Returns the set of tunnel pairs the route uses."""
    used = set()
    walkX, walkY = [path[0][0][1]], [path[0][0][0]]
    for prev, nxt, action in zip(path[:-1], path[1:], actions, strict=True):
        p, q = prev[0], nxt[0]
        if action == "teleport":
            used.add(frozenset((p, q)))
            ax.plot(walkX, walkY, color="black", linewidth=1.0, zorder=3)
            ax.annotate(
                "",
                xy=(q[1], q[0]),
                xytext=(p[1], p[0]),
                arrowprops=dict(
                    arrowstyle="->",
                    color="purple",
                    linewidth=1.4,
                    linestyle="--",
                    connectionstyle="arc3,rad=0.3",
                ),
                zorder=4,
            )
            ax.text(
                (p[1] + q[1]) / 2,
                (p[0] + q[0]) / 2,
                "T",
                color="purple",
                fontsize=8,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=6,
                bbox=dict(boxstyle="circle,pad=0.15", facecolor="white", edgecolor="purple"),
            )
            walkX, walkY = [q[1]], [q[0]]
        else:
            walkX.append(q[1])
            walkY.append(q[0])
    ax.plot(walkX, walkY, color="black", linewidth=1.0, zorder=3)
    return used


def gridFigure(instance, world):
    cmap = ListedColormap(terrainColours)
    cmap.set_bad("black")
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(np.ma.masked_less(instance.grid, 0), cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    result = aStar(DeliveryProblem(world), tunnelMst(StopDistances(world)))
    used = drawRoute(ax, result.path, result.actions)
    for a, b in world.tunnels.items():
        if a < b and frozenset((a, b)) not in used:
            ax.plot([a[1], b[1]], [a[0], b[0]], ":", color="purple", linewidth=1.0, zorder=2)
    ends = list(world.tunnels)
    ax.scatter([c for _, c in ends], [r for r, _ in ends], s=30, color="purple", zorder=5)
    for i, (r, c) in enumerate(instance.stops):
        ax.scatter(c, r, s=160, color="white", edgecolor="black", zorder=7)
        ax.text(c, r, str(i + 1), ha="center", va="center", fontsize=9, zorder=8)
    ax.scatter(instance.start[1], instance.start[0], marker="*", s=260, color="red", zorder=8)
    handles = [
        Patch(color=colour, label=name) for name, colour in zip(terrainNames, terrainColours, strict=True)
    ]
    handles += [
        Patch(color="black", label="blocked"),
        Line2D([], [], marker="*", color="red", linestyle="", markersize=12, label="depot (start)"),
        Line2D([], [], marker="o", color="purple", linestyle="", label="tunnel endpoint"),
        Line2D([], [], color="purple", linestyle="--", label="teleport hop on route (T)"),
        Line2D([], [], color="purple", linestyle=":", label="unused tunnel pair"),
        Line2D([], [], color="black", label=f"A*(tunnelMst) route, cost {result.cost:g}"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    ax.set_title(f"Instance {instance.instanceId}: terrain, stops 1..6, tunnels", fontsize=11)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    return fig, result.cost


def main():
    print(stampLine)
    instance = Instance.load()
    world = World(instance)
    benchmarkPart(instance)
    fig, cost = gridFigure(instance, world)
    path = saveFigure(fig, "grid_map.png")
    print(f"route cost {cost:g}; saved {path}")


if __name__ == "__main__":
    main()
