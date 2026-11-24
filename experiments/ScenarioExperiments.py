import math
import time
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from algo import aStar, greedy, ucs
from core.Config import stampLine
from core.Instance import Instance
from experiments.Stamp import printTable, saveFigure, writeCsv
from world import DeliveryProblem, World
from world.Heuristics import StopDistances, exactNearest, manhattanFarthest, manhattanNearest, tunnelMst

scenarioCount = 30
tCrit = 2.045  # two-sided 95% t value, 29 degrees of freedom
eps = 1e-9
samplesPerAblation = 30


def reachableCells(distances):
    return sorted(distances.toNode[0])  # cells that can reach stop 1: the main connected component


def makeScenario(base, baseWorld, baseDistances, k):
    """Same grid, terrain and tunnels as the CMS instance; random reachable start and 4-6 stops."""
    rng = np.random.default_rng(int(base.instanceId, 16) + 1000 + k)
    tunnelEnds = set(baseWorld.tunnels)
    pool = [c for c in reachableCells(baseDistances) if c not in tunnelEnds]
    start = pool[int(rng.integers(len(pool)))]
    others = [c for c in pool if c != start]
    picks = rng.choice(len(others), int(rng.integers(4, 7)), replace=False)
    inst = replace(base, start=start, stops=tuple(others[i] for i in picks))
    world = World(inst)
    world.tunnels = dict(baseWorld.tunnels)
    return inst, world


def meanCi(values):
    arr = np.asarray(values, dtype=float)
    ci = tCrit * arr.std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return float(arr.mean()), float(ci)


def configurations(world, distances):
    problem = DeliveryProblem(world)
    tunnel = tunnelMst(distances)
    return [
        ("UCS", lambda: ucs(problem)),
        ("A*+manhattanNearest", lambda: aStar(problem, manhattanNearest(world))),
        ("A*+manhattanFarthest", lambda: aStar(problem, manhattanFarthest(world))),
        ("A*+exactNearest", lambda: aStar(problem, exactNearest(distances))),
        ("A*+tunnelMst", lambda: aStar(problem, tunnel)),
        ("Greedy+tunnelMst", lambda: greedy(problem, tunnel)),
        ("wA*(w=2)+tunnelMst", lambda: aStar(problem, tunnel, 2.0)),
    ]


def routeStudy(base, baseWorld, baseDistances):
    records, failures, ablationB, gaps = [], [], [], []
    for k in range(scenarioCount):
        inst, world = makeScenario(base, baseWorld, baseDistances, k)
        distances = StopDistances(world)
        results = {name: run() for name, run in configurations(world, distances)}
        optimal = results["UCS"].cost
        for name, r in results.items():
            records.append(
                (k, name, r.nodesExpanded, r.runtime, r.peakMemory, r.cost, abs(r.cost - optimal) < eps)
            )
        initial = DeliveryProblem(world).initialState()
        for name, h in [
            ("manhattanNearest", manhattanNearest(world)),
            ("manhattanFarthest", manhattanFarthest(world)),
        ]:
            over = h(initial) - optimal
            if over > eps:
                failures.append((k, f"{name} overestimates the root", f"h={h(initial):g} > h*={optimal:g}"))
        ratio = results["Greedy+tunnelMst"].cost / optimal
        if ratio > 1.5:
            failures.append((k, "greedy cost > 1.5x optimal", f"ratio {ratio:.2f}"))
        gaps.append((100 * (results["wA*(w=2)+tunnelMst"].cost - optimal) / optimal, k))
        noTunnelWorld = World(inst)
        noTunnelWorld.tunnels = {}
        noTunnel = StopDistances(noTunnelWorld)
        b = aStar(DeliveryProblem(world), tunnelMst(noTunnel))
        ablationB.append((b.nodesExpanded, abs(b.cost - optimal) < eps))
    worstGap, worstK = max(gaps)
    failures.append((worstK, "worst weighted A* gap (w=2)", f"{worstGap:.2f}%"))
    return records, failures, ablationB


def depotStudy(base, baseWorld, baseDistances):
    """States with one stolen (uncarried) pending parcel: tunnelMst with vs without the depot node."""
    rng = np.random.default_rng(4242)
    cells = reachableCells(baseDistances)
    full = tunnelMst(baseDistances)

    def noDepot(state):
        return full((state[0], state[1], baseWorld.fullMask))

    variants = {"full tunnelMst": full, "(c) no depot term": noDepot}
    nodes = {name: [] for name in variants}
    overs = {name: 0 for name in variants}
    for _ in range(samplesPerAblation):
        cell = cells[int(rng.integers(len(cells)))]
        delivered = 0
        while bin(baseWorld.fullMask & ~delivered).count("1") < 2:
            delivered = 0
            for i in range(len(base.stops)):
                if rng.random() < 0.3:
                    delivered |= 1 << i
        pending = [i for i in range(len(base.stops)) if not (delivered >> i) & 1]
        stolen = pending[int(rng.integers(len(pending)))]
        carried = baseWorld.fullMask & ~delivered & ~(1 << stolen)
        problem = DeliveryProblem(baseWorld, start=cell, deliveredMask=delivered, carriedMask=carried)
        hStar = ucs(problem).cost
        for name, h in variants.items():
            nodes[name].append(aStar(problem, h).nodesExpanded)
            overs[name] += h((cell, delivered, carried)) - hStar > eps
    return nodes, overs


def main():
    print(stampLine)
    started = time.perf_counter()
    base = Instance.load()
    baseWorld = World(base)
    baseDistances = StopDistances(baseWorld)

    records, failures, ablationB = routeStudy(base, baseWorld, baseDistances)
    names = list(dict.fromkeys(r[1] for r in records))
    stats = {}
    for name in names:
        rows = [r for r in records if r[1] == name]
        stats[name] = {
            "nodes": meanCi([r[2] for r in rows]),
            "runtime": meanCi([r[3] for r in rows]),
            "peak": meanCi([r[4] / 1e6 for r in rows]),
            "cost": meanCi([r[5] for r in rows]),
            "optimal": 100 * sum(r[6] for r in rows) / len(rows),
        }
    header = [
        "config",
        "nodesExpanded",
        "nodesCI95",
        "runtimeS",
        "runtimeCI95",
        "peakMB",
        "peakCI95",
        "cost",
        "costCI95",
        "optimalPct",
    ]
    tableRows = [
        (
            name,
            s["nodes"][0],
            s["nodes"][1],
            s["runtime"][0],
            s["runtime"][1],
            s["peak"][0],
            s["peak"][1],
            s["cost"][0],
            s["cost"][1],
            s["optimal"],
        )
        for name, s in stats.items()
    ]
    printTable(
        f"Route configurations over {scenarioCount} scenarios (mean, 95% CI, t={tCrit})", header, tableRows
    )
    writeCsv(
        "scenarios_route.csv",
        ["scenario", "config", "nodesExpanded", "runtime", "peakMemory", "cost", "optimal"],
        [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in records],
    )

    depotNodes, depotOver = depotStudy(base, baseWorld, baseDistances)
    exactRow = stats["A*+exactNearest"]
    fullRow = stats["A*+tunnelMst"]
    ablationB_nodes = meanCi([n for n, _ in ablationB])
    ablationRows = [
        ("full tunnelMst", fullRow["nodes"][0], fullRow["nodes"][1], fullRow["optimal"], None, None),
        (
            "(a) no MST term = exactNearest",
            exactRow["nodes"][0],
            exactRow["nodes"][1],
            exactRow["optimal"],
            None,
            None,
        ),
        (
            "(b) no tunnel awareness",
            ablationB_nodes[0],
            ablationB_nodes[1],
            100 * sum(ok for _, ok in ablationB) / len(ablationB),
            None,
            None,
        ),
    ]
    for name, nodes in depotNodes.items():
        m, ci = meanCi(nodes)
        ablationRows.append(
            (name, m, ci, None, depotOver[name], f"{samplesPerAblation} stolen-parcel states")
        )
    ablationHeader = ["ablation", "nodesExpanded", "nodesCI95", "optimalPct", "overestimates", "note"]
    printTable("Ablations", ablationHeader, ablationRows)
    writeCsv("scenarios_ablation.csv", ablationHeader, ablationRows)

    failureRows = [(f"scenario {k}", issue, detail) for k, issue, detail in failures]
    printTable(
        "Failure analysis (scenario ids cite the report)", ["scenario", "issue", "detail"], failureRows
    )
    writeCsv("scenarios_failures.csv", ["scenario", "issue", "detail"], failureRows)

    fig, ax = plt.subplots(figsize=(10, 5))
    means = [stats[n]["nodes"][0] for n in names]
    cis = [stats[n]["nodes"][1] for n in names]
    lower = [min(c, 0.99 * m) for m, c in zip(means, cis, strict=True)]
    ax.errorbar(range(len(names)), means, yerr=[lower, cis], fmt="o", capsize=4)
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("nodes expanded (mean, 95% CI)")
    ax.set_title(f"Nodes expanded per configuration over {scenarioCount} generated scenarios")
    print("saved", saveFigure(fig, "scenarios_nodes.png"))
    print(f"runtime {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()
