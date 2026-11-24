import numpy as np

from algo import aStar, bfs, costField, greedy, ids, ucs
from core.Config import stampLine
from core.Instance import Instance
from experiments.Stamp import printTable, writeCsv
from world import DeliveryProblem, World
from world.Heuristics import StopDistances, exactNearest, manhattanFarthest, manhattanNearest, tunnelMst

eps = 1e-9


def comparison(problem, heuristics):
    runs = [
        ("BFS", lambda: bfs(problem)),
        ("UCS", lambda: ucs(problem)),
        ("Greedy+tunnelMst", lambda: greedy(problem, heuristics["tunnelMst"])),
    ]
    runs += [(f"A*+{name}", lambda h=h: aStar(problem, h)) for name, h in heuristics.items()]
    header = ["algorithm", "found", "cost", "nodesExpanded", "runtime", "peakMemory"]
    rows, results = [], {}
    for name, run in runs:
        r = run()
        results[name] = r
        rows.append((name, r.found, r.cost, r.nodesExpanded, r.runtime, r.peakMemory))
    printTable("a) comparison", header, rows)
    writeCsv("comparison.csv", header, rows)
    return results


def consistency(problem, heuristics):
    states = sorted(costField(problem))
    edges = [(n, c, n2) for n in states for _, n2, c in problem.successors(n)]
    header = ["heuristic", "states", "edges", "violations", "maxViolation"]
    rows = []
    for name, h in heuristics.items():
        hv = {n: h(n) for n in states}
        violations, maxViolation = 0, 0.0
        for n, c, n2 in edges:
            diff = hv[n] - c - hv[n2]
            if diff > eps:
                violations += 1
                maxViolation = max(maxViolation, diff)
        rows.append((name, len(states), len(edges), violations, maxViolation))
    printTable("b) consistency h(n) <= c(n,n') + h(n')", header, rows)
    writeCsv("consistency.csv", header, rows)
    return states


def admissibility(problem, world, instance, heuristics, states):
    rng = np.random.default_rng(int(instance.instanceId, 16))
    picks = sorted(rng.choice(len(states), 30, replace=False))
    samples = [states[i] for i in picks]
    trueCosts = [
        ucs(DeliveryProblem(world, start=cell, deliveredMask=mask, carriedMask=carried)).cost
        for cell, mask, carried in samples
    ]
    header = ["heuristic", "samples", "overestimates", "maxOvershoot"]
    rows = []
    for name, h in heuristics.items():
        over = [h(s) - t for s, t in zip(samples, trueCosts, strict=True)]
        rows.append((name, len(samples), sum(d > eps for d in over), max(over)))
    printTable("c) admissibility on 30 sampled states", header, rows)
    writeCsv("admissibility.csv", header, rows)


def exactAdmissibility(instance, distances, heuristics):
    """One pending stop i, carried: true cost-to-go is exactly distances.toNode[i][cell]."""
    full = (1 << len(instance.stops)) - 1
    records = {name: [] for name in heuristics}
    for i, stop in enumerate(instance.stops):
        mask = full ^ (1 << i)
        for cell, hStar in distances.toNode[i].items():
            for name, h in heuristics.items():
                value = h((cell, mask, 1 << i))
                records[name].append((value - hStar, cell, stop, value, hStar))
    header = [
        "heuristic",
        "checks",
        "overestimates",
        "maxOvershoot",
        "worstCell",
        "worstStop",
        "worstH",
        "worstHStar",
    ]
    rows = []
    for name, recs in records.items():
        worst = max(recs, key=lambda r: r[0])
        count = sum(r[0] > eps for r in recs)
        rows.append((name, len(recs), count, worst[0], str(worst[1]), str(worst[2]), worst[3], worst[4]))
    printTable("f) exact admissibility on single-undelivered masks", header, rows)
    writeCsv("admissibility_exact.csv", header, rows)
    for name in ("manhattanNearest", "manhattanFarthest"):
        worst = max(records[name], key=lambda r: r[0])
        print(f"worst {name}: cell={worst[1]} stop={worst[2]} h={worst[3]:.4g} h*={worst[4]:.4g}")


def inadmissible(problem, distances, optimal):
    header = ["weight", "found", "cost", "gap", "gapPercent", "nodesExpanded", "runtime"]
    rows = []
    for weight in (1.0, 1.5, 2.0, 3.0):
        r = aStar(problem, tunnelMst(distances), weight)
        gap = r.cost - optimal
        rows.append((weight, r.found, r.cost, gap, 100 * gap / optimal, r.nodesExpanded, r.runtime))
    printTable("d) weighted A* with tunnelMst", header, rows)
    writeCsv("inadmissible.csv", header, rows)


def budgetRow(k, budget, name, r):
    withinBudget = r.found and r.nodesExpanded <= budget
    return (k, budget, name, r.found, withinBudget, r.cost, r.nodesExpanded, r.peakMemory, r.runtime)


def budgetComparison(world, instance, distances):
    header = ["stops", "budget", "algorithm", "found", "withinBudget"] + [
        "cost",
        "nodesExpanded",
        "peakMemory",
        "runtime",
    ]
    rows = []
    order = sorted(range(len(instance.stops)), key=lambda i: distances.cellCost(instance.start, i))
    for k in (1, 2):
        subset = tuple(instance.stops[i] for i in order[:k])
        sub = DeliveryProblem(world, stops=subset)
        astar = aStar(sub, tunnelMst(distances, stops=subset))
        depth = len(bfs(sub).actions) + 5
        for budget in (1_000, 10_000, 100_000):
            rows.append(budgetRow(k, budget, "IDS", ids(sub, maxDepth=depth, maxNodes=budget)))
            rows.append(budgetRow(k, budget, "A*+tunnelMst", astar))
    printTable(
        "e) IDS vs A* under node budgets (A* has no budget; withinBudget = nodes <= budget)", header, rows
    )
    writeCsv("ids_vs_astar.csv", header, rows)


def main():
    print(stampLine)
    instance = Instance.load()
    world = World(instance)
    problem = DeliveryProblem(world)
    distances = StopDistances(world)
    heuristics = {
        "manhattanNearest": manhattanNearest(world),
        "manhattanFarthest": manhattanFarthest(world),
        "tunnelMst": tunnelMst(distances),
        "exactNearest": exactNearest(distances),
    }
    results = comparison(problem, heuristics)
    states = consistency(problem, heuristics)
    admissibility(problem, world, instance, heuristics, states)
    exactAdmissibility(instance, distances, heuristics)
    inadmissible(problem, distances, results["UCS"].cost)
    budgetComparison(world, instance, distances)


if __name__ == "__main__":
    main()
