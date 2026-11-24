import csv
import itertools
import time
from pathlib import Path

import numpy as np

from algo.AlphaBeta import alphaBeta
from algo.AStar import aStar
from algo.Minimax import minimax
from core.Config import stampLine
from core.Instance import Instance
from game.Episode import drawMultiplier, runEpisode
from game.Evaluation import Evaluation, defaultWeights, manhattan
from game.MoveOrdering import orderMoves
from game.Policies import BlockerRival, DeliveryAgent, PassiveRival
from game.RivalGame import RivalGame, defaultRadius, windowOf
from world.DeliveryProblem import DeliveryProblem
from world.Heuristics import StopDistances, tunnelMst
from world.Simulator import rivalStartFor
from world.World import World

outDir = Path(__file__).resolve().parents[1] / "outputs"
positionCount = 10


def printTable(title, header, rows):
    print(f"\n== {title} ==")
    print(stampLine)
    lines = [header] + [[f"{v:.4g}" if isinstance(v, float) else str(v) for v in row] for row in rows]
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


def reachableCells(world: World):
    start = world.instance.start
    seen, frontier = {start}, [start]
    while frontier:
        cell = frontier.pop()
        for _, nxt, _ in world.neighbors(cell):
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return sorted(seen)


def buildPositions(world: World):
    """Seeded roots (agent, rival, deliveredMask, carriedMask): rival 2-4 cells from the agent."""
    rng = np.random.default_rng(int(world.instance.instanceId, 16))
    cells = reachableCells(world)
    full = (1 << len(world.instance.stops)) - 1
    roots = []
    while len(roots) < positionCount:
        agent = cells[int(rng.integers(len(cells)))]
        candidates = sorted(c for c in windowOf(world, agent, defaultRadius) if 2 <= manhattan(agent, c) <= 4)
        if not candidates:
            continue
        rival = candidates[int(rng.integers(len(candidates)))]
        delivered = int(rng.integers(0, full))
        carried = int(rng.integers(0, full + 1)) & ~delivered & full
        roots.append((agent, rival, delivered, carried))
    return roots


def nodeTable(world: World, roots, distances):
    evaluation = Evaluation(distances, defaultWeights, world)
    maxDepth = world.instance.adversaryDepth
    header = [
        "depth",
        "minimaxNodes",
        "alphaBetaNodes",
        "orderedNodes",
        "pctSavedAlphaBeta",
        "pctSavedOrdered",
    ]
    rows = []
    for depth in range(2, maxDepth + 1):
        mm, ab, abo = [], [], []
        for root in roots:
            game = RivalGame(world, root, evaluation=evaluation)
            resMinimax = minimax(game, root, depth)
            resAlpha = alphaBeta(game, root, depth)
            resOrdered = alphaBeta(game, root, depth, order=orderMoves(game))
            assert resMinimax.value == resAlpha.value == resOrdered.value, (root, depth)
            mm.append(resMinimax.nodes)
            ab.append(resAlpha.nodes)
            abo.append(resOrdered.nodes)
        savedAb = np.mean([100 * (1 - a / m) for a, m in zip(ab, mm, strict=True)])
        savedOrd = np.mean([100 * (1 - o / m) for o, m in zip(abo, mm, strict=True)])
        rows.append((depth, np.mean(mm), np.mean(ab), np.mean(abo), savedAb, savedOrd))
    return header, rows


tuneGrid = {"rivalDistance": [0.25, 1.0], "theftExposure": [-4.0, -8.0], "blockage": [-1.0, -3.0]}
tuneSeeds = 6
evalSeeds = 30
agentDepth = 4
z95 = 1.96  # normal approximation for all 95% intervals


def seedBase(world: World) -> int:
    return int(world.instance.instanceId, 16)


def playGame(world, distances, weights, k, starts, optimalSteps, rivalKind="blocker"):
    """Scenario k: rival start from default_rng(base + k); episode slips from the same seed."""
    if k not in starts:
        starts[k] = rivalStartFor(world, np.random.default_rng(seedBase(world) + k))
    agent = DeliveryAgent(world, distances, weights, depth=agentDepth)
    rival = BlockerRival(world, distances) if rivalKind == "blocker" else PassiveRival()
    rng = np.random.default_rng(seedBase(world) + k)
    return runEpisode(world, rng, agent, rival, starts[k], drawMultiplier * optimalSteps)


def tuneWeights(world, distances, optimalSteps, starts):
    keys = list(tuneGrid)
    header = [*keys, "winRate", "meanDelivered", "meanCost"]
    rows, scored = [], []
    for values in itertools.product(*tuneGrid.values()):
        weights = {**defaultWeights, **dict(zip(keys, values, strict=True))}
        results = [playGame(world, distances, weights, k, starts, optimalSteps) for k in range(tuneSeeds)]
        winRate = sum(r["outcome"] == "win" for r in results) / tuneSeeds
        meanDelivered = float(np.mean([r["delivered"] for r in results]))
        meanCost = float(np.mean([r["cost"] for r in results]))
        rows.append((*values, winRate, meanDelivered, meanCost))
        scored.append(((winRate, meanDelivered, -meanCost), values))
    bestValues = max(scored, key=lambda item: item[0])[1]
    return header, rows, dict(zip(keys, bestValues, strict=True))


def wilson(successes: int, n: int) -> tuple[float, float]:
    p = successes / n
    denom = 1 + z95 * z95 / n
    center = (p + z95 * z95 / (2 * n)) / denom
    half = z95 * np.sqrt(p * (1 - p) / n + z95 * z95 / (4 * n * n)) / denom
    return max(center - half, 0.0), min(center + half, 1.0)


def evaluationSummary(results):
    n = len(results)
    header = ["metric", "value", "ci95"]
    rows = []
    for outcome in ("win", "draw", "loss"):
        count = sum(r["outcome"] == outcome for r in results)
        low, high = wilson(count, n)
        rows.append((f"{outcome} rate", f"{count}/{n} = {count / n:.3f}", f"[{low:.3f}, {high:.3f}]"))
    for metric in ("cost", "delivered", "thefts", "slips"):
        values = [r[metric] for r in results]
        half = z95 * np.std(values, ddof=1) / np.sqrt(n)
        rows.append((f"mean {metric}", f"{np.mean(values):.2f}", f"± {half:.2f}"))
    return header, rows


def replanningSummary(instance, results):
    steps = sum(r["steps"] for r in results)
    replans = sum(r["replans"] for r in results)
    slipReplans = sum(r["slipReplans"] for r in results)
    blockReplans = sum(r["blockReplans"] for r in results)
    sidesteps = sum(r["sidesteps"] for r in results)
    agentStays = sum(r["agentStays"] for r in results)
    rivalStays = sum(r["rivalStays"] for r in results)
    slips = sum(r["slips"] for r in results)
    directional = sum(r["directionalSteps"] for r in results)
    expected = sum(r["expectedSlips"] for r in results)
    contest = sum(r["contestTurns"] for r in results)
    planned = np.mean([r["plannedCost"] for r in results])
    actual = np.mean([r["cost"] for r in results])
    notes = [
        f"- slips: observed {slips / directional:.3f} vs {expected / directional:.3f} from the "
        "load-dependent rate; the deterministic plan assumes none",
        f"- the deterministic plan costs {planned:.1f} vs actual {actual:.1f} on average",
        "- the game model has no theft cooldown; the simulator uses 10",
        f"- {contest / steps:.1%} of steps are contest turns, where the planned route is discarded",
    ]
    print("\nnotes:\n" + "\n".join(notes))
    header = ["metric", "value"]
    rows = [
        ("mean replans per game", f"{replans / len(results):.2f}"),
        ("  of which slip-triggered", f"{slipReplans / len(results):.2f}"),
        ("  of which rival-blocked", f"{blockReplans / len(results):.2f}"),
        ("mean sidesteps per game", f"{sidesteps / len(results):.2f}"),
        ("fraction of steps replanning", f"{replans / steps:.3f}"),
        ("fraction of steps in contest", f"{contest / steps:.3f}"),
        ("fraction of agent turns that stay", f"{agentStays / steps:.3f}"),
        ("fraction of rival turns that stay", f"{rivalStays / steps:.3f}"),
        ("mean plannedCost", f"{planned:.1f}"),
        ("mean extraCost", f"{np.mean([r['extraCost'] for r in results]):.1f}"),
        ("observed slip rate (per directional step)", f"{slips / directional:.3f}"),
        ("expected slip rate at each step's load", f"{expected / directional:.3f}"),
        (f"slipFor({len(instance.stops)}) full load", f"{instance.slipFor(len(instance.stops)):.3f}"),
    ]
    return header, rows


def main():
    started = time.perf_counter()
    instance = Instance.load()
    world = World(instance)
    distances = StopDistances(world)

    roots = buildPositions(world)
    header, rows = nodeTable(world, roots, distances)
    title = f"1) node counts, window radius {defaultRadius}, mean over {positionCount} seeded positions"
    printTable(title, header, rows)
    writeCsv("game_nodes.csv", header, rows)

    optimalSteps = len(aStar(DeliveryProblem(world), tunnelMst(distances)).actions)
    starts = {}
    t0 = time.perf_counter()
    header, rows, best = tuneWeights(world, distances, optimalSteps, starts)
    printTable(f"2) weight tuning, {tuneSeeds} games per config, best {best}", header, rows)
    print(f"section 2 took {time.perf_counter() - t0:.1f}s")
    writeCsv("game_tuning.csv", header, rows)

    weights = {**defaultWeights, **best}
    blocker = [playGame(world, distances, weights, k, starts, optimalSteps) for k in range(evalSeeds)]
    passive = [
        playGame(world, distances, weights, k, starts, optimalSteps, "passive") for k in range(evalSeeds)
    ]
    header, rows = evaluationSummary(blocker)
    printTable(f"3a) {evalSeeds} games, tuned weights vs BlockerRival", header, rows)
    writeCsv("game_eval.csv", header, rows)
    header, rows = evaluationSummary(passive)
    printTable(f"3b) diagnostic: {evalSeeds} games, same seeds vs PassiveRival", header, rows)
    writeCsv("game_eval_passive.csv", header, rows)

    header, rows = replanningSummary(instance, blocker)
    printTable("4) replanning under slip (blocker games)", header, rows)
    writeCsv("game_replanning.csv", header, rows)
    print(f"\nruntime {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
