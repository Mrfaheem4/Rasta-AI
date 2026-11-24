import csv
from pathlib import Path

import numpy as np

from algo.AlphaBeta import alphaBeta
from algo.Minimax import minimax
from core.Config import stampLine
from core.Instance import Instance
from game.Evaluation import Evaluation, defaultWeights, manhattan
from game.MoveOrdering import orderMoves
from game.RivalGame import RivalGame, defaultRadius, windowOf
from world.Heuristics import StopDistances
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


def main():
    instance = Instance.load()
    world = World(instance)
    distances = StopDistances(world)
    roots = buildPositions(world)
    header, rows = nodeTable(world, roots, distances)
    title = f"1) node counts, window radius {defaultRadius}, mean over {positionCount} seeded positions"
    printTable(title, header, rows)
    writeCsv("game_nodes.csv", header, rows)


if __name__ == "__main__":
    main()
