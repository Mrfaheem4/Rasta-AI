import time

import numpy as np


def buildCostMap(grid, terrainCost):
    """Cost of entering each cell; np.inf for blocked cells (grid == -1)."""
    costs = np.array(list(terrainCost.values()), dtype=float)
    return np.where(grid >= 0, costs[grid], np.inf)


def buildCostMapLoop(grid, terrainCost):
    values = list(terrainCost.values())
    rows = [[np.inf if t < 0 else values[t] for t in row] for row in grid.tolist()]
    return np.array(rows, dtype=float)


def timeCall(fn, repeats):
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def compareOn(grid, terrainCost, repeats):
    vec = buildCostMap(grid, terrainCost)
    assert np.array_equal(vec, buildCostMapLoop(grid, terrainCost)), "vectorised and loop disagree"
    vecTime = timeCall(lambda: buildCostMap(grid, terrainCost), repeats)
    loopTime = timeCall(lambda: buildCostMapLoop(grid, terrainCost), repeats)
    return {"vectorised": vecTime, "loop": loopTime, "speedup": loopTime / vecTime}


def benchmark(instance, repeats=50) -> dict:
    """Mean seconds per build on the real instance grid and on a random 400x400 grid."""
    rng = np.random.default_rng(0)
    big = rng.choice(4, size=(400, 400), p=[0.4, 0.3, 0.2, 0.1])
    big[rng.random((400, 400)) < 0.2] = -1
    return {
        "instance": compareOn(instance.grid, instance.terrainCost, repeats),
        "random400": compareOn(big, instance.terrainCost, repeats),
    }
