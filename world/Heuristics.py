import numpy as np

from algo.CostField import costField
from world.Tunnels import tunnelCost
from world.World import World, moves


class ReverseGrid:
    """Backward search from a target cell. Successors are predecessors, costed by the forward edge."""

    def __init__(self, world: World, target):
        self.world = world
        self.target = target

    def initialState(self):
        return self.target

    def isGoal(self, state) -> bool:
        return False

    def successors(self, cell):
        world = self.world
        r, c = cell
        enter = float(world.costMap[r, c])
        for _, dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < world.rows and 0 <= nc < world.cols and np.isfinite(world.costMap[nr, nc]):
                yield "reverse", (nr, nc), enter
        if cell in world.tunnels:
            yield "teleport", world.tunnels[cell], tunnelCost


class StopDistances:
    """Exact tunnel-aware costs over stops plus the depot (last index).

    toNode[i][cell] = cell to node i; matrix[i, j] = node i to node j; symmetric = min(matrix, matrix.T).
    """

    def __init__(self, world: World):
        self.stops = world.instance.stops
        self.depotIndex = len(self.stops)
        nodes = [*self.stops, world.instance.start]
        self.toNode = [costField(ReverseGrid(world, node)) for node in nodes]
        n = len(nodes)
        self.matrix = np.array([[self.toNode[j].get(nodes[i], np.inf) for j in range(n)] for i in range(n)])
        self.symmetric = np.minimum(self.matrix, self.matrix.T)

    def cellCost(self, cell, i) -> float:
        return self.toNode[i].get(cell, np.inf)


def activeIndices(allStops, stops) -> list[int]:
    return [i for i, cell in enumerate(allStops) if stops is None or cell in stops]


def primWeight(weights, nodes) -> float:
    """Prim's MST weight over the given node indices."""
    if len(nodes) < 2:
        return 0.0
    best = {v: weights[nodes[0], v] for v in nodes[1:]}
    total = 0.0
    while best:
        v = min(best, key=best.get)
        total += best.pop(v)
        for u in best:
            best[u] = min(best[u], weights[v, u])
    return float(total)


def pendingNodes(state, active, depotIndex):
    """Undelivered active stops, plus the depot if any of them is not carried (must reload first)."""
    _, delivered, carried = state
    pending = tuple(i for i in active if not (delivered >> i) & 1)
    if any(not (carried >> i) & 1 for i in pending):
        return pending + (depotIndex,)
    return pending


def manhattanHeuristic(world: World, pick, stops=None):
    active = activeIndices(world.instance.stops, stops)
    minCost = min(world.instance.terrainCost.values())

    def h(state) -> float:
        (r, c), delivered, _ = state
        pending = [world.instance.stops[i] for i in active if not (delivered >> i) & 1]
        if not pending:
            return 0.0
        return minCost * pick(abs(r - tr) + abs(c - tc) for tr, tc in pending)

    return h


def manhattanNearest(world: World, stops=None):
    return manhattanHeuristic(world, min, stops)


def manhattanFarthest(world: World, stops=None):
    return manhattanHeuristic(world, max, stops)


def tunnelMst(distances: StopDistances, stops=None):
    """Exact cost to the nearest pending node + MST weight over pending nodes (cached per node set)."""
    active = activeIndices(distances.stops, stops)
    cache = {}

    def h(state) -> float:
        nodes = pendingNodes(state, active, distances.depotIndex)
        if not nodes:
            return 0.0
        if nodes not in cache:
            cache[nodes] = primWeight(distances.symmetric, nodes)
        return min(distances.cellCost(state[0], i) for i in nodes) + cache[nodes]

    return h


def exactNearest(distances: StopDistances, stops=None):
    """Ablation: tunnelMst without the MST term."""
    active = activeIndices(distances.stops, stops)

    def h(state) -> float:
        nodes = pendingNodes(state, active, distances.depotIndex)
        if not nodes:
            return 0.0
        return min(distances.cellCost(state[0], i) for i in nodes)

    return h
