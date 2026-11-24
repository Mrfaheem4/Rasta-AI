import math
from dataclasses import replace
from functools import cache
from typing import NamedTuple

import numpy as np

from algo import aStar, bfs, greedy, ucs
from core.Config import stampLine
from core.Instance import Instance
from world import DeliveryProblem, World
from world.Heuristics import StopDistances, exactNearest, manhattanFarthest, manhattanNearest, tunnelMst
from world.Simulator import rivalStartFor

algorithmNames = ["bfs", "ucs", "greedy", "astar"]
heuristicNames = ["tunnelMst", "exactNearest", "manhattanNearest", "manhattanFarthest"]
terrainNames = ["road", "grass", "mud", "ford"]


@cache
def baseInstance() -> Instance:
    return Instance.load()


class Context(NamedTuple):
    instance: Instance
    world: World
    distances: StopDistances


def allowedCell(instance: Instance, avoid: frozenset, cell) -> bool:
    grid = instance.grid[cell[0], cell[1]]
    return grid >= 0 and terrainNames[grid] not in avoid


@cache
def contextFor(avoid: frozenset) -> Context:
    """Instance, world and distances with avoided terrain costing inf; teleports touching it are dropped."""
    base = baseInstance()
    terrainCost = {name: (math.inf if name in avoid else cost) for name, cost in base.terrainCost.items()}
    instance = replace(base, terrainCost=terrainCost)
    world = World(instance)
    world.tunnels = {
        a: b
        for a, b in world.tunnels.items()
        if allowedCell(instance, avoid, a) and allowedCell(instance, avoid, b)
    }
    return Context(instance, world, StopDistances(world))


@cache
def noTunnelDistances() -> StopDistances:
    world = World(baseInstance())
    world.tunnels = {}
    return StopDistances(world)


def makeHeuristic(ctx: Context, name: str, stops):
    if name == "tunnelMst":
        return tunnelMst(ctx.distances, stops=stops)
    if name == "exactNearest":
        return exactNearest(ctx.distances, stops=stops)
    if name == "manhattanNearest":
        return manhattanNearest(ctx.world, stops=stops)
    return manhattanFarthest(ctx.world, stops=stops)


searchers = {
    "bfs": lambda problem, heuristic: bfs(problem),
    "ucs": lambda problem, heuristic: ucs(problem),
    "greedy": lambda problem, heuristic: greedy(problem, heuristic),
    "astar": lambda problem, heuristic: aStar(problem, heuristic),
}


class Restricted:
    """Problem wrapper: while an order is pending, the route may not enter other undelivered stops.
    Removing edges only raises true costs, so the admissible heuristics stay admissible."""

    def __init__(self, problem, forbidden):
        self.problem = problem
        self.forbidden = forbidden

    def initialState(self):
        return self.problem.initialState()

    def isGoal(self, state) -> bool:
        return self.problem.isGoal(state)

    def successors(self, state):
        for action, nxt, cost in self.problem.successors(state):
            if nxt[0] not in self.forbidden:
                yield action, nxt, cost


def planRoute(
    algorithm: str = "astar",
    heuristic: str = "tunnelMst",
    order=None,
    avoidTerrain=None,
    maxCost=None,
    start=None,
) -> dict:
    """Plan a full delivery route. order = stop numbers (1-based) to deliver first, in that sequence."""
    avoid = frozenset(avoidTerrain or [])
    ctx = contextFor(avoid)
    world = ctx.world
    cell = tuple(start) if start is not None else ctx.instance.start
    delivered, carried = 0, world.fullMask
    segments = [[ctx.instance.stops[n - 1]] for n in (order or [])] + [None]
    states, total = [(cell, delivered, carried)], 0.0
    actions = []
    stopCells = ctx.instance.stops
    for segment in segments:
        stops = None if segment is None else tuple(segment)
        problem = DeliveryProblem(
            world, start=cell, deliveredMask=delivered, carriedMask=carried, stops=stops
        )
        if segment is not None:
            pendingOthers = {
                c for i, c in enumerate(stopCells) if not (delivered >> i) & 1 and c != segment[0]
            }
            problem = Restricted(problem, pendingOthers)
        result = searchers[algorithm](problem, makeHeuristic(ctx, heuristic, stops))
        if not result.found:
            return {"found": False, "algorithm": algorithm, "heuristic": heuristic, "stamp": stampLine}
        states.extend(result.path[1:])
        actions.extend(result.actions)
        total += result.cost
        cell, delivered, carried = result.path[-1]
    visits, seen = [], 0
    for state in states:
        newBits = state[1] & ~seen
        visits.extend(i + 1 for i in range(len(ctx.instance.stops)) if newBits >> i & 1)
        seen |= state[1]
    terrainSteps = {}
    for state in states[1:]:
        name = terrainNames[ctx.instance.grid[state[0][0], state[0][1]]]
        terrainSteps[name] = terrainSteps.get(name, 0) + 1
    return {
        "found": True,
        "algorithm": algorithm,
        "heuristic": heuristic,
        "avoidTerrain": sorted(avoid),
        "cost": total,
        "steps": len(actions),
        "tunnelSteps": actions.count("teleport"),
        "stopVisitOrder": visits,
        "orderRespected": None if not order else visits[: len(order)] == list(order),
        "withinBudget": maxCost is None or total <= maxCost,
        "terrainSteps": terrainSteps,
    }


def simulate(steps: int = 300, seed: int = 0) -> dict:
    from game.Episode import runEpisode
    from game.Policies import BlockerRival, DeliveryAgent

    ctx = contextFor(frozenset())
    rng = np.random.default_rng(seed)
    rivalStart = rivalStartFor(ctx.world, rng)
    agent = DeliveryAgent(ctx.world, ctx.distances)
    rival = BlockerRival(ctx.world, ctx.distances)
    episode = runEpisode(ctx.world, rng, agent, rival, rivalStart, steps)
    keys = ["outcome", "cost", "steps", "slips", "thefts", "replans"]
    return {"seed": seed, **{key: episode[key] for key in keys}, "stamp": stampLine}


def tunnelShortcut(fromStop: int, toStop: int) -> dict:
    ctx = contextFor(frozenset())
    i, j = fromStop - 1, toStop - 1
    withTunnels = float(ctx.distances.matrix[i][j])
    withoutTunnels = float(noTunnelDistances().matrix[i][j])
    pairs = [[list(a), list(b)] for a, b in ctx.world.tunnels.items() if a < b]
    return {
        "fromStop": fromStop,
        "toStop": toStop,
        "costWithTunnels": withTunnels,
        "costWithoutTunnels": withoutTunnels,
        "saving": withoutTunnels - withTunnels,
        "tunnelPairs": pairs,
    }


tools = {"planRoute": planRoute, "simulate": simulate, "tunnelShortcut": tunnelShortcut}

toolSpecs = [
    {
        "name": "planRoute",
        "description": "Plan a route delivering all stops. Optional: a required stop order, avoided terrain, "
        "a cost budget and a start cell. Returns found, cost, steps, stopVisitOrder, withinBudget, "
        "terrainSteps.",
        "parameters": {
            "type": "object",
            "properties": {
                "algorithm": {"type": "string", "enum": algorithmNames},
                "heuristic": {"type": "string", "enum": heuristicNames},
                "order": {"type": "array", "items": {"type": "integer"}, "uniqueItems": True},
                "avoidTerrain": {
                    "type": "array",
                    "items": {"type": "string", "enum": terrainNames},
                    "uniqueItems": True,
                },
                "maxCost": {"type": "number"},
                "start": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
            },
            "required": [],
        },
    },
    {
        "name": "simulate",
        "description": "Run one stochastic episode (slips, rival thefts) with the agent policy. "
        "Returns outcome, cost, steps, slips, thefts, replans.",
        "parameters": {
            "type": "object",
            "properties": {"steps": {"type": "integer"}, "seed": {"type": "integer"}},
            "required": [],
        },
    },
    {
        "name": "tunnelShortcut",
        "description": "Exact cost between two stops with and without teleport tunnels, "
        "the saving, and the tunnel pairs.",
        "parameters": {
            "type": "object",
            "properties": {"fromStop": {"type": "integer"}, "toStop": {"type": "integer"}},
            "required": ["fromStop", "toStop"],
        },
    },
]
