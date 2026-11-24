import numpy as np

from core.Instance import Cell
from game.Evaluation import Evaluation, defaultWeights, manhattan
from world.Heuristics import StopDistances
from world.World import World, moves

defaultRadius = 4


def windowOf(world: World, center: Cell, radius: int) -> frozenset:
    """Free cells within Chebyshev distance radius of center (center is always included)."""
    r0, c0 = center
    cells = {
        (r, c)
        for r in range(max(r0 - radius, 0), min(r0 + radius, world.rows - 1) + 1)
        for c in range(max(c0 - radius, 0), min(c0 + radius, world.cols - 1) + 1)
        if np.isfinite(world.costMap[r, c])
    }
    cells.add(center)
    return frozenset(cells)


class RivalGame:
    """Reduced delivery game: MAX = delivery agent, MIN = blocker rival, both confined to a window.

    State = (agent, rival, deliveredMask, carriedMask). Deterministic on purpose: no slip, and no
    theft cooldown (the simulator uses 10), and no no-camping rule (the simulator forbids a rival
    from staying twice in a row; that rule is simulator-only). Entering the depot reloads
    undelivered parcels; entering a stop delivers it if carried; the rival steals the lowest
    carried parcel when it is within Manhattan distance 1 of the agent after any ply.
    Deliberately pessimistic: the rival moves on every ply (full speed), while the simulator's
    rival is half speed. Assuming the stronger rival keeps the agent's contest decisions safe.
    """

    def __init__(self, world: World, root, radius: int = defaultRadius, evaluation: Evaluation | None = None):
        self.world = world
        self.root = root
        self.window = windowOf(world, root[0], radius)
        if evaluation is None:
            evaluation = Evaluation(StopDistances(world), defaultWeights, world)
        self.evaluation = evaluation

    def initialState(self):
        return self.root

    def isTerminal(self, state) -> bool:
        return state[2] == self.world.fullMask

    def evaluate(self, state) -> float:
        return self.evaluation(state)

    def moves(self, state, maximizing: bool):
        agent, rival, _, _ = state
        if maximizing:
            options = [("stay", agent)]
            options += [
                (a, cell)
                for a, cell, _ in self.world.neighbors(agent)
                if cell in self.window and cell != rival
            ]
        else:
            options = [("stay", rival)]
            options += [(a, (rival[0] + dr, rival[1] + dc)) for a, dr, dc in moves]
            options = [(a, cell) for a, cell in options if cell in self.window and cell != agent]
        for action, cell in options:
            yield action, self.advance(state, maximizing, cell)

    def advance(self, state, maximizing: bool, cell):
        agent, rival, delivered, carried = state
        if maximizing:
            if cell != agent:
                delivered, carried = self.world.enter(cell, delivered, carried)
            agent = cell
        else:
            rival = cell
        if carried and manhattan(agent, rival) <= 1:
            carried &= carried - 1  # steal the lowest carried parcel
        return agent, rival, delivered, carried
