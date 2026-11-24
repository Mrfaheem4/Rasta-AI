from dataclasses import dataclass

from algo.CostField import costField
from core.Instance import Cell
from world.DeliveryProblem import DeliveryProblem
from world.World import World, moves

directions = tuple(action for action, _, _ in moves)


@dataclass
class SimState:
    agent: Cell
    rival: Cell
    deliveredMask: int
    carriedMask: int
    step: int
    cost: float
    slips: int
    thefts: int
    theftCooldown: int


def pickMove(options, action):
    for option in options:
        if option[0] == action:
            return option
    raise ValueError(f"illegal action {action!r}")


class Simulator:
    """Stochastic world: agent slips on 4-way moves, rival steals carried parcels when adjacent."""

    def __init__(self, world: World, rng, rivalStart: Cell, theftCooldown: int = 3):
        self.world = world
        self.rng = rng
        self.cooldownLength = theftCooldown
        self.state = SimState(world.instance.start, rivalStart, 0, world.fullMask, 0, 0.0, 0, 0, 0)

    def legalMoves(self, cell, other):
        options = [m for m in self.world.neighbors(cell) if m[1] != other]
        options.append(("stay", cell, float(self.world.costMap[cell])))
        return options

    def agentStep(self, action) -> bool:
        """Apply the agent's action, with slip. Returns True if the move slipped."""
        s = self.state
        legal = self.legalMoves(s.agent, s.rival)
        chosen = pickMove(legal, action)
        carried = bin(s.carriedMask).count("1")
        slipped = action in directions and self.rng.random() < self.world.instance.slipFor(carried)
        if slipped:
            others = [m for m in legal if m[0] in directions and m[0] != action]
            chosen = others[int(self.rng.integers(len(others)))] if others else pickMove(legal, "stay")
        _, s.agent, cost = chosen
        s.deliveredMask, s.carriedMask = self.world.enter(s.agent, s.deliveredMask, s.carriedMask)
        s.cost += cost
        s.step += 1
        s.slips += slipped
        return slipped

    def rivalStep(self, action) -> bool:
        """Move the rival (no slip), then apply theft. Returns True if a parcel was stolen."""
        s = self.state
        if s.theftCooldown > 0:
            s.theftCooldown -= 1
        walks = [m for m in self.legalMoves(s.rival, s.agent) if m[0] != "teleport"]  # rival can't teleport
        _, s.rival, _ = pickMove(walks, action)
        adjacent = abs(s.agent[0] - s.rival[0]) + abs(s.agent[1] - s.rival[1]) <= 1
        if adjacent and s.theftCooldown == 0 and s.carriedMask:
            s.carriedMask ^= s.carriedMask & -s.carriedMask
            s.thefts += 1
            s.theftCooldown = self.cooldownLength
            return True
        return False

    def done(self) -> bool:
        return self.state.deliveredMask == self.world.fullMask


def rivalStartFor(world: World, rng) -> Cell:
    """Random cell from the costlier half of cells reachable from the depot (exact forward cost)."""
    best = {}
    for state, g in costField(DeliveryProblem(world)).items():
        cell = state[0]
        if g < best.get(cell, float("inf")):
            best[cell] = g
    ranked = sorted(best, key=best.get)
    top = ranked[len(ranked) // 2 :]
    return top[int(rng.integers(len(top)))]
