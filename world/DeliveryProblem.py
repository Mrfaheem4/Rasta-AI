from core.Instance import Cell
from world.World import World


class DeliveryProblem:
    """State = (cell, deliveredMask). Bits index instance.stops; entering any stop delivers it."""

    def __init__(self, world: World, start: Cell | None = None, deliveredMask: int = 0, stops=None):
        self.world = world
        self.start = world.instance.start if start is None else start
        self.deliveredMask = deliveredMask
        self.stopBit = {cell: 1 << i for i, cell in enumerate(world.instance.stops)}
        goals = world.instance.stops if stops is None else stops
        self.goalMask = sum(self.stopBit[cell] for cell in goals)

    def initialState(self):
        return (self.start, self.deliveredMask)

    def isGoal(self, state) -> bool:
        _, mask = state
        return mask & self.goalMask == self.goalMask

    def successors(self, state):
        cell, mask = state
        for action, nxt, cost in self.world.neighbors(cell):
            yield action, (nxt, mask | self.stopBit.get(nxt, 0)), cost

    def carried(self, state) -> int:
        """Undelivered parcels (one per instance stop)."""
        _, mask = state
        return len(self.stopBit) - bin(mask).count("1")
