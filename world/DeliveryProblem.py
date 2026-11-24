from core.Instance import Cell
from world.World import World


class DeliveryProblem:
    """State = (cell, deliveredMask, carriedMask). Delivery needs the parcel carried; depot reloads all."""

    def __init__(
        self,
        world: World,
        start: Cell | None = None,
        deliveredMask: int = 0,
        stops=None,
        carriedMask: int | None = None,
    ):
        self.world = world
        self.start = world.instance.start if start is None else start
        self.deliveredMask = deliveredMask
        self.carriedMask = world.fullMask & ~deliveredMask if carriedMask is None else carriedMask
        goals = world.instance.stops if stops is None else stops
        self.goalMask = sum(world.stopBit[cell] for cell in goals)

    def initialState(self):
        return (self.start, self.deliveredMask, self.carriedMask)

    def isGoal(self, state) -> bool:
        delivered = state[1]
        return delivered & self.goalMask == self.goalMask

    def successors(self, state):
        cell, delivered, carried = state
        for action, nxt, cost in self.world.neighbors(cell):
            yield action, (nxt, *self.world.enter(nxt, delivered, carried)), cost

    def carried(self, state) -> int:
        """Parcels currently carried."""
        return bin(state[2]).count("1")
