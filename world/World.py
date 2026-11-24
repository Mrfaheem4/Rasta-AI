import numpy as np

from core.Instance import Cell, Instance
from world.CostMap import buildCostMap
from world.Tunnels import makeTunnels, tunnelCost

moves = (("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1))


class World:
    def __init__(self, instance: Instance):
        self.instance = instance
        self.costMap = buildCostMap(instance.grid, instance.terrainCost)
        self.tunnels = makeTunnels(instance)
        self.rows, self.cols = self.costMap.shape
        self.stopBit = {cell: 1 << i for i, cell in enumerate(instance.stops)}
        self.fullMask = (1 << len(instance.stops)) - 1

    def neighbors(self, cell: Cell) -> list[tuple[str, Cell, float]]:
        r, c = cell
        out = []
        for action, dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and np.isfinite(self.costMap[nr, nc]):
                out.append((action, (nr, nc), float(self.costMap[nr, nc])))
        if cell in self.tunnels:
            out.append(("teleport", self.tunnels[cell], tunnelCost))
        return out

    def enter(self, cell: Cell, delivered: int, carried: int) -> tuple[int, int]:
        """Entering a stop delivers it if carried; entering the depot reloads every undelivered parcel."""
        bit = self.stopBit.get(cell, 0)
        if bit and carried & bit:
            delivered |= bit
            carried &= ~bit
        if cell == self.instance.start:
            carried = self.fullMask & ~delivered
        return delivered, carried
