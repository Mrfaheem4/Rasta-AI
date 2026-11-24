import dataclasses

from core.Config import cityBounds, members, stampLine
from core.Instance import Instance
from world import World
from world.Heuristics import StopDistances

fieldNames = {f.name for f in dataclasses.fields(Instance)}
readOnly = {"instanceId", "grid"}


class Session:
    """Current instance with its World and cached StopDistances. update() is the live viva edit path."""

    def __init__(self):
        self.overrides = {}
        self.distanceKey = None
        self.install(Instance.load())

    def update(self, overrides: dict) -> None:
        unknown = sorted(set(overrides) - fieldNames)
        if unknown:
            raise ValueError(f"unknown override keys: {unknown}")
        locked = sorted(set(overrides) & readOnly)
        if locked:
            raise ValueError(f"read-only override keys: {locked}")
        normalized = {**overrides}
        if "start" in normalized:
            normalized["start"] = tuple(normalized["start"])
        if "stops" in normalized:
            normalized["stops"] = tuple(tuple(cell) for cell in normalized["stops"])
        candidate = {**self.overrides, **normalized}
        instance = Instance.load(**candidate)
        checkCells(instance)
        self.overrides = candidate
        self.install(instance)

    def reset(self) -> None:
        self.overrides = {}
        self.install(Instance.load())

    def install(self, instance: Instance) -> None:
        self.instance = instance
        self.world = World(instance)
        # distances depend only on terrain, start and stops; slip edits reuse them
        key = (tuple(instance.terrainCost.items()), instance.start, instance.stops)
        if key != self.distanceKey:
            self.distances = StopDistances(self.world)
            self.distanceKey = key

    def payload(self) -> dict:
        inst = self.instance
        return {
            "stamp": stampLine,
            "members": [{"name": name, "cmsId": cms} for name, cms in members],
            "instanceId": inst.instanceId,
            "rows": self.world.rows,
            "cols": self.world.cols,
            "grid": inst.grid.tolist(),
            "terrainCost": inst.terrainCost,
            "start": list(inst.start),
            "stops": [list(cell) for cell in inst.stops],
            "tunnels": [[list(a), list(b)] for a, b in self.world.tunnels.items() if a < b],
            "adversary": inst.adversary,
            "adversaryDepth": inst.adversaryDepth,
            "slipProbability": inst.slipProbability,
            "slipPerParcel": inst.slipPerParcel,
            "slipCap": inst.slipCap,
            "twists": [*inst.twists, "load_dependent_slip"],
            "geo": {
                **cityBounds,
                "convention": "row 0 = north edge, col 0 = west edge, cells are equal lat/lon steps",
            },
        }


def checkCells(instance: Instance) -> None:
    rows, cols = instance.grid.shape
    for r, c in [instance.start, *instance.stops]:
        if not (0 <= r < rows and 0 <= c < cols) or instance.grid[r, c] < 0:
            raise ValueError(f"cell {(r, c)} is outside the grid or blocked")
