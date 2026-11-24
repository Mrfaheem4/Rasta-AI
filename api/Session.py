import dataclasses

from core.Config import cityBounds, members, stampLine
from core.Instance import Instance
from world import World
from world.CityGrid import cityGrid
from world.CityMap import rawOsmPath
from world.Heuristics import StopDistances

fieldNames = {f.name for f in dataclasses.fields(Instance)}
readOnly = {"instanceId", "grid"}
demoNote = "real-city demo: numbers differ from the CMS-ID instance"


class Session:
    """Current instance with its World and cached StopDistances. update() is the live viva edit path."""

    def __init__(self):
        self.overrides = {}
        self.distanceKey = None
        self.mode = "instance"
        self.install(Instance.load())

    def buildInstance(self, overrides: dict) -> Instance:
        base = Instance.load(**overrides)
        if self.mode == "instance":
            return base
        grid, start, stops = cityGrid(rawOsmPath, base.start, base.stops)
        return dataclasses.replace(base, grid=grid, start=start, stops=stops)

    def setMode(self, mode: str) -> None:
        if mode not in ("instance", "city"):
            raise ValueError(f"mode must be 'instance' or 'city', got {mode!r}")
        previous, self.mode = self.mode, mode
        try:
            self.install(self.buildInstance(self.overrides))
        except ValueError:
            self.mode = previous
            raise

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
        instance = self.buildInstance(candidate)
        checkCells(instance)
        self.overrides = candidate
        self.install(instance)

    def reset(self) -> None:
        self.overrides = {}
        self.install(self.buildInstance({}))

    def install(self, instance: Instance) -> None:
        self.instance = instance
        self.world = World(instance)
        # distances depend only on terrain, grid, start and stops; slip edits reuse them
        key = (tuple(instance.terrainCost.items()), instance.grid.tobytes(), instance.start, instance.stops)
        if key != self.distanceKey:
            self.distances = StopDistances(self.world)
            self.distanceKey = key

    def payload(self) -> dict:
        inst = self.instance
        extra = {"demoNote": demoNote} if self.mode == "city" else {}
        return {
            "stamp": stampLine,
            "mode": self.mode,
            **extra,
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
