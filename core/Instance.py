from dataclasses import dataclass, replace

import numpy as np

from core.Config import cmsIds
from oel_instance import make_instance

Cell = tuple[int, int]


@dataclass(frozen=True, eq=False)
class Instance:
    instanceId: str
    grid: np.ndarray  # terrain index 0-3 (road, grass, mud, ford), -1 = blocked
    terrainCost: dict[str, int]
    start: Cell
    stops: tuple[Cell, ...]
    adversary: str
    slipProbability: float
    adversaryDepth: int
    twists: tuple[str, ...]
    slipPerParcel: float = 0.03  # third twist: load-dependent slip
    slipCap: float = 0.40

    @classmethod
    def load(cls, ids: list[str] = cmsIds, **overrides) -> "Instance":
        """Regenerate the instance from CMS IDs; overrides change any field (used for live viva edits)."""
        raw, grid = make_instance(ids)
        instance = cls(
            instanceId=raw["instance_id"],
            grid=grid,
            terrainCost=raw["terrain_cost"],
            start=raw["start"],
            stops=tuple(raw["delivery_stops"]),
            adversary=raw["adversary"],
            slipProbability=raw["slip_probability"],
            adversaryDepth=raw["adversary_search_depth"],
            twists=tuple(raw["twists"]),
        )
        return replace(instance, **overrides)

    def slipFor(self, carried: int) -> float:
        return min(self.slipProbability + self.slipPerParcel * carried, self.slipCap)
