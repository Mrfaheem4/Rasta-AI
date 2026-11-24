from world.Heuristics import StopDistances, tunnelMst
from world.World import World

rivalCap = 6  # rival distances beyond this all look the same to the evaluation
defaultWeights = {
    "progress": 1.0,
    "delivered": 10.0,
    "rivalDistance": 0.5,
    "theftExposure": -4.0,
    "blockage": -3.0,
}


def manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def popcount(mask: int) -> int:
    return bin(mask).count("1")


class Evaluation:
    """Score from MAX's view. Features, each a weighted term in __call__:
    progress: negated tunnelMst cost to finish (remaining work);
    delivered: parcels already delivered (what MAX is paid for);
    rivalDistance: Manhattan gap to the rival, capped (MAX wants room to move);
    theftExposure: carried parcels while the rival is adjacent (stealable);
    blockage: 1 if the rival stands on the agent's best next cell toward its nearest pending stop.
    """

    def __init__(self, distances: StopDistances, weights: dict, world: World):
        self.distances = distances
        self.weights = weights
        self.world = world
        self.progressHeuristic = tunnelMst(distances)

    def blockage(self, agent, rival, delivered) -> int:
        pending = [i for i in range(len(self.distances.stops)) if not (delivered >> i) & 1]
        if not pending:
            return 0
        nearest = min(pending, key=lambda i: self.distances.cellCost(agent, i))
        best = min(
            self.world.neighbors(agent),
            key=lambda step: step[2] + self.distances.cellCost(step[1], nearest),
            default=None,
        )
        return int(best is not None and best[1] == rival)

    def features(self, state) -> dict:
        agent, rival, delivered, carried = state
        gap = manhattan(agent, rival)
        return {
            "progress": -self.progressHeuristic((agent, delivered, carried)),
            "delivered": popcount(delivered),
            "rivalDistance": min(gap, rivalCap),
            "theftExposure": popcount(carried) * int(gap <= 1),
            "blockage": self.blockage(agent, rival, delivered),
        }

    def __call__(self, state) -> float:
        return sum(self.weights[name] * value for name, value in self.features(state).items())
