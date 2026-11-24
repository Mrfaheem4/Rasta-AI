from algo.AlphaBeta import alphaBeta
from algo.BestFirst import bestFirst
from game.Evaluation import Evaluation, defaultWeights, manhattan
from game.MoveOrdering import orderMoves
from game.RivalGame import RivalGame
from world.DeliveryProblem import DeliveryProblem
from world.Heuristics import StopDistances, tunnelMst
from world.World import World

contestRadius = 2  # contest only when a parcel is carried and the rival is this close (theft threat)
standoffWindow = 6  # contest turns in a row without a drop in tunnelMst h count as a standoff
escapeTurns = 4  # turns of rival-avoiding plan after a standoff


def chebyshev(a, b) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def memoized(h):
    """Cache a heuristic per state: replans revisit the same states."""
    cache = {}

    def cached(state) -> float:
        if state not in cache:
            cache[state] = h(state)
        return cache[state]

    return cached


class AvoidCell:
    """Wraps a delivery problem so the given cell cannot be entered."""

    def __init__(self, problem, cell):
        self.problem = problem
        self.cell = cell

    def initialState(self):
        return self.problem.initialState()

    def isGoal(self, state) -> bool:
        return self.problem.isGoal(state)

    def successors(self, state):
        return (step for step in self.problem.successors(state) if step[1][0] != self.cell)


class DeliveryAgent:
    """Follows an A*(tunnelMst) plan. Replans after slips, thefts, or when the rival blocks or is
    adjacent to the next cell (blockReplans); plans avoid the rival's cell. Contests a threatening
    rival with alphaBeta; a standoff (no progress for standoffWindow contest turns) is broken by
    following the rival-avoiding plan for escapeTurns."""

    def __init__(self, world: World, distances: StopDistances, weights=defaultWeights, depth=4, radius=4):
        self.world = world
        self.distances = distances
        self.heuristic = memoized(tunnelMst(distances))
        self.evaluation = Evaluation(distances, weights, world)
        self.depth = depth
        self.radius = radius
        self.route = None
        self.plannedCost = None
        self.inContest = False
        self.progress = []
        self.escapeLeft = 0
        self.lastSlipped = False
        self.lastTheft = False
        self.replans = 0
        self.slipReplans = 0
        self.blockReplans = 0
        self.sidesteps = 0
        self.contestTurns = 0

    def observe(self, slipped: bool, theft: bool):
        self.lastSlipped, self.lastTheft = slipped, theft

    def planFrom(self, s, avoid=False):
        """Plan from s. With avoid, route around the rival's cell if that is possible at all; a cut-off
        rival makes the avoiding search exhaustive, so it falls back to the plain plan."""
        problem = DeliveryProblem(self.world, s.agent, s.deliveredMask, carriedMask=s.carriedMask)
        if avoid:
            pending = [c for c, bit in self.world.stopBit.items() if not s.deliveredMask & bit]
            seen = self.reachable(s.agent, s.rival)
            if all(c in seen for c in (*pending, self.world.instance.start)):
                result = self.astar(AvoidCell(problem, s.rival))
                if result.found:
                    return result
        return self.astar(problem)

    def astar(self, problem):
        # A* without the @measured wrapper: planning inside an episode is not benchmarked
        return bestFirst(problem, self.heuristic, lambda g, h: g + h)

    def reachable(self, source, blockedCell) -> set:
        seen, frontier = {source}, [source]
        while frontier:
            for _, nxt, _ in self.world.neighbors(frontier.pop()):
                if nxt != blockedCell and nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return seen

    def nextCell(self, s):
        if not self.route:
            return None
        return {a: c for a, c, _ in self.world.neighbors(s.agent)}.get(self.route[0])

    def contest(self, s) -> str:
        self.contestTurns += 1
        self.inContest = True
        self.route = None
        root = (s.agent, s.rival, s.deliveredMask, s.carriedMask)
        game = RivalGame(self.world, root, self.radius, self.evaluation)
        move = alphaBeta(game, root, self.depth, order=orderMoves(game)).move
        self.progress.append(self.heuristic((s.agent, s.deliveredMask, s.carriedMask)))
        if len(self.progress) >= standoffWindow and self.progress[-1] >= self.progress[-standoffWindow]:
            self.escapeLeft = escapeTurns
            self.progress = []
        return move

    def __call__(self, sim) -> str:
        s = sim.state
        slipped, theft = self.lastSlipped, self.lastTheft
        self.lastSlipped = self.lastTheft = False
        if self.plannedCost is None:
            first = self.planFrom(s)
            self.plannedCost = first.cost
            self.route = list(first.actions)
        threatened = s.carriedMask != 0 and chebyshev(s.agent, s.rival) <= contestRadius
        if threatened and self.escapeLeft == 0:
            return self.contest(s)
        if self.escapeLeft > 0:
            self.escapeLeft -= 1
        if self.inContest:
            self.inContest = False
            self.progress = []
            self.route = None
        nxt = self.nextCell(s)
        blocked = nxt is not None and manhattan(nxt, s.rival) <= 1
        if self.route is None or not self.route or slipped or theft or blocked:
            self.replans += 1
            self.slipReplans += slipped
            self.blockReplans += blocked
            self.route = list(self.planFrom(s, avoid=blocked or self.escapeLeft > 0).actions)
        if not self.route or self.route[0] not in {m[0] for m in sim.legalMoves(s.agent, s.rival)}:
            return self.sidestep(sim, s)
        return self.route.pop(0)

    def sidestep(self, sim, s) -> str:
        """The next step is blocked: step to the legal neighbour closest to the nearest useful target,
        then replan next turn. Stay only when no neighbour is legal."""
        self.sidesteps += 1
        self.route = []
        options = [m for m in sim.legalMoves(s.agent, s.rival) if m[0] != "stay"]
        if not options:
            return "stay"
        target = self.target(s)
        return min(options, key=lambda m: m[2] + self.distances.cellCost(m[1], target))[0]

    def target(self, s) -> int:
        """Nearest pending stop; the depot if that parcel is not carried (it must be reloaded first)."""
        pending = [i for i in range(len(self.distances.stops)) if not (s.deliveredMask >> i) & 1]
        nearest = min(pending, key=lambda i: self.distances.cellCost(s.agent, i))
        return nearest if (s.carriedMask >> nearest) & 1 else self.distances.depotIndex


class BlockerRival:
    """Minimising alphaBeta in the reduced game when the agent is within the window; otherwise
    greedily walks toward the agent's nearest pending stop (4-way or stay, no teleport)."""

    def __init__(self, world: World, distances: StopDistances, depth=None, radius=4, weights=defaultWeights):
        self.world = world
        self.distances = distances
        self.depth = world.instance.adversaryDepth if depth is None else depth
        self.radius = radius
        self.evaluation = Evaluation(distances, weights, world)

    def __call__(self, sim) -> str:
        s = sim.state
        if chebyshev(s.agent, s.rival) <= self.radius:
            root = (s.agent, s.rival, s.deliveredMask, s.carriedMask)
            game = RivalGame(self.world, root, self.radius, self.evaluation)
            move = alphaBeta(game, root, self.depth, maximizing=False, order=orderMoves(game)).move
            if move == "stay" and sim.rivalStayed:  # no camping: take the best non-stay child instead
                moves = [(m, nxt) for m, nxt in game.moves(root, False) if m != "stay"]
                if moves:
                    move = min(moves, key=lambda mv: game.evaluate(mv[1]))[0]
            return move
        pending = [i for i in range(len(self.distances.stops)) if not (s.deliveredMask >> i) & 1]
        if not pending:
            return "stay"
        target = min(pending, key=lambda i: self.distances.cellCost(s.agent, i))
        options = [m for m in sim.legalMoves(s.rival, s.agent) if m[0] != "teleport"]
        if sim.rivalStayed and len(options) > 1:
            options = [m for m in options if m[0] != "stay"]
        return min(options, key=lambda m: self.distances.cellCost(m[1], target))[0]


class PassiveRival:
    """Baseline: stays put, except that the simulator's no-camping rule forces a step after a stay
    (a small confound: this baseline is not a pure no-adversary game)."""

    def __call__(self, sim) -> str:
        if not sim.rivalStayed:
            return "stay"
        s = sim.state
        return next(
            (m[0] for m in sim.legalMoves(s.rival, s.agent) if m[0] not in ("stay", "teleport")), "stay"
        )
