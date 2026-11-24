import numpy as np

from game.Episode import drawMultiplier, winMultiplier
from game.Policies import BlockerRival, DeliveryAgent, PassiveRival
from world.Simulator import Simulator, rivalStartFor
from world.World import moves

deltas = {name: (dr, dc) for name, dr, dc in moves}


def stepCell(world, cell, action):
    if action == "stay":
        return cell
    if action == "teleport":
        return world.tunnels.get(cell)
    if action in deltas:
        dr, dc = deltas[action]
        return (cell[0] + dr, cell[1] + dc)
    return None


class SimSession:
    """One live episode, stepped turn by turn in the same order as game.Episode.runEpisode.
    plannedPath is sent only when the new plan is not a suffix of the last sent plan. An absent
    plannedPath means the last sent plan minus the cells the agent has already passed (the frontend
    trims its leading cells). The first frame always carries it."""

    def __init__(
        self, world, distances, plannedSteps: int, seed: int = 0, rival: str = "blocker", maxSteps=None
    ):
        self.world = world
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sim = Simulator(world, self.rng, rivalStartFor(world, self.rng))
        self.agent = DeliveryAgent(world, distances)
        self.rival = BlockerRival(world, distances) if rival == "blocker" else PassiveRival()
        self.rivalName = rival
        self.maxSteps = maxSteps or drawMultiplier * plannedSteps
        self.lastPath = None

    def finished(self) -> bool:
        return self.sim.done() or self.sim.state.step >= self.maxSteps

    def outcome(self):
        if not self.sim.done():
            return "loss" if self.finished() else None
        return "win" if self.sim.state.step <= self.maxSteps * winMultiplier // drawMultiplier else "draw"

    def plannedPath(self, stale: bool) -> list:
        """Remaining plan as cells. After a slip or theft the stored route is stale (the agent replans on
        its next turn), so a fresh plan from the current state is returned instead."""
        if stale:
            return [list(s[0]) for s in self.agent.planFrom(self.sim.state).path]
        cell, path = self.sim.state.agent, [list(self.sim.state.agent)]
        for action in self.agent.route or []:
            cell = stepCell(self.world, cell, action)
            if cell is None:
                break
            path.append(list(cell))
        return path

    def frame(self, action, rivalAction, slipped: bool, theft: bool, mode: str) -> dict:
        s = self.sim.state
        frame = {
            "step": s.step,
            "agent": list(s.agent),
            "rival": list(s.rival),
            "deliveredMask": s.deliveredMask,
            "carriedMask": s.carriedMask,
            "cost": s.cost,
            "slipped": slipped,
            "theft": theft,
            "action": action,
            "rivalAction": rivalAction,
            "mode": mode,
            "done": self.finished(),
        }
        path = self.plannedPath(slipped or theft)
        last = self.lastPath
        isSuffix = last is not None and len(path) <= len(last) and last[len(last) - len(path) :] == path
        if not isSuffix:
            frame["plannedPath"] = path
            self.lastPath = path
        if frame["done"]:
            frame["outcome"] = self.outcome()
        return frame

    def initialFrame(self) -> dict:
        return self.frame(None, None, False, False, "plan")

    def step(self) -> dict:
        """One agent turn, then one rival turn (none if the agent has already delivered everything)."""
        agent, sim = self.agent, self.sim
        contests, sidesteps = agent.contestTurns, agent.sidesteps
        action = agent(sim)
        mode = (
            "contest"
            if agent.contestTurns > contests
            else "sidestep"
            if agent.sidesteps > sidesteps
            else "plan"
        )
        slipped = sim.agentStep(action)
        rivalAction, theft = None, False
        if not sim.done():
            rivalAction = self.rival(sim)
            theft = sim.rivalStep(rivalAction)
            agent.observe(slipped, theft)
        return self.frame(action, rivalAction, slipped, theft, mode)
