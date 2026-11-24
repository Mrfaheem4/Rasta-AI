from game.Evaluation import popcount
from world.Simulator import Simulator, directions

# Expected slip inflation is about 1/(1 - 0.4) at full load, plus replanning, so the budgets are
# generous multiples of the deterministic plan length.
winMultiplier = 3
drawMultiplier = 6


def runEpisode(world, rng, agent, rival, rivalStart, maxSteps: int) -> dict:
    """Alternate agent and rival steps until delivery or maxSteps agent steps. Callers pass
    maxSteps = drawMultiplier * plannedSteps. win: delivered within winMultiplier * plannedSteps;
    draw: delivered by maxSteps; loss: otherwise."""
    sim = Simulator(world, rng, rivalStart)
    expectedSlips = 0.0
    directionalSteps = 0
    agentStays = rivalStays = 0
    while not sim.done() and sim.state.step < maxSteps:
        action = agent(sim)
        agentStays += action == "stay"
        if action in directions:
            directionalSteps += 1
            expectedSlips += world.instance.slipFor(popcount(sim.state.carriedMask))
        slipped = sim.agentStep(action)
        if sim.done():
            break
        rivalAction = rival(sim)
        stolen = sim.rivalStep(rivalAction)
        rivalStays += sim.rivalMoved and rivalAction == "stay"
        agent.observe(slipped, stolen)
    s = sim.state
    if not sim.done():
        outcome = "loss"
    elif s.step <= maxSteps * winMultiplier // drawMultiplier:
        outcome = "win"
    else:
        outcome = "draw"
    return {
        "outcome": outcome,
        "cost": s.cost,
        "steps": s.step,
        "slips": s.slips,
        "thefts": s.thefts,
        "replans": agent.replans,
        "slipReplans": agent.slipReplans,
        "blockReplans": agent.blockReplans,
        "sidesteps": agent.sidesteps,
        "contestTurns": agent.contestTurns,
        "agentStays": agentStays,
        "rivalStays": rivalStays,
        "plannedCost": agent.plannedCost,
        "extraCost": s.cost - agent.plannedCost,
        "delivered": popcount(s.deliveredMask),
        "expectedSlips": expectedSlips,
        "directionalSteps": directionalSteps,
    }
