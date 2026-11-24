Faheem Ahmed Khan (534662) | Abdullah Waleed Ahmed (509666) | Syed Ehtisham Haider Bukhari (511754) | M Rehan Haider (508160)

# Part A: Environment, agent design and world model (instance 9dc56812)

## Instance

- 39 x 39 grid. Terrain entry costs: road 3, grass 6, mud 7, ford 9. Blocked cells are impassable.
- Depot (start) at (27, 14), where all six parcels are loaded. Six delivery stops, one parcel each.
- Rival: blocker, search depth 6. It occupies cells and tries to block the agent.
- Base slip probability 0.25. Our load-dependent twist raises it by 0.03 per carried parcel, capped at 0.40. The agent starts with six parcels, so it slips at the cap (0.40) while carrying five or six parcels, that is until its second delivery (four carried gives 0.37).
- Teleport tunnels: three symmetric pairs of free cells, each move costs 2.
- Rival steals parcels: a stolen parcel returns to the depot and must be reloaded there.

## (a) PEAS

| | Description |
|---|---|
| Performance | Deliver all six parcels with low total terrain cost. Penalties: slips, stolen parcels, and extra replanning. Episode win: all parcels delivered within 3x the optimal step count; draw within 6x; loss otherwise. A rival theft can happen at most once every 10 steps. |
| Environment | 39 x 39 grid with four terrain types and blocked cells. Three teleport tunnel pairs (cost 2). Depot, six stops, and a rival agent that moves each turn and steals carried parcels when adjacent. Slip on each directional move, with probability 0.25 + 0.03 x carried, capped at 0.40. |
| Actuators | Four directional moves, stay, and teleport (only at a tunnel endpoint). Entering a stop delivers its parcel if carried. Entering the depot reloads every undelivered parcel. |
| Sensors | Full grid and terrain costs, own position and carried/delivered parcels, rival position. Slip and theft outcomes are reported after each move. |

## (b) Environment classification

- **Fully observable.** At each step the state (grid, both positions and the parcel masks) is fully observable. Stochastic outcomes such as slips and thefts are not hidden state: each one is revealed by the next percept.
- **Stochastic.** Each directional move can slip with a probability that depends on the cargo load, and the rival's thefts depend on adjacency, so the same action can have different outcomes.
- **Sequential.** Early decisions matter later: a stolen parcel forces a depot trip, and the depot resets all undelivered parcels, so the route depends on history.
- **Dynamic.** The rival moves every turn and can block corridors or steal parcels while the agent plans, so the world changes between plans.
- **Discrete.** Cells, the action set and the time steps are all discrete.
- **Multi-agent, competitive.** The blocker, which also steals, has goals opposed to the agent's, so it is an adversary, not a passive obstacle.
- **Known.** The rules, costs and tunnel pairs are published, so the agent knows the transition model up to the slip and rival randomness.

## (c) Agent type: utility-based

The goal is "all parcels delivered", but routes differ in total cost, slip risk and exposure to the rival, so the agent cannot just follow one route. It chooses between following the cheapest plan and contesting a threatening rival, and that choice is made by a weighted evaluation rather than by cost alone.

- **Goal-based alone is not enough.** A goal-based planner accepts any route that delivers everything, so it cannot price slip risk (which grows with each carried parcel), the chance of a theft near the rival, or the cost of a depot reload. It would take the cheapest path regardless of these risks.
- **Simple reflex fails.** A reflex agent maps the current percept to an action. It cannot remember that a parcel was stolen and must be reloaded, cannot look ahead through tunnels and terrain costs, and cannot recover from a slip except by reacting to the new cell.
- **Utility-based design.** The planner follows A* routes with the exact tunnel-aware heuristic `tunnelMst` over the delivery state (cell, delivered, carried). Each slip or theft triggers a replan from the new state, which covers the depot reload. The agent runs an alpha-beta contest, scored by a weighted evaluation of progress, deliveries, rival distance, theft exposure and blockage, only when it carries parcels and the rival is within Chebyshev distance 2. Otherwise it simply replans with A* around the rival's cell. A standoff breaker prevents endless contesting.

## (d) Cost-map benchmark

Building the per-cell entry-cost map was benchmarked vectorised (NumPy lookup) against a pure Python loop, 50 repeats each, mean seconds per build.

| Grid | Vectorised (s) | Loop (s) | Speed-up |
|---|---|---|---|
| Instance 39 x 39 | 1.20e-5 | 2.24e-4 | 18.6x |
| Random 400 x 400 | 2.41e-3 | 3.84e-2 | 15.9x |

The speed-up is 16 to 20 times on both sizes in this run. Earlier runs gave 11.9x to 30x, so treat the exact ratio as approximate (timings vary between runs). The vectorised map is used for every world, so the search never builds costs cell by cell.

Source: `experiments/WorldExperiments.py`, output `outputs/costmap_benchmark.csv`.
