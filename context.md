# Rasta-AI: Project Context

Living summary of the project. Every PR updates this file.

**Group:** Faheem Ahmed Khan (534662) | Abdullah Waleed Ahmed (509666) | Syed Ehtisham Haider Bukhari (511754) | M Rehan Haider (508160)

## 1. What we are building
CS-370 Open Ended Lab: an autonomous delivery agent on a grid city. It has to cope with uncertain movement (slip) and with a competing rival. The parts are:
- from-scratch search (BFS, UCS, DLS, IDS, Greedy, A*)
- game search (Minimax, Alpha-Beta)
- an LLM dispatcher using a local Qwen3.5-0.8B, edited with weight-space programming
- a FastAPI backend and a React + three.js frontend on a real Islamabad map

## 2. Our instance `9dc56812` (regenerated from the CMS IDs)
| Item | Value |
|---|---|
| Grid | 39 x 39, obstacle density 0.169 |
| Terrain entry cost | road 3, grass 6, mud 7, ford 9 |
| Depot (start) | (27, 14) |
| Stops 1..6 | (20,18), (6,6), (5,18), (32,34), (13,32), (4,33) |
| Rival | blocker, search depth 6 |
| Slip | 0.25 base |
| Assigned twists | rival steals parcels, teleport tunnels |
| Our third twist | load-dependent slip |

### Twist rules as implemented
- **Teleport tunnels:** 3 symmetric pairs, seeded from the instance id: (0,0)-(16,15), (28,11)-(29,38), (34,6)-(38,4). A teleport costs 2.
- **Rival steals parcels:** when the rival is adjacent (Manhattan distance 1 or less), it steals the lowest carried parcel. Stealing has a 10-step cooldown. A stolen parcel goes back to the depot, and entering the depot reloads every undelivered parcel.
- **Load-dependent slip (ours):** p(slip) = min(0.25 + 0.03 x carried, 0.40). A slipped move goes to a random other legal 4-way move. Teleport and stay never slip.
- **No camping (simulator only):** the rival may not stay two turns in a row.

## 3. Search state
s = (x, D, C), where:
- x is the cell
- D is the bitmask of delivered stops
- C is the bitmask of carried parcels

A parcel is delivered only if it is carried. The goal is reached when D covers all stops. About 81k states are reachable.

## 4. Our heuristic: `tunnelMst`

### 4.1 Definition
Notation:
- S is the set of stops and P = S \ D is the set of pending stops.
- `dep` is the depot.
- d(u, v) is the **exact** cheapest cost from cell u to cell v in the real move graph: 4-way moves that cost the entered cell's terrain, plus teleport edges that cost 2.

Each d(., n) is precomputed for every target n (6 stops and the depot) by one exhaustive uniform-cost search over the **reversed** graph (`ReverseGrid` + `costField`). That is 7 searches over 1,521 cells, done once per instance.

d is asymmetric, because a move costs the entered cell, so for pairs of targets we take

  w(i, j) = min( d(n_i, n_j), d(n_j, n_i) )

The nodes that must still be visited are

  N(s) = P ∪ {dep}   if some pending parcel is not carried (P \ C ≠ ∅)
  N(s) = P           otherwise

and the heuristic is

  **h(s) = min over n ∈ N(s) of d(x, n)  +  MST_w( N(s) )**,   with h(goal) = 0,

where MST_w is the weight of a minimum spanning tree over N(s) under w. It is computed with Prim's algorithm. There are at most 2^7 distinct node sets, so each one is cached once.

### 4.2 Why it is admissible (h ≤ h*)
1. Any feasible completion from s must visit every node in N(s). Each pending stop must be entered to deliver it. If a pending parcel is not carried, the depot must be entered before that stop.
2. Let n_1, ..., n_k be the order in which the optimal completion first visits those nodes. Its cost is at least d(x, n_1) + Σ d(n_i, n_{i+1}), because each leg is a walk between those cells and d is the exact minimum. All edge costs are non-negative.
3. d(x, n_1) ≥ min_n d(x, n), and each d(n_i, n_{i+1}) ≥ w(n_i, n_{i+1}).
4. The legs n_1 → n_2 → ... → n_k form a Hamiltonian path over N(s). A Hamiltonian path is a spanning tree, so its weight is at least MST_w(N(s)).
5. Combining these, h*(s) ≥ min_n d(x, n) + MST_w(N(s)) = h(s). ∎

The two terms are what make it tight:
- the exact tunnel-aware distance, instead of Manhattan
- the MST lower bound on the remaining tour, instead of "nearest stop only"

### 4.3 Consistency (h(s) ≤ c(s, s') + h(s'))
We checked this **empirically on every reachable state and every edge**: 81,472 states and 267,776 edges, with **0 violations**. We do not claim a general proof.

### 4.4 Results (A*, full 6-stop problem, optimal cost 463)
| Heuristic | Nodes expanded | Admissible? | Consistency violations |
|---|---|---|---|
| UCS (h = 0) | 65,664 | yes | 0 |
| Manhattan x min cost, nearest stop (baseline 1) | 50,147 | **no**: 417 / 7,656 exact checks overestimate | 188 |
| Manhattan x min cost, farthest stop (baseline 2) | 29,574 | **no**: 417 / 7,656 | 189 |
| exactNearest (ablation: no MST term) | 42,326 | yes | 0 |
| **tunnelMst (ours)** | **210** | **yes**: 0 / 7,656 | **0** |

- **Why the textbook baselines break here:** the tunnels make them inadmissible. At endpoint (0,0), Manhattan estimates h = 114 to reach stop (20,18), but the true cost h* is 37, because a cost-2 teleport skips most of the distance.
- **Ablation:** removing the MST term costs 200 times more node expansions, so most of the strength comes from the MST.
- **Inadmissible variant:** weighted A* (w ≥ 1.5) returns 474 instead of 463, a 2.4% worse plan, with about 40% fewer expansions.

### 4.5 How niche is it?
- **Known ingredients:**
  - The MST lower bound on a tour is the classic TSP heuristic, used for example in Russell & Norvig's TSP exercises.
  - Precomputed exact distance tables are the idea behind "true-distance" heuristics and landmark heuristics.
- **Our own parts (not textbook):**
  - Exact distances on a graph with **teleport edges and asymmetric entry costs**, obtained by reverse UCS and symmetrised with min so the MST bound stays valid.
  - A **state-dependent node set**: the depot joins the tree only when theft has left a pending parcel uncarried. This ties the heuristic directly to the "rival steals parcels" twist.
  - Per-node-set caching that makes it effectively O(|P|) per state after the 7 precomputed searches.
- **Verdict:** the building blocks are standard, but this combination is specific to our twisted state space. The usual assignment answer, Manhattan scaled by minimum cost, is provably inadmissible on our map.

## 5. Game (Part C)
- **Reduced game:** a radius-4 window around the agent. MAX is the agent and MIN is the blocker. The game is deterministic by design: no slip and no theft cooldown.
- **Evaluation:** a weighted sum of 5 features:
  - progress: -tunnelMst
  - parcels delivered
  - rival distance, capped at 6
  - theft exposure: carried parcels x (rival adjacent)
  - blockage: the rival is on the agent's best next cell
- **Move ordering:** by 1-ply evaluation. Average over 10 positions:

| Depth | Minimax nodes | Alpha-Beta nodes | Alpha-Beta + ordering nodes |
|---|---|---|---|
| 6 | 7,471 | 1,269 (79% saved) | 334 (93.5% saved) |

- **Agent policy:**
  - It follows the A* (tunnelMst) plan, replanning after every slip or theft.
  - It switches to Alpha-Beta at depth 4 only when it carries parcels and the rival is within Chebyshev distance 2.
  - Its plans route around the rival's cell.
  - A standoff breaker makes it leave Alpha-Beta mode after 6 turns with no progress, and it sidesteps instead of waiting.
- **Outcome:**
  - win: everything delivered within 3x the optimal step count
  - draw: within 6x
  - loss: otherwise
- **Latest finished evaluation run** (the sidestep and no-camping fixes are still re-running):

| Rival | Wins | Draws | Losses | Mean cost |
|---|---|---|---|---|
| Blocker | 1/30 | 0/30 | 29/30 | 2,985 |
| Passive | 23/30 | 6/30 | 1/30 | 1,340 |

- **What the deterministic assumption gets wrong:** observed slip was 0.355 against 0.350 expected. The planned cost was 463, but the real mean was much higher.

## 6. LLM dispatcher (Part D)
- **Tools:**
  - `planRoute`, with stop order, terrain avoidance and a cost budget
  - `simulate`
  - `tunnelShortcut` (custom, tied to the tunnel twist)
- **Validation layer:** rejects unknown tools, invented arguments, stop numbers outside 1..6, blocked or off-grid cells, and bad types.
- **Requests:** 20 of increasing difficulty, including traps (stop 7, an impossible budget, an infeasible avoid). The direct planner baseline succeeds on 20/20.
- **Model:** Qwen3.5-0.8B (still downloading). Weight-space programming follows Takehi et al.: no gradients, and an edit of the form W' = W - α·u(uᵀW) on the MLP output columns along a faulty direction.

## 7. Architecture
| Folder | Contents |
|---|---|
| `core/` | Config (members, stamp, city bounds), Instance (CMS-ID instance plus live overrides) |
| `algo/` | General SearchProblem and GameProblem; bfs, ucs, dls, ids, greedy, aStar, costField, minimax, alphaBeta |
| `world/` | CostMap, Tunnels, World (enter rule), DeliveryProblem, Heuristics, Simulator, Recorder, CityMap, CityGrid |
| `game/` | RivalGame, Evaluation, MoveOrdering, Policies, Episode |
| `llm/` | Tools, Validation, Requests, Dispatcher |
| `api/` | FastAPI: /instance, /plan (with expansion trace), /params, /mode |
| `experiments/` | Route, Game, World experiments and Plots; every output is stamped |
| `docs/` | PartA.md |
| `frontend/` | React + three.js (in progress) |

## 8. Conventions
- PascalCase for classes and file names; camelCase for variables, methods and JSON fields.
- Every algorithm lives in `algo/` and is reused from there.
- Small commits, one PR per feature, no AI co-author lines, no `claude/` branch prefixes.
- Every output carries the stamp line.

## 9. Status
| Phase | State |
|---|---|
| 0 Setup | done |
| 1 World model (A) | done |
| 2 Search algorithms | done |
| 3 Heuristics and route experiments (B) | done |
| 4 Rival, simulator, game, games (C) | games re-running with fixes |
| 5 API | done, plus search trace and city mode |
| 6 LLM tools (D) | done; model download about 50% |
| 7 Frontend (Islamabad F-6/F-7, Cipher-style, search animation) | waiting on OSM data and npm |
| 8 Report assets | Part A and figures done; RunAll.py pending |
