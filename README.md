# Rasta-AI

CS-370 Artificial Intelligence Open Ended Lab: an autonomous delivery agent under uncertainty and competition.

**Group:** Faheem Ahmed Khan (534662) | Abdullah Waleed Ahmed (509666) | Syed Ehtisham Haider Bukhari (511754) | M Rehan Haider (508160)

## Instance `9dc56812`
- 39 x 39 grid; terrain costs: road 3, grass 6, mud 7, ford 9
- Start (27, 14); 6 delivery stops
- Rival: blocker, search depth 6
- Slip probability 0.25
- Twists: rival steals parcels, teleport tunnels
- Third twist: **load-dependent slip**. Slip = 0.25 + 0.03 x parcels carried, capped at 0.40

## Setup
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python RunAll.py            # every experiment and figure (add --quick for a shorter run)
```
- Tables are printed with the group stamp and written as CSV to `outputs/`.
- Figures are written as PNG to `outputs/figures/`.
- The city map for the frontend is built by `world.CityMap.buildCityMap()` from `data/osm/raw.json` (not committed).

Run the API (FastAPI) for the frontend:
```
.venv\Scripts\python.exe -m uvicorn api.Server:app --port 8000
```

## Layout
| Folder | Contents |
|---|---|
| `core/` | Config, member stamp, instance loading |
| `world/` | Grid, cost map, twists, delivery state |
| `algo/` | General search and game algorithms (BFS, UCS, DLS, IDS, Greedy, A*, Minimax, Alpha-Beta) |
| `game/` | Rival agent, reduced game, evaluation function |
| `llm/` | LLM dispatcher, tools, validation, weight edits |
| `api/` | FastAPI backend |

See [CONTRIBUTING.md](CONTRIBUTING.md) for code conventions.
