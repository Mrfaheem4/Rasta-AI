import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from algo import aStar, bfs, dls, greedy, ids, ucs
from api.Session import Session
from core.Config import stampLine
from world import DeliveryProblem
from world.Heuristics import exactNearest, manhattanFarthest, manhattanNearest, tunnelMst

app = FastAPI(title="Rasta-AI")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)
session = Session()
searchLock = threading.Lock()  # tracemalloc in @measured is process-global: one search at a time

algorithms = {"bfs", "ucs", "greedy", "astar", "ids", "dls"}  # keys of the runners in plan()
heuristics = {
    "tunnelMst": lambda stops: tunnelMst(session.distances, stops=stops),
    "exactNearest": lambda stops: exactNearest(session.distances, stops=stops),
    "manhattanNearest": lambda stops: manhattanNearest(session.world, stops=stops),
    "manhattanFarthest": lambda stops: manhattanFarthest(session.world, stops=stops),
}


class ParamsRequest(BaseModel):
    overrides: dict[str, Any]


class PlanRequest(BaseModel):
    algorithm: str
    heuristic: str = "tunnelMst"
    weight: float = 1.0
    start: list[int] | None = None
    deliveredMask: int = 0
    carriedMask: int | None = None
    stops: list[list[int]] | None = None
    limit: int | None = None
    maxNodes: int | None = None


def badRequest(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def checkCell(value) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise badRequest(f"cell must be [row, col], got {value!r}")
    r, c = value
    rows, cols = session.world.rows, session.world.cols
    if not (0 <= r < rows and 0 <= c < cols) or session.instance.grid[r, c] < 0:
        raise badRequest(f"cell {[r, c]} is outside the grid or blocked")
    return (r, c)


@app.get("/health")
def health():
    return {"status": "ok", "stamp": stampLine}


@app.get("/instance")
def instance():
    return session.payload()


@app.post("/params")
def setParams(req: ParamsRequest):
    try:
        session.update(req.overrides)
    except (ValueError, TypeError) as err:
        raise badRequest(str(err)) from err
    return session.payload()


@app.post("/params/reset")
def resetParams():
    session.reset()
    return session.payload()


@app.post("/plan")
def plan(req: PlanRequest):
    if req.algorithm not in algorithms:
        raise badRequest(f"unknown algorithm {req.algorithm!r}")
    if req.heuristic not in heuristics:
        raise badRequest(f"unknown heuristic {req.heuristic!r}")
    fullMask = session.world.fullMask
    if not 0 <= req.deliveredMask <= fullMask or (
        req.carriedMask is not None and not 0 <= req.carriedMask <= fullMask
    ):
        raise badRequest("mask out of range")
    stops = None
    if req.stops is not None:
        stops = tuple(checkCell(cell) for cell in req.stops)
        if not stops or not set(stops) <= set(session.instance.stops):
            raise badRequest("stops must be a non-empty subset of the instance stops")
    start = None if req.start is None else checkCell(req.start)
    problem = DeliveryProblem(
        session.world,
        start=start,
        deliveredMask=req.deliveredMask,
        stops=stops,
        carriedMask=req.carriedMask,
    )
    if req.algorithm == "dls" and req.limit is None:
        raise badRequest("dls needs a limit")
    h = heuristics[req.heuristic](stops)
    runners = {
        "bfs": lambda: bfs(problem),
        "ucs": lambda: ucs(problem),
        "greedy": lambda: greedy(problem, h),
        "astar": lambda: aStar(problem, h, req.weight),
        "ids": lambda: ids(problem, maxNodes=req.maxNodes),
        "dls": lambda: dls(problem, req.limit, req.maxNodes),
    }
    with searchLock:
        result = runners[req.algorithm]()
    return {
        "found": result.found,
        "cost": result.cost if result.found else None,
        "path": [list(state[0]) for state in result.path],
        "actions": result.actions,
        "nodesExpanded": result.nodesExpanded,
        "runtime": result.runtime,
        "peakMemory": result.peakMemory,
        "stamp": stampLine,
    }
