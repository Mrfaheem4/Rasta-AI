import itertools
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from algo import aStar, bfs, dls, greedy, ids, ucs
from api.Session import Session
from api.SimSession import SimSession
from core.Config import stampLine
from world import DeliveryProblem
from world.Heuristics import exactNearest, manhattanFarthest, manhattanNearest, tunnelMst
from world.Recorder import Recorder

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


class ModeRequest(BaseModel):
    mode: str


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


@app.post("/mode")
def setMode(req: ModeRequest):
    try:
        session.setMode(req.mode)
    except (ValueError, TypeError) as err:
        raise badRequest(str(err)) from err
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
    problem = Recorder(
        DeliveryProblem(
            session.world,
            start=start,
            deliveredMask=req.deliveredMask,
            stops=stops,
            carriedMask=req.carriedMask,
        )
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
        "expanded": problem.firstPerCellLayer(),
        "expansionCount": len(problem.expanded),
        "stamp": stampLine,
    }


class SimStartRequest(BaseModel):
    seed: int = 0
    rival: str = "blocker"
    maxSteps: int | None = None


class SimStepRequest(BaseModel):
    simId: int
    steps: int = 1


sims = {}  # simId -> SimSession, oldest first; at most maxSims are kept
simCounter = itertools.count(1)
maxSims = 8


def startSim(req: SimStartRequest):
    if req.rival not in ("blocker", "passive"):
        raise badRequest(f"unknown rival {req.rival!r}")
    if req.maxSteps is not None and req.maxSteps < 1:
        raise badRequest("maxSteps must be at least 1")
    with searchLock:
        plannedSteps = len(aStar(DeliveryProblem(session.world), tunnelMst(session.distances)).actions)
        sim = SimSession(session.world, session.distances, plannedSteps, req.seed, req.rival, req.maxSteps)
    simId = next(simCounter)
    if len(sims) >= maxSims:
        sims.pop(next(iter(sims)))
    sims[simId] = sim
    return simId, sim


@app.post("/sim/start")
def simStart(req: SimStartRequest):
    simId, sim = startSim(req)
    return {
        "simId": simId,
        "maxSteps": sim.maxSteps,
        "rival": sim.rivalName,
        "frame": sim.initialFrame(),
        "stamp": stampLine,
    }


@app.post("/sim/step")
def simStep(req: SimStepRequest):
    if not 1 <= req.steps <= 50:
        raise badRequest("steps must be between 1 and 50")
    sim = sims.get(req.simId)
    if sim is None:
        raise HTTPException(status_code=404, detail=f"unknown simId {req.simId}")
    frames = []
    with searchLock:
        while len(frames) < req.steps and not sim.finished():
            frames.append(sim.step())
    return {"simId": req.simId, "frames": frames, "stamp": stampLine}


@app.post("/sim/run")
def simRun(req: SimStartRequest):
    simId, sim = startSim(req)
    frames = [sim.initialFrame()]
    with searchLock:
        while not sim.finished():
            frames.append(sim.step())
    return {
        "simId": simId,
        "maxSteps": sim.maxSteps,
        "outcome": frames[-1].get("outcome"),
        "frames": frames,
        "stamp": stampLine,
    }
