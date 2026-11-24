import heapq
import itertools
from collections.abc import Callable

from .SearchProblem import Heuristic, SearchProblem, SearchResult, buildResult

Priority = Callable[[float, float], float]


def bestFirst(
    problem: SearchProblem, heuristic: Heuristic, priority: Priority, reopen: bool = True
) -> SearchResult:
    """Graph best-first search keyed by priority(g, h); goal test on expansion.

    reopen=False never revisits a generated state (greedy: priority ignores g, so a cheaper g changes nothing).
    """
    start = problem.initialState()
    parents = {start: None}
    gScore = {start: 0.0}
    tieBreak = itertools.count()
    heap = [(priority(0.0, heuristic(start)), next(tieBreak), 0.0, start)]
    nodesExpanded = 0
    while heap:
        _, _, g, state = heapq.heappop(heap)
        if g > gScore[state]:
            continue
        if problem.isGoal(state):
            return buildResult(parents, state, g, nodesExpanded)
        nodesExpanded += 1
        for action, nextState, stepCost in problem.successors(state):
            newG = g + stepCost
            if newG < gScore.get(nextState, float("inf")) and (reopen or nextState not in gScore):
                gScore[nextState] = newG
                parents[nextState] = (state, action)
                key = priority(newG, heuristic(nextState))
                heapq.heappush(heap, (key, next(tieBreak), newG, nextState))
    return SearchResult(False, nodesExpanded=nodesExpanded)
