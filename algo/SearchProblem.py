import time
import tracemalloc
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, field
from typing import Protocol

State = Hashable
Heuristic = Callable[[State], float]


class SearchProblem(Protocol):
    def initialState(self) -> State: ...
    def isGoal(self, state: State) -> bool: ...
    def successors(self, state: State) -> Iterable[tuple[str, State, float]]:
        """Yield (action, nextState, stepCost)."""
        ...


@dataclass
class SearchResult:
    found: bool
    path: list[State] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    cost: float = float("inf")
    nodesExpanded: int = 0
    runtime: float = 0.0
    peakMemory: int = 0  # bytes, from tracemalloc


def measured(search: Callable[..., SearchResult]) -> Callable[..., SearchResult]:
    """Wrap a search function so its result carries runtime and peak memory."""

    def wrapper(*args, **kwargs) -> SearchResult:
        tracemalloc.start()
        start = time.perf_counter()
        result = search(*args, **kwargs)
        result.runtime = time.perf_counter() - start
        result.peakMemory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return result

    wrapper.__name__ = search.__name__
    return wrapper


def buildResult(parents: dict, goal: State, cost: float, nodesExpanded: int) -> SearchResult:
    """Rebuild path and actions from parents[state] = (parentState, action); root maps to None."""
    path, actions, state = [goal], [], goal
    while parents[state] is not None:
        state, action = parents[state]
        path.append(state)
        actions.append(action)
    return SearchResult(True, path[::-1], actions[::-1], cost, nodesExpanded)
