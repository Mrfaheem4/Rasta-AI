from .BestFirst import bestFirst
from .SearchProblem import SearchProblem, State


class Exhaustive:
    """Wraps a problem so no state is a goal: best-first then settles every reachable state."""

    def __init__(self, problem: SearchProblem):
        self.problem = problem

    def initialState(self) -> State:
        return self.problem.initialState()

    def isGoal(self, state: State) -> bool:
        return False

    def successors(self, state: State):
        return self.problem.successors(state)


def costField(problem: SearchProblem) -> dict[State, float]:
    """Exact cheapest cost from the initial state to every reachable state (exhaustive UCS)."""
    return bestFirst(Exhaustive(problem), lambda state: 0.0, lambda g, h: g).reached
