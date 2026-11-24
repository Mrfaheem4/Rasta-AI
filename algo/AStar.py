from .BestFirst import bestFirst
from .SearchProblem import Heuristic, SearchProblem, SearchResult, measured


@measured
def aStar(problem: SearchProblem, heuristic: Heuristic, weight: float = 1.0) -> SearchResult:
    return bestFirst(problem, heuristic, lambda g, h: g + weight * h)
