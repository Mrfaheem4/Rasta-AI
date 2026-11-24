from .BestFirst import bestFirst
from .SearchProblem import Heuristic, SearchProblem, SearchResult, measured


@measured
def greedy(problem: SearchProblem, heuristic: Heuristic) -> SearchResult:
    return bestFirst(problem, heuristic, lambda g, h: h, reopen=False)
