from .BestFirst import bestFirst
from .SearchProblem import SearchProblem, SearchResult, measured


@measured
def ucs(problem: SearchProblem) -> SearchResult:
    return bestFirst(problem, lambda state: 0.0, lambda g, h: g)
