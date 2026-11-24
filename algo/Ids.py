from .Dls import depthLimited, resultFromPath
from .SearchProblem import SearchProblem, SearchResult, measured


@measured
def ids(problem: SearchProblem, maxDepth: int = 200, maxNodes: int | None = None) -> SearchResult:
    """Iterative deepening. The depth of the solution is len(result.actions)."""
    total = 0
    for limit in range(maxDepth + 1):
        remaining = None if maxNodes is None else maxNodes - total
        path, expanded, cutoff = depthLimited(problem, limit, remaining)
        total += expanded
        if path is not None:
            return resultFromPath(path, total)
        if not cutoff or (maxNodes is not None and total >= maxNodes):
            break
    return SearchResult(False, nodesExpanded=total)
