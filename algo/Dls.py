from .SearchProblem import SearchProblem, SearchResult, buildResult, measured


def depthLimited(problem: SearchProblem, limit: int, maxNodes: int | None = None):
    """Unmeasured core shared with ids. Returns (path or None, nodesExpanded, cutoffHit).

    path is a list of (state, action, g) from the root to the goal.
    """
    start = problem.initialState()
    path = [(start, None, 0.0)]
    onPath = {start}
    nodesExpanded = 0
    cutoff = False

    def visit(state, g: float, depth: int) -> bool:
        nonlocal nodesExpanded, cutoff
        if problem.isGoal(state):
            return True
        if depth == limit:
            cutoff = True
            return False
        if maxNodes is not None and nodesExpanded >= maxNodes:
            cutoff = True
            return False
        nodesExpanded += 1
        for action, nextState, stepCost in problem.successors(state):
            if nextState in onPath:
                continue
            path.append((nextState, action, g + stepCost))
            onPath.add(nextState)
            if visit(nextState, g + stepCost, depth + 1):
                return True
            onPath.discard(nextState)
            path.pop()
        return False

    found = visit(start, 0.0, 0)
    return (path if found else None), nodesExpanded, cutoff


def resultFromPath(path: list, nodesExpanded: int) -> SearchResult:
    parents = {path[0][0]: None}
    for i in range(1, len(path)):
        parents[path[i][0]] = (path[i - 1][0], path[i][1])
    goal, cost = path[-1][0], path[-1][2]
    return buildResult(parents, goal, cost, nodesExpanded)


@measured
def dls(problem: SearchProblem, limit: int, maxNodes: int | None = None) -> SearchResult:
    path, nodesExpanded, _ = depthLimited(problem, limit, maxNodes)
    if path is None:
        return SearchResult(False, nodesExpanded=nodesExpanded)
    return resultFromPath(path, nodesExpanded)
