from collections import deque

from .SearchProblem import SearchProblem, SearchResult, buildResult, measured


@measured
def bfs(problem: SearchProblem) -> SearchResult:
    start = problem.initialState()
    parents = {start: None}
    if problem.isGoal(start):
        return buildResult(parents, start, 0.0, 0)
    frontier = deque([(start, 0.0)])
    nodesExpanded = 0
    while frontier:
        state, g = frontier.popleft()
        nodesExpanded += 1
        for action, nextState, stepCost in problem.successors(state):
            if nextState in parents:
                continue
            parents[nextState] = (state, action)
            if problem.isGoal(nextState):
                return buildResult(parents, nextState, g + stepCost, nodesExpanded)
            frontier.append((nextState, g + stepCost))
    return SearchResult(False, nodesExpanded=nodesExpanded)
