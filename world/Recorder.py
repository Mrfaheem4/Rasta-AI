class Recorder:
    """Wraps a SearchProblem and logs each expanded state (successors runs once per expansion)."""

    def __init__(self, problem):
        self.problem = problem
        self.expanded = []

    def initialState(self):
        return self.problem.initialState()

    def isGoal(self, state) -> bool:
        return self.problem.isGoal(state)

    def successors(self, state):
        self.expanded.append(state)
        return self.problem.successors(state)

    def firstPerCellLayer(self) -> list[list[int]]:
        """[row, col, layer] for the first expansion of each (cell, layer); layer = parcels delivered."""
        seen, out = set(), []
        for state in self.expanded:
            cell, layer = state[0], bin(state[1]).count("1")
            if (cell, layer) not in seen:
                seen.add((cell, layer))
                out.append([cell[0], cell[1], layer])
        return out
