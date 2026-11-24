from .GameProblem import GameProblem, GameResult


def minimax(problem: GameProblem, state, depth: int, maximizing: bool = True) -> GameResult:
    """Depth-limited minimax. Leaves (depth 0, terminal, or no moves) are scored by evaluate."""
    nodes = 0

    def value(s, d: int, maxing: bool):
        nonlocal nodes
        nodes += 1
        options = [] if d == 0 or problem.isTerminal(s) else list(problem.moves(s, maxing))
        if not options:
            return problem.evaluate(s), None
        best, bestMove = None, None
        for move, nxt in options:
            v, _ = value(nxt, d - 1, not maxing)
            if best is None or (v > best if maxing else v < best):
                best, bestMove = v, move
        return best, bestMove

    v, move = value(state, depth, maximizing)
    return GameResult(v, move, nodes)
