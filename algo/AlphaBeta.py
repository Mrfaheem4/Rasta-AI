from .GameProblem import GameProblem, GameResult, MoveOrder


def alphaBeta(
    problem: GameProblem, state, depth: int, maximizing: bool = True, order: MoveOrder | None = None
) -> GameResult:
    """Depth-limited minimax with alpha-beta pruning; order reorders moves at every node."""
    nodes = 0

    def value(s, d: int, maxing: bool, alpha: float, beta: float):
        nonlocal nodes
        nodes += 1
        options = [] if d == 0 or problem.isTerminal(s) else list(problem.moves(s, maxing))
        if not options:
            return problem.evaluate(s), None
        if order is not None:
            options = order(s, options, maxing)
        best, bestMove = None, None
        for move, nxt in options:
            v, _ = value(nxt, d - 1, not maxing, alpha, beta)
            if best is None or (v > best if maxing else v < best):
                best, bestMove = v, move
            if maxing:
                alpha = max(alpha, best)
            else:
                beta = min(beta, best)
            if alpha >= beta:
                break
        return best, bestMove

    v, move = value(state, depth, maximizing, float("-inf"), float("inf"))
    return GameResult(v, move, nodes)
