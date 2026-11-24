from algo.GameProblem import GameProblem, MoveOrder


def orderMoves(game: GameProblem) -> MoveOrder:
    """Cheap 1-ply ordering: MAX tries the best-looking child first, MIN the lowest."""

    def order(state, moveList, maximizing):
        return sorted(moveList, key=lambda mv: game.evaluate(mv[1]), reverse=maximizing)

    return order
