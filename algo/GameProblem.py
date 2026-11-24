from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Protocol

State = Hashable
Move = Hashable


class GameProblem(Protocol):
    """Two-player zero-sum game. evaluate() scores a state from the MAX player's point of view."""

    def isTerminal(self, state: State) -> bool: ...
    def evaluate(self, state: State) -> float: ...
    def moves(self, state: State, maximizing: bool) -> Iterable[tuple[Move, State]]:
        """Yield (move, nextState) for the player to move."""
        ...


MoveOrder = Callable[[State, list[tuple[Move, State]], bool], list[tuple[Move, State]]]


@dataclass
class GameResult:
    value: float
    move: Move | None
    nodes: int  # states visited, for the pruning comparison
