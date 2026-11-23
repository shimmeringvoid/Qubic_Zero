"""Qubic_Zero: an AlphaZero-style player for 4 x 4 x 4 tic-tac-toe, and the exact tools to check it."""

from .exact import (
    SearchBudgetExceeded,
    Table,
    ThreatResult,
    WalkReport,
    check_wins_every_game,
    forced_win_length,
    solve,
    solve_table,
    threat_moves,
    threat_search,
)
from .games import GAME_NAMES, LineGame, cube3, game_by_name, qubic, tictactoe
from .players import MatchResult, completing_cells, play_games, random_player, rule_player

__version__ = "0.1.0"

__all__ = [
    "LineGame",
    "tictactoe",
    "cube3",
    "qubic",
    "game_by_name",
    "GAME_NAMES",
    "random_player",
    "rule_player",
    "completing_cells",
    "play_games",
    "MatchResult",
    "Table",
    "solve_table",
    "solve",
    "threat_search",
    "threat_moves",
    "forced_win_length",
    "ThreatResult",
    "check_wins_every_game",
    "WalkReport",
    "SearchBudgetExceeded",
]
