import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qubic_zero import (  # noqa: E402
    completing_cells,
    qubic,
    random_player,
    rule_player,
    solve_table,
    threat_search,
    tictactoe,
)
from qubic_zero.agents import PolicyPlayer, SearchPlayer, ThreatPlayer, one_at_a_time  # noqa: E402
from qubic_zero.evaluate import TacticsSuite, match, table_score, value_error  # noqa: E402
from qubic_zero.net import make_net  # noqa: E402


@pytest.fixture(scope="module")
def suite():
    return TacticsSuite.build(qubic(), per_kind=40, seed=0)


def test_network_players_answer_with_empty_cells():
    game = qubic()
    net = make_net(game, "line", seed=0, width=16, depth=2)
    rng = np.random.default_rng(0)
    boards = game.empty(30)
    for _ in range(9):
        boards = game.play(boards, random_player(game, boards, rng))
    for player in (PolicyPlayer(net), SearchPlayer(net, n_sims=8, batch=16)):
        moves = player(game, boards)
        assert moves.shape == (30,) and (boards[np.arange(30), moves] == 0).all()
    assert one_at_a_time(game, PolicyPlayer(net))(boards[0]) == PolicyPlayer(net)(game, boards[:1])[0]
    visits, values = SearchPlayer(net, n_sims=8).visits(game, boards)
    assert (visits.sum(axis=1) == 8).all() and (np.abs(values) <= 1).all()


def test_the_threat_search_in_front_of_a_player_takes_forced_wins(suite):
    game = suite.game
    always_first = lambda game, boards, rng=None: (boards == 0).argmax(axis=1)  # noqa: E731
    player = ThreatPlayer(always_first)
    for kind in ("win", "force"):
        assert suite.score(player)[kind] == 1.0
    assert suite.score(always_first)["force"] < 0.5
    quiet = game.empty(3)
    assert (player(game, quiet) == 0).all()  # nothing to force on an empty board: the wrapped player decides


def test_tactics_suite_holds_positions_with_the_answers_it_claims(suite):
    game = suite.game
    assert set(suite.boards) == {"win", "block", "force"} and all(len(b) == 40 for b in suite.boards.values())
    assert "40 win" in repr(suite)
    for kind, boards in suite.boards.items():
        assert len({b.tobytes() for b in boards}) == len(boards) and not game.finished(boards).any()
        assert suite.right[kind].any(axis=1).all() and (boards[suite.right[kind]] == 0).all()
    me = game.to_move(suite.boards["win"])
    assert (completing_cells(game, suite.boards["win"], me) == suite.right["win"]).all()
    # in a "block" position there is one right move, and any other loses to the reply
    blocks = suite.boards["block"]
    assert (suite.right["block"].sum(axis=1) == 1).all()
    wrong = (blocks == 0) & ~suite.right["block"]
    after = game.play(blocks, wrong.argmax(axis=1))
    assert completing_cells(game, after, game.to_move(after)).any(axis=1).all()
    # a "force" position has no line to complete, and the threat search confirms each listed move
    forces = suite.boards["force"]
    assert not completing_cells(game, forces, game.to_move(forces)).any()
    for board, right in zip(forces[:10], suite.right["force"][:10]):
        assert right[threat_search(game, board).move]
    rng = np.random.default_rng(0)
    assert suite.score(rule_player, rng)["win"] == 1.0 and suite.score(rule_player, rng)["block"] == 1.0
    assert suite.score(random_player, rng)["block"] < 0.3


def test_scores_against_the_table_of_tictactoe():
    game = tictactoe()
    table = solve_table(game, discount=0.9)
    idx = np.flatnonzero(table.decision)

    def perfect(game, boards, rng=None):
        return table.fastest[table.lookup(boards)].argmax(axis=1)

    score = table_score(table, perfect)
    assert score["kept"] == 1.0 and score["mistakes"] == 0 and score["fastest"] == 1.0
    careless = table_score(table, random_player, np.random.default_rng(0))
    assert careless["kept"] < 0.8
    assert careless["mistakes"] == careless["mistakes_in_won"] + careless["mistakes_in_drawn"] + careless["mistakes_in_lost"]
    assert careless["mistakes_in_lost"] == 0  # a lost position cannot be made worse
    exact = value_error(table, table.value[idx])
    assert exact["value_error"] == 0.0 and exact["value_outcome_right"] == 1.0
    assert value_error(table, np.zeros(len(idx)))["value_outcome_right"] == pytest.approx((table.outcome[idx] == 0).mean())
    result = match(game, perfect, random_player, 200, np.random.default_rng(1))
    assert result["first_lost"] == 0 and result["second_lost"] == 0 and result["first_won"] > 0.8
