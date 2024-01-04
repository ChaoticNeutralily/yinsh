from random_bots import UniformRandomPlayer
from heuristic_tree_bots import (
    FixedDepthMiniMaxTreePlayer,
    floyd_estimate,
    num_ring_connections,
    num_controlled_markers,
    num_unique_controlled_markers,
    markers_x10,
    total_ring_moves,
    combined_heuristic,
)
from glicko2 import (
    load_data,
    save_player_data,
    new_bot_data_entry,
    elo_expected_player_score,
    elo_update,
)
import numpy as np


bot_list = ["random", "floyd", "floyd_combined", "floyd_u", "connections", "moves"]


# num_controlled_markers, num_unique_controlled_markers, markers_x10, total_ring_moves, combined_heuristic
def get_bot(kind_of_player: str, player):
    depth = 3  # 3 on david's site
    start_depth = 1
    if kind_of_player == "random":
        return UniformRandomPlayer()
    elif kind_of_player == "floyd":
        return FixedDepthMiniMaxTreePlayer(player, depth, floyd_estimate, start_depth)
    elif kind_of_player == "floyd_combined":

        def estimate(game_state):
            return floyd_estimate(
                game_state,
                ring_heuristic=combined_heuristic(
                    [
                        num_controlled_markers,
                        num_unique_controlled_markers,
                        num_ring_connections,
                        total_ring_moves,
                    ]
                ),
            )

        return FixedDepthMiniMaxTreePlayer(player, depth, estimate, start_depth)

    elif kind_of_player == "floyd_u":

        def estimate(game_state):
            return floyd_estimate(
                game_state, ring_heuristic=num_unique_controlled_markers
            )

        return FixedDepthMiniMaxTreePlayer(player, depth, estimate, start_depth)
    elif kind_of_player == "connections":

        def estimate(game_state):
            return floyd_estimate(game_state, ring_heuristic=num_ring_connections)

        return FixedDepthMiniMaxTreePlayer(player, depth, estimate, start_depth)
    elif kind_of_player == "moves":

        def estimate(game_state):
            return floyd_estimate(game_state, ring_heuristic=total_ring_moves)

        return FixedDepthMiniMaxTreePlayer(player, depth, estimate, start_depth)
    return None  # "human", or any not implemented bot


def get_winner(points):
    if points[0] == max(points) and points[1] != points[0]:
        return 1
    elif points[1] != points[0]:
        return 2
    return 3  # it's a draw


def scores(winner):
    if winner == 1:
        sp1 = 1
        sp2 = 0
    elif winner == 2:
        sp1 = 0
        sp2 = 1
    else:
        sp1 = 0.5
        sp2 = 0.5
    return sp1, sp2


def update_full_elos(player_data, p1, p2, winner):
    sp1, sp2 = scores(winner)
    elo1 = player_data[p1]["full_elo"]
    elo2 = player_data[p2]["full_elo"]
    # update wins, losses, draws, elo using winner
    # p1 full elo
    ep1 = elo_expected_player_score(elo1, elo2)
    new_p1_elo = elo_update(elo1, sp1, ep1)
    player_data[p1]["full_elo"] = new_p1_elo

    ep2 = elo_expected_player_score(elo2, elo1)
    new_p2_elo = elo_update(elo2, sp2, ep2)
    player_data[p2]["full_elo"] = new_p2_elo


def update_win_loss_draw_arrays(winner, player_data, p1, p2):
    if winner == 3:
        # yes these could be one file/array and just transposed, very observant. I'm not efficient here
        draws1 = np.load("scores/draws1.npy")
        draws1[player_data[p1]["index"], player_data[p2]["index"]] += 1
        draws2 = np.load("scores/draws2.npy")
        draws2[player_data[p2]["index"], player_data[p1]["index"]] += 1
        np.save("scores/draws1.npy", draws1)
        np.save("scores/draws2.npy", draws2)
    elif winner == 1:
        # yes these could be one file/array and just transposed, very observant. I'm not efficient here
        wins1 = np.load("scores/wins1.npy")
        wins1[player_data[p1]["index"], player_data[p2]["index"]] += 1
        losses2 = np.load("scores/losses2.npy")
        losses2[player_data[p2]["index"], player_data[p1]["index"]] += 1
        np.save("scores/wins1.npy", wins1)
        np.save("scores/losses2.npy", losses2)
    elif winner == 2:
        # yes these could be one file/array and just transposed, very observant. I'm not efficient here
        losses1 = np.load("scores/losses1.npy")
        losses1[player_data[p1]["index"], player_data[p2]["index"]] += 1
        wins2 = np.load("scores/wins2.npy")
        wins2[player_data[p2]["index"], player_data[p1]["index"]] += 1
        np.save("scores/wins2.npy", wins2)
        np.save("scores/losses1.npy", losses1)


def update_score_information(p1, p2, winner, player_data):
    if p1 != p2:
        update_full_elos(player_data, p1, p2, winner)
    sp1, sp2 = scores(winner)
    # new p1's player 1 elo
    ep11 = elo_expected_player_score(
        player_data[p1]["p1_elo"], player_data[p2]["p2_elo"]
    )
    new_p1_elo1 = elo_update(player_data[p1]["p1_elo"], ep11, sp1)

    # new p2's player 2 elo
    ep22 = elo_expected_player_score(
        player_data[p2]["p2_elo"], player_data[p1]["p1_elo"]
    )
    new_p2_elo2 = elo_update(player_data[p2]["p2_elo"], ep22, sp2)

    # update the 1 sided elos computed above
    player_data[p1]["p1_elo"] = new_p1_elo1
    player_data[p2]["p2_elo"] = new_p2_elo2

    # save all these updates to the pickled player data
    player_data[p1]["games_played"] = player_data[p1]["games_played"] + 1
    player_data[p2]["games_played"] = player_data[p2]["games_played"] + 1
    save_player_data(player_data)
    # update wins and losses for computing glicko2 and seeing stats
    update_win_loss_draw_arrays(winner, player_data, p1, p2)
