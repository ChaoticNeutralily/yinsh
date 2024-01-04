"""Implements http://www.glicko.net/glicko/glicko2.pdf"""
import numpy as np
import pickle

TAU = 0.5
INITIAL_RATING = 1500
INITIAL_DEV = 350  # dev is RD is rating deviation
INITIAL_VOLATILITY = 0.06  # \sigma from paper
G2_SCALE = 400 * np.log10(np.e)


def glicko2_rating(rating):
    """\mu from paper.

    mu = (r - 1500) / 173.7178
    173.7178 is 400 * log_10(e), scales score into glicko 2 range
    """
    return (rating - INITIAL_RATING) / G2_SCALE


def glicko2_dev(deviation):
    """\phi from paper.

    phi = RD / 173.7178
    deviation in glicko2 rating
    """
    return deviation / G2_SCALE


def g(deviations):
    return 1 / np.sqrt(1 + 3 * (deviations**2) / (np.pi**2))


def expected_scores(rating, ratings, deviations):
    """expected scores of player with rating

    against opponents with ratings and deviations( of those ratings).
    """
    return 1 / (1 + np.exp(-g(deviations) * (rating - ratings)))


def game_outcome_variance(rating, ratings, deviations, gs, expected_scores):
    return 1 / (sum((gs**2) * expected_scores * (1 - expected_scores)))


def estimate_rating_improvement(
    player_rating, opponent_ratings, opponent_deviations, player_scores_vs_opponents
):
    """\Delta from the paper

    estimated improvement in rating by comparing pre-period rating to the performance rating based only on game outcomes.
    """
    variance = game_outcome_variance(rating, ratings, deviations)
    e = expected_scores(rating, ratings, deviations)
    gs = g(deviations)
    return variance * sum(gs * (scores - e))


def ff(x, D2, p2, v, a):
    ex = np.exp(x)
    return ((ex) * (D2 - p2 - v - ex)) / (2 * (p2 + v + ex) ** 2) - (
        (x - a) / (TAU**2)
    )


def initial_log_volativity_bounds(volatility, D2, p2, v, f):
    upper = 2 * np.log(volatility)
    if D2 > p2 + variance:
        lower = np.log(D2 - p2 - v)
    else:
        k = 1
        while f(A - k * TAU) < 0:
            k += 1
        lower = upper - k * TAU
    return upper, lower


def compute_volatility(
    old_volatility,
    estimated_rating_improvement,
    deviation,
    variance,
    tolerance=0.000001,
):
    """Run iterative optimization to compute new volatility within tolerance.

    This is \sigma' from paper
    Illiniois algorithm procedure from the paper.
    """
    a = 2 * np.log(old_volatility)
    D2 = estimated_rating_improvement**2
    p2 = deviation**2
    f = lambda x: ff(x, D2, p2, volatility, a)
    upper, lower = initial_log_volativity_bounds(old_volatility, D2, p2, variance, f)
    f_upper = f(upper)
    f_lower = f(lower)
    iters = 0
    while np.abs(upper - lower) > tolerance:
        print(f"LB, UB, diff = {lower}, {upper}, {np.abs(upper-lower)}")
        new = upper + (upper - lower) * f_upper / (f_lower - f_upper)
        f_new = f(new)
        if f_new * f_lower <= 0:
            upper = lower
            f_upper = f_lower
        else:
            f_upper = f_upper / 2
        lower = new
        f_lower = f_new
        iters += 1

    print(f"iterations to compute volatility: {iters}")
    volatility = np.exp(upper / 2)


def new_pre_period_dev(deviation, volatility):
    """If a player does not compete in rating period, then this is their new dev.

    Rating and volatility remain the same.
    """
    return np.sqrt(old_deviation**2 + new_volatility**2)


def new_glicko2_dev(new_pre_period_dev, outcome_variance):
    return 1 / np.sqrt((1 / new_pre_period_dev**2) + 1 / outcome_variance)


def new_glicko2_rating(glicko2_rating, new_glicko2_dev, gs, scores, e_scores):
    glicko2_rating + (new_glicko2_dev**2) * sum(gs * (scores - e_scores))


def unglicko2_rating(glicko2_rating):
    """r = unglicko2_rating(glicko2_rating(r))."""
    return G2_SCALE * glicko2_rating + INITIAL_RATING


def unglicko2_dev(dev):
    """d = unglicko2_dev(glicko2_dev(dev))."""
    return G2_SCALE * dev


def get_player_scores_vs_opponents(index):
    # 1 for win, 0 for loss, 0.5 for draw
    wins = np.load("scores/wins1.npy")
    draws = np.load("scores/draws1.npy")
    p1_scores = wins[index, :] + 0.5 * draws[index, :]
    # p1_scores = np.concatenate(p1_scores[:index], p1_scores[index + 1 :], axis=1)

    wins = np.load("scores/wins2.npy")
    draws = np.load("scores/draws2.npy")
    p2_scores = wins[index, :] + 0.5 * draws[index, :]
    # p2_scores = np.concatenate(p2_scores[:index], p2_scores[index + 1 :], axis=1)

    full_scores = p1_scores + p2_scores

    return p1_scores, p2_scores, full_scores


# def get_all_scores():
#     # scores against each opponent
#     wins1 = np.load("bot_scores/wins1.npy")
#     wins2 = np.load("bot_scores/wins2.npy")
#     draws = np.load("bot_scores/draws.npy")
#     score1_data = wins1 + draws / 2
#     score2_data = wins2 + draws / 2
#     p1_scores = np.reshape(np.sum(score1_data, axis=1), (-1,))
#     p2_scores = np.reshape(np.sum(score2_data, axis=1), (-1,))
#     # full score won't use self play to rank
#     # full_scores = p1_scores + p2_scores - np.diag(wins) - np.diag(draws) / 2
#     return p1_scores, p2_scores, full_scores


def get_rdv(data, num_players, prefix):
    ratings = np.zeros((num_players,))
    deviations = np.zeros((num_players,))
    volatilities = np.zeros((num_players,))
    for opponent in data.keys():
        tmp = data[opponent]
        for arr, info in zip(
            [ratings, deviation, volatility], ["rating", "deviation", "volatility"]
        ):
            arr[tmp[index]] = tmp[f"{prefix}_{info}"]
    return glicko2_rating(ratings), glicko2_dev(deviations), volatilities


def update_ranks_of_existing_player(player):
    # step 1
    data = load_data()
    num_players = len(data)
    index = data[player]["index"]
    p1_scores, p2_scores, full_scores = get_player_scores_vs_opponents(index)
    new_player_data = {}
    for prefix, score, scores in zip(
        ["p1", "p2", "full"], [p1_scores, p2_scores, full_scores]
    ):
        # step 2
        ratings, deviations, volatilities = get_rdv(data, num_players, prefix)
        rating = ratings[index]
        ratings = np.concatenate(
            (ratings[:index], ratings[index + 1 :]), axis=None
        )  # debug check this, no internet rn
        deviation = deviations[index]
        deviations = np.concatenate(
            (deviations[:index], deviations[index + 1 :]), axis=None
        )
        scores = np.concatenate((scores[:index], scores[index + 1 :]), axis=None)
        # step 3
        e_scores = expected_scores(rating, ratings, deviations)
        gs = g(deviations)
        var = game_outcome_variance(rating, ratings, deviations, gs, e_scores)
        # step 4
        est_improvement = estimate_rating_improvement(
            rating, ratings, deviations, scores
        )
        # step 5
        new_volatility = compute_volatility(
            data[player]["volatility"],
            estimated_rating_improvement,
            deviation,
            variance,
        )
        # step 6
        if data["player"]["games_played"] == 0:
            pre_period_dev = new_pre_period_dev(deviation, volatility)
            new_gdeviation = pre_period_dev
            new_grating = rating
        else:
            # step 7
            pre_period_dev = new_pre_period_dev(deviation, new_volatility)
            new_gdeviation = new_glicko2_dev(pre_period_dev, variance)
            new_grating = new_glicko2_rating(
                rating, new_gdeviation, gs, scores, e_scores
            )
        # step 8
        nr = unglicko2_rating(new_grating)
        nd = unglicko2_dev(new_gdeviation)
        new_player_data[f"{prefix}_rating"] = nr
        new_player_data[f"{prefix}_deviation"] = nd
        new_player_data[f"{prefix}_volatility"] = new_volatility
    return new_player_data
    # load in {player1}_{*}_wins


# scores = scores against each oponent (+1 for each win, +0.5 for each draw, 0 for each loss)


# bot_data = {
#     bot_name: {
#         rating:
#
#     }
# }
def load_data():
    with open("scores/rating_data.pickle", "rb") as file:
        data = pickle.load(file)
    return data


def save_player_data(data):
    with open("scores/rating_data.pickle", "wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def new_bot_data_entry(bot_name, bot_data, overwrite=False):
    if overwrite or bot_data.get(bot_name) is None:
        if bot_data.get(bot_name) is not None:
            index = bot_data[bot_name]["index"]
        else:
            index = add_new_index_to_score_arrays()
        bot_data[bot_name] = {
            "p1_rating": INITIAL_RATING,
            "p1_deviation": INITIAL_DEV,
            "p1_volatility": INITIAL_VOLATILITY,
            "p2_rating": INITIAL_RATING,
            "p2_deviation": INITIAL_DEV,
            "p2_volatility": INITIAL_VOLATILITY,
            "full_rating": INITIAL_RATING,
            "full_deviation": INITIAL_DEV,
            "full_volatility": INITIAL_VOLATILITY,
            "index": index,
            "games_played": 0,
            "p1_elo": INITIAL_RATING,
            "p2_elo": INITIAL_RATING,
            "full_elo": INITIAL_RATING,
        }
    save_player_data(bot_data)

    return bot_data


def add_new_index_to_2darray(array):
    new_array = np.zeros((array.shape[0] + 1, array.shape[1] + 1))
    new_array[:-1, :-1] = array
    return new_array


def add_new_index_to_score_arrays() -> int:
    for filename in [
        "scores/wins1.npy",
        "scores/wins2.npy",
        "scores/losses1.npy",
        "scores/losses2.npy",
        "scores/draws1.npy",
        "scores/draws2.npy",
    ]:
        try:
            array = np.load(filename)
            index = array.shape[0]
            new_array = add_new_index_to_2darray(array)
            # print(array.shape)
        except:
            # print((0, 0))
            new_array = np.zeros((1, 1))
            index = 0
        np.save(filename, new_array)
    print(new_array.shape)
    return index


### thinking live updates with elo and staggered updates with glicko2

K = 32  # everyone will have K factor of 32 no matter their rating or games played. perhaps over sensisitive for established players and undersensitive for low ranked good players


def elo_expected_player_score(player_elo, opponent_elo):
    return 1 / (1 + 10 ** ((opponent_elo - player_elo) / 400))


def elo_update(player_elo, player_score, expected_score):
    return player_elo + K * (player_score - expected_score)
