# Example file showing a circle moving on screen

import argparse
from dataclasses import fields
import numpy as np

from glicko2 import update_all_glicko2s, reset_all_glicko2s
from competition_utils import *
from utils import *
from yinsh import *


def make_valid_move(yinsh_game, valid_move):
    # non valid moves are cheating and may break the game
    yinsh_game.take_turn(valid_move)


def print_game_state_initialization(game_state: GameState):
    print("game_state = GameState(")
    for field in fields(GameState):
        value = getattr(game_state, field.name)
        print(f"    {field.name} = {repr(value)}")
    print(")")


def play_game(player1, player2, yinsh_game):
    players = [player1, player2]
    # print(players)
    gs = yinsh_game.get_game_state()
    while not gs.terminal:
        player = players[gs.active_player]
        move = player.make_move(gs)
        # assumed to be valid
        make_valid_move(yinsh_game, move)
        gs = yinsh_game.get_game_state()
        # print(f"Player {gs.active_player+1} {gs.turn_type}: {move}")
    return gs


def get_wins_losses_draws(points):
    return (
        [int(points[0] > points[1]), int(points[0] < points[1])],
        [int(points[0] < points[1]), int(points[0] > points[1])],
        [int(points[0] == points[1]), int(points[0] == points[1])],
    )


def main(player1, player2, num):
    # fw = f"bot_scores/{player1}_{player2}_wins.npy"
    # fl = f"bot_scores/{player1}_{player2}_losses.npy"
    # fd = f"bot_scores/{player1}_{player2}_draws.npy"

    # try:
    #     wins = np.load(fw)
    #     losses = np.load(fl)
    #     draws = np.load(fd)
    # except Exception as e:
    #     print(e)
    #     wins = np.array([0, 0])
    #     losses = np.array([0, 0])
    #     draws = np.array([0, 0])

    #     np.save(fw, wins)
    #     np.save(fl, losses)
    #     np.save(fd, draws)

    # bot1 = get_bot(player1)
    # bot2 = get_bot(player2)

    # win, loss, draw = get_wins_losses_draws(state.points)
    # # print(state.points)
    # # del state.board, state, yinsh_game

    # wins = wins + np.array(win)
    # losses = losses + np.array(loss)
    # draws = draws + np.array(draw)

    # print(f"wins: {wins}")
    # print(f"loss: {losses}")
    # print(f"draw: {draws}")

    # np.save(fw, wins)
    # np.save(fl, losses)
    # np.save(fd, draws)
    try:
        player_data = load_data()
        # print(player_data)
    except:
        player_data = {}
    player_data = new_bot_data_entry(player1, player_data, overwrite=False)
    player_data = new_bot_data_entry(player2, player_data, overwrite=False)

    bot1 = get_bot(player1, 0)
    bot2 = get_bot(player2, 1)

    elo1 = player_data[player1]["full_elo"]
    elo2 = player_data[player2]["full_elo"]
    print(f"{player1}: elo = {elo1}")
    print(f"{player2}: elo = {elo2}")

    yinsh_game = YinshGame()
    # print(yinsh_game.board)
    # print_game_state_initialization(yinsh_game.get_game_state())
    state = play_game(bot1, bot2, yinsh_game)
    winner = get_winner(state.points)

    if winner < 3:
        print(
            f"---The winner of game {num[0]} / {num[1]} is {[player1, player2][winner-1]}!---"
        )
    else:
        print("---Draw---")

    update_score_information(player1, player2, winner, player_data)
    elo1 = player_data[player1]["full_elo"]
    elo2 = player_data[player2]["full_elo"]
    print(f"new {player1}: elo = {elo1}")
    print(f"new {player2}: elo = {elo2}")
    print("scores updated successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    rng = np.random.default_rng()
    parser.add_argument("num_games", type=int, default=1)
    parser.add_argument("-b1", "--bot1", type=str, default="--")
    parser.add_argument("-b2", "--bot2", type=str, default="--")
    args = parser.parse_args()
    print(args)
    g = 0
    games = []
    if args.bot1 == "--":
        bot_list1 = bot_list
    else:
        bot_list1 = [args.bot1]

    if args.bot2 == "--":
        bot_list2 = bot_list
    else:
        bot_list2 = [args.bot2]

    for b1 in bot_list1:
        for b2 in bot_list2:
            if b1 != b2:
                games.append((b1, b2))
    for i in range(args.num_games):
        np.random.shuffle(games)
        for game in games:
            g += 1
            main(
                game[0],
                game[1],
                (g, len(games) * args.num_games),
            )
        data = load_data()
        games_played = 0
        num_bots = len(bot_list)
        for b in bot_list:
            games_played += data[b]["games_played"]
        # if games_played / num_bots >= 20:
        #     reset_all_glicko2s()
        #     update_all_glicko2s()

    # fw = f"bot_scores/{args.player1}_{args.player2}_wins.npy"
    # fl = f"bot_scores/{args.player1}_{args.player2}_losses.npy"
    # fd = f"bot_scores/{args.player1}_{args.player2}_draws.npy"
    # wins = np.load(fw)
    # losses = np.load(fl)
    # draws = np.load(fd)
    # print(f"wins: {wins}")
    # print(f"loss: {losses}")
    # print(f"draw: {draws}")

    # print(f"total: {wins[0]+losses[0]+draws[0]}")
