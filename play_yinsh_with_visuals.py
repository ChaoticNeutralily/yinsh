# Example file showing a circle moving on screen

import argparse
from dataclasses import fields
from math import sqrt
import numpy as np
import time

import pygame

# from utils import *
from pieces import Piece, screen_point
from utils import *
from yinsh import *


def pgvec(coord):
    return pygame.Vector2((screen_point(*coord)))


VALID = 2
HIGHLIGHT = 1
ACTUAL = 0


def draw_move(move, turn_type, active_player, color_index: int, screen):
    if (
        turn_type == "setup new rings"
        or turn_type == "move ring"
        or turn_type == "remove ring"
    ):
        highlight = Piece(move, active_player, "ring", color_index)
        highlight.draw(screen)
    elif turn_type == "add marker":
        highlight = Piece(move, active_player, "marker", color_index)
        highlight.draw(screen)
    elif turn_type == "remove run":
        for marker in move:
            highlight = Piece(marker, active_player, "marker", color_index)
            highlight.draw(screen)


def get_remove_run_from_mouse(runs):
    pos = pygame.mouse.get_pos()
    # find any run(s) mouse is hovering over
    cc = closest_coord(*pos)
    cc_in_runs = []
    for run in runs:
        if cc in run:
            cc_in_runs.append(run)
    # only return a move if there is at least one run mouse is over
    if len(cc_in_runs) > 0:
        # return the single run
        if len(cc_in_runs) == 1:
            return cc_in_runs[0]
        elif len(cc_in_runs) > 1:
            # tie breaker for multiple runs with cumulative distance to markers
            run_dists = []
            for run in cc_in_runs:
                curr_dists2 = 0
                for marker in run:
                    pt = screen_point(*marker)
                    curr_dists2 += (pt[0] - pos[0]) ** 2 + (pt[1] - pos[1]) ** 2
                run_dists.append(curr_dists2)
            index_min = min(range(len(run_dists)), key=run_dists.__getitem__)
            closest_run = cc_in_runs[index_min]
            return closest_run
    return []


def highlight_human_move(valid_moves, screen, turn_type, active_player):
    if turn_type != "remove run":
        move = closest_coord(*pygame.mouse.get_pos())
        if move in valid_moves:
            draw_move(move, turn_type, active_player, HIGHLIGHT, screen)
            return True, move
    else:
        # turn_type == "remove run", move is list of 5 markers instead of coord
        move = get_remove_run_from_mouse(valid_moves)
        if move:
            draw_move(move, turn_type, active_player, HIGHLIGHT, screen)
            return True, move
    return False, move


def draw_valid_moves(valid_moves, turn_type, active_player, screen):
    for move in valid_moves:
        draw_move(move, turn_type, active_player, VALID, screen)


def draw_pieces_on_board(board, screen):
    for coord, element in board.elements.items():
        piece = Piece(coord, element[1], element[0])
        piece.draw(screen)


# -- | Get the exact board coordinate which is closest to the given screen coordinate.
# closestCoord :: ScreenCoord -> YCoord
def closest_coord(xi, yi):
    def dist(x, y):
        return sqrt((x - xi) ** 2 + (y - yi) ** 2)

    distances = [dist(*screen_point(*point)) for point in coords]
    index_min = min(range(len(distances)), key=distances.__getitem__)
    return coords[index_min]


def draw_background(screen):
    screen.fill((200, 200, 200))
    pygame.draw.circle(screen, white, pgvec((0, 0)), spacing * 5)


def draw_grid(screen):
    for coord in coords:
        pygame.draw.circle(screen, black, coord, 1)
        for coord2 in coords:
            if adjacent(coord, coord2):
                pygame.draw.aaline(screen, "black", pgvec(coord), pgvec(coord2))


def get_shadow_text(text, color, font1, font2):
    t = font1.render(text, True, color)
    ts = font2.render(text, True, black)
    return t, ts


def display_shadowed_text(text, color, font1, font2, position, screen):
    text_render, shadow_render = get_shadow_text(text, color, font1, font2)
    for i in range(-1, 1):
        screen.blit(shadow_render, (position[0] - i, position[1] - i))
    screen.blit(text_render, position)


def draw_end_game_text(points, display_text):
    winner = get_winner(points)
    win_color = [p1_color, p2_color, white][winner - 1]
    display_text(f"Game over,", win_color, (5, 5))
    if winner != 3:
        display_text(f"Player {winner} wins!", win_color, (5, 40))
    else:
        display_text(f"It's a draw!", win_color, (5, 40))


def draw_turn_text(active_player, turn_type, display_text):
    display_text(
        f"Player {active_player+1}'s turn",
        [p1_color, p2_color][active_player],
        (5, 5),
    )

    display_text(
        f"{turn_type}",
        [p1_color, p2_color][active_player],
        (5, 40),
    )


def draw_game_text(
    screen,
    turn_type,
    active_player,
    points,
    terminal,
):
    # define how to display the text
    font1 = pygame.font.SysFont("arial.ttf", 42)
    font2 = pygame.font.SysFont("arial.ttf", 42)

    def display_text(text, color, position):
        display_shadowed_text(text, color, font1, font2, position, screen)

    if terminal:
        draw_end_game_text(points, display_text)
    else:
        draw_turn_text(active_player, turn_type, display_text)

    # draw points
    display_text(f"p1 points: {points[0]}", p1_color, (int(600 * res * 4 / 5) - 40, 5))
    display_text(f"p2 points: {points[1]}", p2_color, (int(600 * res * 4 / 5) - 40, 40))


def print_game_state_initialization(game_state: GameState):
    print("game_state = GameState(")
    for field in fields(GameState):
        value = getattr(game_state, field.name)
        print(f"    {field.name} = {repr(value)}")
    print(")")


def play_game(player1, player2, delay: int = 1):
    # pygame setup
    pygame.init()
    screen = pygame.display.set_mode((600 * res, 630 * res))
    running = True

    # Set the pygame icon to be an empty Yinsh board
    draw_background(screen)
    draw_grid(screen)
    pygame.display.set_icon(screen)

    yinsh_game = YinshGame()
    players = [player1, player2]

    already_highlighted = False
    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        # check for mouse clicks
        clicked = False
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                clicked = True
            if event.type == pygame.QUIT:
                running = False

        draw_background(screen)

        # get the game state
        gs = yinsh_game.get_game_state()
        player = players[gs.active_player]
        draw_game_text(screen, gs.turn_type, gs.active_player, gs.points, gs.terminal)

        # draw valid moves (under board grid)
        if not gs.terminal:
            draw_valid_moves(gs.valid_moves, gs.turn_type, gs.active_player, screen)
        # draw board grid
        draw_grid(screen)
        # draw all the pieces ACTUALLY on the board
        draw_pieces_on_board(gs.board, screen)

        # highlight and make selected move if applicable
        if not gs.terminal:
            if players[gs.active_player] is None:
                # highlight the move that the mouse is hovering over if it's a valid
                # make that move if it's also clicked
                valid, move = highlight_human_move(
                    gs.valid_moves, screen, gs.turn_type, gs.active_player
                )
                if valid and clicked:
                    yinsh_game.take_turn(move)
                pygame.display.flip()
            else:
                pygame.display.flip()
                if not already_highlighted:
                    time.sleep(1 * delay / 3)
                    move = player.make_move(gs)
                    draw_move(move, gs.turn_type, gs.active_player, HIGHLIGHT, screen)
                    already_highlighted = True
                    pygame.display.flip()
                    time.sleep(2 * delay / 3)
                else:
                    draw_move(move, gs.turn_type, gs.active_player, HIGHLIGHT, screen)
                    yinsh_game.take_turn(move)
                    already_highlighted = False
        pygame.display.flip()
    print_game_state_initialization(gs)
    pygame.quit()
    return get_winner(gs.points)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("player1", type=str, default="human")
    parser.add_argument("player2", type=str, default="human")
    parser.add_argument("delay", type=float, default=1)
    args = parser.parse_args()

    p1 = args.player1
    p2 = args.player2
    try:
        player_data = load_data()
        print(player_data)
    except:
        player_data = {}
    player_data = new_bot_data_entry(p1, player_data, overwrite=False)
    player_data = new_bot_data_entry(p2, player_data, overwrite=False)

    bot1 = get_bot(p1, 0)
    bot2 = get_bot(p2, 1)

    elo1 = player_data[p1]["full_elo"]
    elo2 = player_data[p2]["full_elo"]
    print(f"{p1}: elo = {elo1}")
    print(f"{p2}: elo = {elo2}")
    winner = play_game(
        bot1,
        bot2,
        args.delay,
        # elo1, # not implemented drawn elo yet.
        # elo2,
    )
    update_score_information(p1, p2, winner, player_data)


if __name__ == "__main__":
    main()
