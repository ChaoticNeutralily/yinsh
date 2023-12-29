# Example file showing a circle moving on screen

import argparse
from math import sqrt
import time

import pygame

# from utils import *
from pieces import Piece, screen_point
from utils import *
from yinsh import *
from random_bot import UniformRandomPlayer


def pgvec(coord):
    return pygame.Vector2((screen_point(*coord)))


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


def make_game_text(
    screen,
    turn_type,
    active_player,
    board,
    points,
    valid_moves,
    prev_move,
    terminal,
):
    # print text above game state
    font1 = pygame.font.SysFont("chalkduster.ttf", 42)
    font2 = pygame.font.SysFont("chalkduster.ttf", 42)
    if not terminal:
        player_turn_text = font1.render(
            f"Player {active_player+1}'s turn",
            True,
            [p1_color, p2_color][active_player],
        )
        player_turn_text2 = font2.render(
            f"Player {active_player+1}'s turn", True, black
        )
        turn_type_text = font1.render(
            f"{turn_type}", True, [p1_color, p2_color][active_player]
        )
        turn_type_text2 = font2.render(f"{turn_type}", True, black)
    else:
        if points[0] == max(points) and points[1] != points[0]:
            winner = 1
        elif points[1] != points[0]:
            winner = 2
        else:
            winner = "draw"
        player_turn_text2 = font2.render(f"Game over,", True, black)
        if type(winner) == int:
            player_turn_text = font1.render(
                f"Game over,", True, [p1_color, p2_color][winner - 1]
            )

            turn_type_text = font1.render(
                f"Player {winner} wins!", True, [p1_color, p2_color][winner - 1]
            )
            turn_type_text2 = font2.render(f"Player {winner} wins!", True, black)
        else:
            player_turn_text = font1.render(f"Game over,", True, white)
            turn_type_text = font1.render(f"It's a draw!", True, white)
            turn_type_text2 = font2.render(f"It's a draw!", True, black)

    p1_points_text = font1.render(f"p1 points: {points[0]}", True, p1_color)
    p1_points_text2 = font2.render(f"p1 points: {points[0]}", True, black)
    p2_points_text = font1.render(f"p2 points: {points[1]}", True, p2_color)
    p2_points_text2 = font1.render(f"p2 points: {points[1]}", True, black)

    # display text shadow
    for i in range(-1, 1):
        screen.blit(player_turn_text2, (5 - i, 5 - i))
        screen.blit(turn_type_text2, (5 - i, 40 - i))
        screen.blit(p1_points_text2, (int(600 * res * 4 / 5) - 40 - i, 5 - i))
        screen.blit(p2_points_text2, (int(600 * res * 4 / 5) - 40 - i, 40 - i))
    # display text
    screen.blit(player_turn_text, (5, 5))
    screen.blit(turn_type_text, (5, 40))
    screen.blit(p1_points_text, (int(600 * res * 4 / 5) - 40, 5))
    screen.blit(p2_points_text, (int(600 * res * 4 / 5) - 40, 40))


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

    # try:
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
        (
            turn_type,
            active_player,
            board,
            points,
            valid_moves,
            prev_move,
            terminal,
        ) = yinsh_game.get_game_state()
        player = players[active_player]
        make_game_text(
            screen,
            turn_type,
            active_player,
            board,
            points,
            valid_moves,
            prev_move,
            terminal,
        )

        if not terminal:
            # draw valid moves
            # depends on turn type
            """
            setup new rings,  # add a ring
            add marker,  # add a marker in ring
            move ring,  # move ring at given position
            remove run # remove one of a platers run(s), param is last player that moved a ring
            remove_ring(player),  # remove one ring
            """
            if (
                turn_type == "setup new rings"
                or turn_type == "move ring"
                or turn_type == "remove ring"
            ):
                for move in valid_moves:
                    valid = Piece(move, active_player, "ring", 2)
                    valid.draw(screen)
            elif turn_type == "add marker":
                for move in valid_moves:
                    valid = Piece(move, active_player, "marker", 2)
                    valid.draw(screen)
            elif turn_type == "remove run":
                for run in valid_moves:
                    for marker in run:
                        valid = Piece(marker, active_player, "marker", 2)
                        valid.draw(screen)

        # draw board lines
        draw_grid(screen)

        # draw all the pieces ACTUALLY on the board
        for coord, element in board.elements.items():
            piece = Piece(coord, element[1], element[0])
            piece.draw(screen)

        if not terminal:
            if players[active_player] is None:
                # highlight the move that the mouse is hovering over if it's a valid
                # make that move if it's also clicked
                cc = closest_coord(*pygame.mouse.get_pos())
                if (
                    turn_type == "setup new rings"
                    or turn_type == "move ring"
                    or turn_type == "remove ring"
                ):
                    if cc in valid_moves:
                        highlight = Piece(cc, active_player, "ring", 1)
                        highlight.draw(screen)
                        if clicked:
                            yinsh_game.take_turn(cc)
                            yinsh_game.setup_next_turn_and_player()
                elif turn_type == "add marker":
                    if cc in valid_moves:
                        highlight = Piece(cc, active_player, "marker", 1)
                        highlight.draw(screen)
                        if clicked:
                            yinsh_game.take_turn(cc)
                            yinsh_game.setup_next_turn_and_player()
                elif turn_type == "remove run":
                    cc_in_runs = []
                    for run in valid_moves:
                        if cc in run:
                            cc_in_runs.append(run)
                    if len(cc_in_runs) > 0:
                        if len(cc_in_runs) == 1:
                            closest_run = cc_in_runs[0]
                        elif len(cc_in_runs) > 1:
                            # tie breaker
                            pos = pygame.mouse.get_pos()
                            run_dists = []
                            for run in cc_in_runs:
                                curr_dists2 = 0
                                for marker in run:
                                    pt = screen_point(*marker)
                                    curr_dists2 += (pt[0] - pos[0]) ** 2 + (
                                        pt[1] - pos[1]
                                    ) ** 2
                                run_dists.append(curr_dists2)
                            index_min = min(
                                range(len(run_dists)), key=run_dists.__getitem__
                            )
                            closest_run = cc_in_runs[index_min]
                        for marker in closest_run:
                            highlight = Piece(marker, active_player, "marker", 1)
                            highlight.draw(screen)
                        if clicked:
                            yinsh_game.take_turn(closest_run)
                            yinsh_game.setup_next_turn_and_player()
                pygame.display.flip()
            else:
                time.sleep(delay)
                yinsh_game.take_turn(players[active_player].make_move(valid_moves))
                yinsh_game.setup_next_turn_and_player()
        # flip() the display to put your work on screen
        pygame.display.flip()
    # except OSError as e:
    #     print(e)
    #     print(f'turn_type = "{turn_type}"')
    #     print(f"active_player = {active_player}")
    #     print(f"board.elements = {board.elements}")
    #     print(f"board.markers = {board.markers}")
    #     print(f"board.rings = {board.rings}")
    #     print(f"points = {points}")
    #     print(f"valid_moves = {valid_moves}")
    #     print(f"prev_move = {prev_move}")
    #     pygame.quit()
    print(f'turn_type = "{turn_type}"')
    print(f"active_player = {active_player}")
    print(f"board.elements = {board.elements}")
    print(f"board.markers = {board.markers}")
    print(f"board.rings = {board.rings}")
    print(f"points = {points}")
    print(f"valid_moves = {valid_moves}")
    print(f"prev_move = {prev_move}")
    pygame.quit()


def get_bot(kind_of_player: str):
    if kind_of_player == "random":
        return UniformRandomPlayer()
    return None  # "human", or any not implemented bot


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("player1", type=str, default="human")
    parser.add_argument("player2", type=str, default="human")
    parser.add_argument("delay", type=float, default=1)
    args = parser.parse_args()

    bot1 = get_bot(args.player1)
    bot2 = get_bot(args.player2)
    play_game(bot1, bot2, args.delay)


if __name__ == "__main__":
    main()
