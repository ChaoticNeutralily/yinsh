# Example file showing a circle moving on screen
from math import sqrt

import pygame

# from utils import *
from pieces import *

coords = [
    (x, y)
    for x in range(-5, 6)
    for y in range(-5, 6)
    if (0.5 * sqrt(3) * x) ** 2 + (0.5 * x - y) ** 2 <= 4.6**2
]


def valid_coord(x, y):
    return (0.5 * sqrt(3) * x) ** 2 + (0.5 * x - y) ** 2 <= 4.6**2


def adjacent(xy, xy2):
    return (xy[0] - xy2[0]) ** 2 + (xy[1] - xy2[1]) ** 2 == 1 or (
        (xy[0] - xy2[0]) == 1 and (xy[1] - xy2[1]) == 1
    )


# -- Keyboard codes
# keyLeft = 37
# keyRight = 39


def pgvec(coord):
    return pygame.Vector2((screen_point(*coord)))


# -- | Get the board coordinate which is closest to the given screen coordinate.
# closestCoord :: ScreenCoord -> YCoord
def closest_coord(xi, yi):
    def dist(x, y):
        return sqrt((x - xi) ** 2 + (y - yi) ** 2)

    distances = [dist(*screen_point(*point)) for point in coords]
    index_min = min(range(len(distances)), key=distances.__getitem__)
    return coords[index_min]


# pygame setup
pygame.init()
screen = pygame.display.set_mode((600 * res, 630 * res))
clock = pygame.time.Clock()
running = True
dt = 0

last_pos = (0, 0)
pieces = []
current_piece = None
player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
turn = 0
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    clicked = (False, last_pos)
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONUP:
            clicked = (True, event.pos)
        if event.type == pygame.QUIT:
            running = False

    # Draw the board
    screen.fill("black")
    pygame.draw.circle(screen, white, pgvec((0, 0)), spacing * 5)

    for coord in coords:
        pygame.draw.circle(screen, black, pgvec(coord), 1)
        for coord2 in coords:
            if adjacent(coord, coord2):
                pygame.draw.line(screen, "black", pgvec(coord), pgvec(coord2))

    if clicked[0]:
        print(clicked[1])
        cc = closest_coord(*clicked[1])
        make_new_piece = current_piece is None
        move_piece = current_piece is not None
        for piece in pieces:
            if cc == piece.coord:
                make_new_piece = False
                move_piece = False
                # this coordinate has just been clicked
                if not piece.clicked and piece.player == turn % 2:
                    current_piece = piece
                    piece.clicked = True
                else:
                    piece.clicked = False
                    current_piece = None
        if make_new_piece:
            pieces.append(Ring(cc, player=turn % 2))
            turn += 1
        if move_piece:
            current_piece.update_coord(cc)
            current_piece.clicked = False
            current_piece = None
            turn += 1

    # Draw the pieces
    for piece in pieces:
        piece.draw(screen)

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()
