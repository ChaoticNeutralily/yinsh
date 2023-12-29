from math import sqrt
from typing import Tuple

import pygame
from pygame import gfxdraw

from utils import *


def screen_point(ya: int, yb: int):
    """Translate hex coordinates to screen coordinates."""
    x = spacing * ya
    y = spacing * yb
    return (0.5 * sqrt(3) * x + origin_x, -y + 0.5 * x + origin_y)


class Piece(pygame.sprite.Sprite):
    def __init__(
        self, coord: Coordinate, player: int, ring_marker: str, highlight: int = 0
    ):
        super().__init__()
        self.coord = coord
        self.draw_coord = screen_point(*self.coord)
        self.player = player
        self.piece_type = ring_marker
        self.highlight = highlight

    def update_coord(self, new_coord):
        self.coord = new_coord
        self.draw_coord = screen_point(*self.coord)

    def get_color(self):
        return ([[p1_color, h1, v1], [p2_color, h2, v2]][self.player])[self.highlight]

    def draw(self, screen):
        if self.piece_type == "ring":
            pygame.draw.circle(
                screen,
                self.get_color(),
                self.draw_coord,
                ring_inner_radius + ring_width / 2,
                width=ring_width,
            )
            if self.highlight < 2:
                for i in range(-1, 1):
                    gfxdraw.aacircle(
                        screen,
                        int(self.draw_coord[0]),
                        int(self.draw_coord[1]),
                        int(ring_inner_radius + ring_width / 2) + i,
                        black,
                        # width=ring_width + 2 * edge,
                    )
                    gfxdraw.aacircle(
                        screen,
                        int(self.draw_coord[0]),
                        int(self.draw_coord[1]),
                        int(ring_inner_radius - ring_width / 2) + i,
                        black,
                        # width=ring_width + 2 * edge,
                    )
        elif self.piece_type == "marker":
            pygame.draw.circle(screen, self.get_color(), self.draw_coord, marker_width)
            if self.highlight < 2:
                for i in range(-1, 1):
                    gfxdraw.aacircle(
                        screen,
                        int(self.draw_coord[0]),
                        int(self.draw_coord[1]),
                        int(marker_width) + i,
                        black,
                    )
