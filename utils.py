from typing import Tuple

Coordinate = Tuple[int]

# Color theme (http://www.colourlovers.com/palette/15/tech_light)
# RGB
green = (247, 165, 121)  # (209, 231, 81)
blue = (149, 103, 224)  # (38, 173, 228)
white = (255, 255, 255)
hl = (255, 0, 0, 0.5)
black = (0, 0, 0)

gh = [(g + w) / 2 for g, w in zip(green, white)]
bh = [(b + w) / 2 for b, w in zip(blue, white)]

gv = [(g + 3 * w) / 4 for g, w in zip(green, white)]
bv = [(b + 3 * w) / 4 for b, w in zip(blue, white)]


# Dimensions
res = 3 / 2
edge = int(1 * res)
marker_width = int(19 * res)
ring_inner_radius = int(22 * res)
ring_width = int(6 * res)
edge = int(1 * res)
spacing = int(60 * res)

origin_x = int((600 / 2) * res)  # Half the canvas size
origin_y = int((630 / 2) * res)
