from typing import Tuple

Coordinate = Tuple[int]

# Color theme (http://www.colourlovers.com/palette/15/tech_light)
# RGB
p2_color = (247, 165, 121)  # orange # (209, 231, 81) # green
p1_color = (149, 103, 224)  # purple # (38, 173, 228) # blue
white = (255, 255, 255)
hl = (255, 0, 0, 0.5)
black = (0, 0, 0)

h1 = [(g + w) / 2 for g, w in zip(p1_color, white)]
h2 = [(b + w) / 2 for b, w in zip(p2_color, white)]

v1 = [(b + 3 * w) / 4 for b, w in zip(p1_color, white)]
v2 = [(g + 3 * w) / 4 for g, w in zip(p2_color, white)]


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
