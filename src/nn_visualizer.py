import matplotlib.pyplot as plt
import numpy as np

GRID_SIZE = 120

GRID_WIDTH = 200
GRID_HEIGHT = 120

BACKGROUND_ACTIVATION = 0.0
COLORMAP = "inferno"

circles_to_draw = []
lines_to_draw = []


grid = np.full((GRID_HEIGHT, GRID_WIDTH), BACKGROUND_ACTIVATION, dtype=float)

def add_circle_with_glow(grid_array, center_x, center_y, radius, activation, glow_intensity=2.5):
    y_coords, x_coords = np.ogrid[:GRID_HEIGHT, :GRID_WIDTH]
    dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    core_mask = np.where(dist <= radius, activation, 0.0)
    dist_outside = np.maximum(dist - radius, 0)
    glow_fade = np.exp(-dist_outside / glow_intensity)
    glow_mask = np.where(dist > radius, activation * glow_fade, 0.0)
    combined_mask = core_mask + glow_mask
    return np.maximum(grid_array, combined_mask)

def add_line_with_glow(grid_array, p0, p1, width, activation, glow_intensity=2.0):
    y_coords, x_coords = np.ogrid[:GRID_HEIGTH, :GRID_WIDTH]
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    line_len_sq = dx**2 + dy**2

    if line_len_sq == 0:
        dist = np.sqrt((x_coords - x0)**2 + (y_coords - y0)**2)
    else:
        t = ((x_coords - x0) * dx + (y_coords - y0) * dy) / line_len_sq
        t = np.clip(t, 0.0, 1)
        closest_x = x0 + t * dx
        closest_y = y0 + t * dy
        dist = np.sqrt((x_coords - closest_x)**2 + (y_coords - closest_y)**2)
    core_mask = np.where(dist <= width, activation, 0.0)

    dist_outside = np.maximum(dist - width, 0)
    glow_fade = np.exp(-dist_outside / glow_intensity)
    glow_mask = np.where(dist > width, activation * glow_fade, 0.0)

    combined_mask = core_mask + glow_mask
    return np.maximum(grid_array, combined_mask)

def render_shapes(grid):
    for line in lines_to_draw:
        grid = add_line_with_glow(
            grid,
            p0=line["p0"],
            p1=line["p1"],
            width=line["width"],
            activation=line["activation"],
            glow_intensity=1
        )

    for circle in circles_to_draw:
        grid = add_circle_with_glow(
            grid,
            center_x=circle["x"],
            center_y=circle["y"],
            radius=circle["radius"],
            activation=circle["activation"],
            glow_intensity=1 # Set default glow softness
        )

    return grid

# Only for 4 layer NNs
def init_neurons(N0, N1, N2, N3, border):

    max_grid = GRID_SIZE - border * 2

    for i in range(len(N0)):
        circles_to_draw.append(
            {
                "x": 40, 
                "y": border + (i * (max_grid / (1 + len(N0))) + max_grid / (1 + len(N0))), 
                "radius": (max_grid / (6 * len(N0))), 
                "activation": (N0[i] + 0.2)
            }
        )
    for i in range(len(N1)):
        circles_to_draw.append(
            {
                "x": 80, 
                "y": border + (i * (max_grid / (1 + len(N1))) + max_grid / (1 + len(N1))), 
                "radius": (max_grid / (6 * len(N1))), 
                "activation": (N1[i] + 0.2)
            }
        )
    for i in range(len(N2)):
        circles_to_draw.append(
            {
                "x": 120, 
                "y": border + (i * (max_grid / (1 + len(N2))) + max_grid / (1 + len(N2))), 
                "radius": (max_grid / (6 * len(N2))), 
                "activation": (N2[i] + 0.2)
            }
        )
    for i in range(len(N3)):
        circles_to_draw.append(
            {
                "x": 160, 
                "y": border + 5 + (i * ((max_grid - 10) / (1 + len(N3))) + (max_grid - 10) / (1 + len(N3))), 
                "radius": (max_grid / (6 * len(N2))), 
                "activation": (N3[i] + 0.2)
            }
        )

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('black') # Match the figure background to activation 0
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


N0 = np.random.rand(784)
N1 = np.random.rand(16)
N2 = np.random.rand(16)
N3 = np.random.rand(10)

init_neurons(N0, N1, N2, N3, 5)

grid = render_shapes(grid)

img = ax.imshow(
    grid,
    cmap=COLORMAP,
    vmin=-0.1,
    vmax=1.4,
    origin="lower",
    extent=(0, GRID_WIDTH, 0, GRID_HEIGHT),
    interpolation="bicubic",
)


ax.set_xticks([])
ax.set_yticks([])
ax.axis("off")
plt.show()