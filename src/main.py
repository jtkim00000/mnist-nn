import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets import Button
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import Normalize


# ==================================================
#                      PARAMS
# ==================================================

COLORMAP = "inferno"

NN_GRID_WIDTH = 200
NN_GRID_HEIGHT = 120

DRAW_GRID_SIZE = 28

# ==================================================
#                  INITIALIZATION
# ==================================================

X = np.zeros(784)
A1 = np.zeros(16)
A2 = np.zeros(16)
A3 = np.zeros(10)

W1 = np.random.rand(16, 784) * 0.005
B1 = np.random.rand(16,  1)

W2 = np.random.rand(16, 16) * 0.005
B2 = np.random.rand(16, 1)

W3 = np.random.rand(10, 16) * 0.005
B3 = np.random.rand(10, 1)

max_weight = 0

max_weight1 = np.max(W1) * 200
max_weight2 = np.max(W2) * 200
max_weight3 = np.max(W3) * 200

if(max_weight1 > max_weight2):
    if(max_weight1 > max_weight3):
        max_weight = max_weight1
    else:
        max_weight = max_weight3
else:
    if(max_weight2 > max_weight3):
        max_weight = max_weight2
    else:
        max_weight = max_weight3

circles_to_draw = []

#lines
lines_to_draw = []
layer_positions = []

nn_grid = np.full((NN_GRID_HEIGHT, NN_GRID_WIDTH), 0.0, dtype=float)
draw_grid = np.zeros((DRAW_GRID_SIZE, DRAW_GRID_SIZE))

last_draw_cell = None

# ==================================================
#                     FUNCTIONS
# ==================================================

# From Forward Propagation

def col_operation(A, W, B):
    X = (W @ A) + B

    Y = 1 / (1 + np.exp(-X))

    return Y

def forward_prop(X, W1, B1, W2, B2, W3, B3):
    A1 = col_operation(X, W1, B1)
    A2 = col_operation(A1, W2, B2)
    A3 = col_operation(A2, W3, B3)

    return A1, A2, A3

# From Neural Network Visualization
def add_circle_with_glow(grid_array, center_x, center_y, radius, activation, glow_intensity=2.5):
    y_coords, x_coords = np.ogrid[:NN_GRID_HEIGHT, :NN_GRID_WIDTH]
    dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
    core_mask = np.where(dist <= radius, activation, 0.0)
    dist_outside = np.maximum(dist - radius, 0)
    glow_fade = np.exp(-dist_outside / glow_intensity)
    glow_mask = np.where(dist > radius, activation * glow_fade, 0.0)
    combined_mask = core_mask + glow_mask
    return np.maximum(grid_array, combined_mask)

def add_line(grid_array, p0, p1, weight):

    x0, y0 = p0
    x1, y1 = p1

    strength = abs(weight) / max_weight

    line = Line2D(
        [x0, x1],
        [y0, y1],
        linewidth=0.3,
        color="#d458ff",   # single purple color
        alpha= 0.8 * strength,
    )

    ax1.add_line(line)

def render_shapes(nn_grid):
    for circle in circles_to_draw:
        nn_grid = add_circle_with_glow(
            nn_grid,
            center_x=circle["x"],
            center_y=circle["y"],
            radius=circle["radius"],
            activation=circle["activation"],
            glow_intensity=1.0 # Set default glow softness
        )

    return nn_grid

def render_connections():
    for line in lines_to_draw:
        add_line(
            None,
            p0=line["p0"],
            p1=line["p1"],
            weight=line["weight"],
        )

def init_neurons(N0, N1, N2, N3):
    border = 5

    max_grid = NN_GRID_HEIGHT - border * 2

    layer0 = []
    for i in range(len(N0)):
        y = border + (i * (max_grid / (1 + len(N0))) + max_grid / (1 + len(N0)))
        if((i % 49 == 0) or (i == (len(N0) - 1))):
            layer0.append((20, y))
        circles_to_draw.append(
            {
                "x": 20, 
                "y": y, 
                "radius": (max_grid / (6 * len(N0))), 
                "activation": (N0[i] + 0.2)
            }
        )

    layer1 = []
    for i in range(len(N1)):
        y = border + (i * (max_grid / (1 + len(N1))) + max_grid / (1 + len(N1)))
        
        layer1.append((70, y))
        circles_to_draw.append(
            {
                "x": 70, 
                "y": y, 
                "radius": (max_grid / (6 * len(N1))), 
                "activation": (N1[i] + 0.2)
            }
        )

    layer2 = []
    for i in range(len(N2)):
        y = border + (i * (max_grid / (1 + len(N2))) + max_grid / (1 + len(N2)))
        layer2.append((120, y))
        circles_to_draw.append(
            {
                "x": 120, 
                "y": y, 
                "radius": (max_grid / (6 * len(N2))), 
                "activation": (N2[i] + 0.2)
            }
        )
    
    layer3 = []
    for i in range(len(N3)):
        y = border + 5 + (i * ((max_grid - 10) / (1 + len(N3))) + (max_grid - 10) / (1 + len(N3)))

        layer3.append((170, y))
        circles_to_draw.append(
            {
                "x": 170, 
                "y": y, 
                "radius": (max_grid / (6 * len(N2))), 
                "activation": (N3[i] + 0.2)
            }
        )

    layer_positions.append(layer0)
    layer_positions.append(layer1)
    layer_positions.append(layer2)
    layer_positions.append(layer3)

def create_connections():
    global lines_to_draw, W1, W2, W3

    lines_to_draw.clear()

    for layer in range(len(layer_positions)-1):

        current_layer = layer_positions[layer]
        next_layer = layer_positions[layer+1]

        for i, neuron1 in enumerate(current_layer):
            for j, neuron2 in enumerate(next_layer):

                # weight_strength = min(abs(W[neuron2][neuron1])*5,1)
                if(layer == 0):
                    lines_to_draw.append(
                        {
                            "p0": neuron1,
                            "p1": neuron2,
                            "weight": (200 * W1[j][i])
                        }
                    )
                elif(layer == 1):
                    lines_to_draw.append(
                        {
                            "p0": neuron1,
                            "p1": neuron2,
                            "weight": (200 * W2[j][i])
                        }
                    )
                else:
                    lines_to_draw.append(
                        {
                            "p0": neuron1,
                            "p1": neuron2,
                            "weight": (200 * W3[j][i])
                        }
                    )

# From Num Draw Grid
def process(event):
    global last_cell

    if event.inaxes != ax2 or event.xdata is None or event.ydata is None:
        return

    col = int(event.xdata)
    row = int(event.ydata)

    if not (0 <= row < DRAW_GRID_SIZE and 0 <= col < DRAW_GRID_SIZE):
        return

    cell = (row, col)

    # Only trigger when entering a new square
    if cell != last_cell:
        # Determine the points to draw (interpolate if dragging, otherwise just the clicked cell)
        if last_cell is None:
            points_to_draw = [cell]
        else:
            r0, c0 = last_cell
            points_to_draw = get_line(r0, c0, row, col)

        last_cell = cell

        brush = {
            1.0: [(0, 0)],
            0.8: [(-1, 0), (1, 0), (0, -1), (0, 1)],
            0.6: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
            0.3: [
                (-2, -1), (-2, 0), (-2, 1),
                ( 2, -1), ( 2, 0), ( 2, 1),
                (-1, -2), ( 0, -2), ( 1, -2),
                (-1,  2), ( 0,  2), ( 1,  2),
            ],
        }

        # Apply the brush to every point in the interpolated line
        for r_draw, c_draw in points_to_draw:
            for value, offsets in brush.items():
                for dr, dc in offsets:
                    r = r_draw + dr
                    c = c_draw + dc

                    if 0 <= r < DRAW_GRID_SIZE and 0 <= c < DRAW_GRID_SIZE:
                        draw_grid[r, c] = max(draw_grid[r, c], value)

        img2.set_data(draw_grid)
        fig.canvas.draw_idle()

def get_line(r0, c0, r1, c1):
    """Bresenham's Line Algorithm"""
    points = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = 1 if r0 < r1 else -1, 1 if c0 < c1 else -1
    err = dr - dc
    while True:
        points.append((r0, c0))
        if r0 == r1 and c0 == c1: break
        e2 = 2 * err
        if e2 > -dc: err -= dc; r0 += sr
        if e2 < dr: err += dr; c0 += sc
    return points

def on_press(event):
    global mouse_down
    mouse_down = True
    process(event)

def on_release(event):
    global mouse_down, last_cell
    mouse_down = False
    last_cell = None

def on_move(event):
    if mouse_down:
        process(event)

def reset(event):
    global draw_grid

    draw_grid[:] = 0          # Clear the existing array
    img2.set_data(draw_grid)   # Update the displayed image
    fig.canvas.draw_idle()

def submit(event):
    global draw_grid

    inference()

    draw_grid[:] = 0          # Clear the existing array
    img2.set_data(draw_grid)   # Update the displayed image
    fig.canvas.draw_idle()

# New

def inference():
    global X, A1, A2, A3, nn_grid

    X = draw_grid.flatten().reshape(784,1)

    A1, A2, A3 = forward_prop(X, W1, B1, W2, B2, W3, B3)

    circles_to_draw.clear()

    layer_positions.clear()

    init_neurons(X, A1, A2, A3)

    nn_grid[:] = 0
    nn_grid = render_shapes(nn_grid)

    img1.set_data(np.ma.masked_where(nn_grid == 0, nn_grid))
    fig.canvas.draw_idle()


# ==================================================
#                     PLOTTING
# ==================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), gridspec_kw={'width_ratios': [4, 1]})
fig.patch.set_facecolor('#030212')
fig.subplots_adjust(
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    wspace=0.0
)

for spine in ax2.spines.values():
    spine.set_edgecolor('#260B50')

ax2.set_xticks(np.arange(0, DRAW_GRID_SIZE + 1, 1), minor=True)
ax2.set_yticks(np.arange(0, DRAW_GRID_SIZE + 1, 1), minor=True)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlim(0, DRAW_GRID_SIZE)
ax2.set_ylim(DRAW_GRID_SIZE, 0)

mouse_down = False
last_cell = None

# Create button area
button_ax1 = plt.axes([0.85, 0.1, 0.1, 0.05])  # x, y, width, height
reset_button = Button(
    button_ax1,
    "Reset",
    color="#260B50",
    hovercolor="#32095D"
)
reset_button.label.set_color("white")

# Create button area
button_ax2 = plt.axes([0.85, 0.2, 0.1, 0.05])  # x, y, width, height
submit_button = Button(
    button_ax2,
    "Submit",
    color="#260B50",
    hovercolor="#32095D"
)
submit_button.label.set_color("white")

# Connect button click
reset_button.on_clicked(reset)
submit_button.on_clicked(submit)


fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_move)

# ==================================================
#                     MAIN LOOP
# ==================================================

init_neurons(np.zeros(784), A1, A2, A3)

nn_grid = render_shapes(nn_grid)

create_connections()

img1 = ax1.imshow(
    np.ma.masked_where(nn_grid == 0, nn_grid),
    cmap=COLORMAP,
    vmin=-0.05,
    vmax=1.4,
    origin="lower",
    extent=(0, NN_GRID_WIDTH, 0, NN_GRID_HEIGHT),
    interpolation="bicubic",
)

render_connections()

img2 = ax2.imshow(
    draw_grid,
    cmap=COLORMAP,
    vmin=-0.05,
    vmax=1.4,
    origin="upper",
    extent=(0, DRAW_GRID_SIZE, DRAW_GRID_SIZE, 0),
    interpolation="bicubic",
)


ax1.set_xticks([])
ax1.set_yticks([])
ax1.axis("off")
plt.show()







