import matplotlib.pyplot as plt
import numpy as np

from matplotlib.widgets import Button

GRID_SIZE = 28

fig, ax = plt.subplots(figsize=(10, 10))

grid = np.zeros((GRID_SIZE, GRID_SIZE))

last_draw_cell = None

img = ax.imshow(
    grid,
    cmap="inferno",
    vmin=-0.1,
    vmax=1.25,
    origin="upper",
    extent=(0, GRID_SIZE, GRID_SIZE, 0),
    interpolation="bicubic",
)

# Draw grid
ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1), minor=True)
ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1), minor=True)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(GRID_SIZE, 0)

mouse_down = False
last_cell = None


def process(event):
    global last_cell

    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    col = int(event.xdata)
    row = int(event.ydata)

    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
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

                    if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                        grid[r, c] = max(grid[r, c], value)

        img.set_data(grid)
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
    global grid

    grid[:] = 0          # Clear the existing array
    img.set_data(grid)   # Update the displayed image
    fig.canvas.draw_idle()

def submit(event):
    global grid
    output_grid()

    grid[:] = 0          # Clear the existing array
    img.set_data(grid)   # Update the displayed image
    fig.canvas.draw_idle()

def output_grid():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            print(grid[i][j], end=" ")
        print("\n")




# Create button area
button_ax1 = plt.axes([0.3, 0.02, 0.2, 0.05])  # x, y, width, height
reset_button = Button(button_ax1, "Reset")

# Create button area
button_ax2 = plt.axes([0.5, 0.02, 0.2, 0.05])  # x, y, width, height
submit_button = Button(button_ax2, "Submit")

# Connect button click
reset_button.on_clicked(reset)
submit_button.on_clicked(submit)


fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", on_move)

plt.show()