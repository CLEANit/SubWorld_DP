import numpy as np
from copy import deepcopy

# Class to generate and keep track of the parameters to generate an island
class islandParameter(object):
    def __init__(self, size, x_decay_min, x_decay_max, y_decay_min, y_decay_max, min_h, max_h):
        self.height = np.random.uniform(low=min_h, high=max_h) # Height parameter
        self.x0 = np.random.uniform(low=0.0, high=size) # x position parameter
        self.y0 = np.random.uniform(low=0.0, high=size) # y position parameter
        self.a = np.random.uniform(low=x_decay_min, high=x_decay_max) # x variance parameter
        self.c = np.random.uniform(low=y_decay_min, high=y_decay_max) # y variance parameter
        b_lim = np.sqrt(self.a*self.c) # Maximum absolute value for co-variance parameter
        self.b = np.random.uniform(low=-1.0*b_lim, high=1.0*b_lim) # Co-variance parameter

# Function to get the height of the land function at specific coordinates
def get_height(x_pos, y_pos, island_list, size):
    n = len(island_list)
    height = -1.0
    dx_height = np.zeros(n, dtype=np.float32)
    dy_height = np.zeros(n, dtype=np.float32)
    # Loop over each island in the land function
    for i in range(n):
        # Loop over a 3x3 grid for PBCs
        for j in range(3):
            for k in range(3):
                A = island_list[i].height
                x0 = island_list[i].x0
                y0 = island_list[i].y0
                a = island_list[i].a
                b = island_list[i].b
                c = island_list[i].c

                # Generate an island as a 2D Gaussian
                height += A * np.exp(
                    -1.0 * (
                        a * ( (x_pos + size * (j-1)) - x0) ** 2.0
                        + 2.0 * b * ((x_pos + size * (j-1))  - x0) * ( (y_pos + size * (k-1)) - y0)
                        + c * ( (y_pos + size * (k-1)) - y0) ** 2
                    )
                )

                # Get the x and y partial derivatives of an island
                dx_height[i] += A * -1.0 * (2.0 * a * (x_pos + size * (j-1) - x0) + b * (y_pos + size * (k-1) - y0)) * np.exp(-1.0 * (a * ( (x_pos + size * (j-1)) - x0) ** 2.0 + 2.0 * b * ((x_pos + size * (j-1))  - x0) * ((y_pos + size * (k-1)) - y0) + c * ((y_pos + size * (k-1)) - y0) ** 2))
                dy_height[i] += A * -1.0 * (2.0 * c * (y_pos + size * (k-1) - y0) + b * (x_pos + size * (j-1) - x0)) * np.exp(-1.0 * (a * ( (x_pos + size * (j-1)) - x0) ** 2.0 + 2.0 * b * ((x_pos + size * (j-1))  - x0) * ((y_pos + size * (k-1)) - y0) + c * ((y_pos + size * (k-1)) - y0) ** 2))

    return height, dx_height, dy_height

# Function to generate a chart
def gen_chart(dim, islands, size, max_cur):
    n_islands = len(islands)
    chart = np.zeros((dim, dim), dtype=np.float32)
    water_c = np.zeros((dim, dim, 2))
    water_cs = np.zeros((n_islands, dim, dim, 2))
    # Loop over each pixel in the chart
    for i in range(dim):
        for j in range(dim):
            # Get the height and partial derivatives of the land function using get_height
            height, dx_height, dy_height = get_height((i+0.5)*size/dim, (j+0.5)*size/dim, islands, size)
            chart[i, j] += height

            # Rotate the partial derivative vector by 90 degrees
            for n in range(n_islands):
                water_cs[n, i, j, 0] -= dy_height[n]
                water_cs[n, i, j, 1] += dx_height[n]

    # Use the first islands partial derivative as an initial water current
    water_c += water_cs[0]
    # Loop over each remaining island
    for n in range(1, n_islands):
        # Loop over each pixel in the chart
        for i in range(dim):
            for j in range(dim):
                # Get magnitude of adding and subtracting islands rotated
                # partial derivate vector to existing water current 
                r1 = np.sqrt((water_c[i, j, 0] + water_cs[n, i, j, 0])**2 + (water_c[i, j, 1] + water_cs[n, i, j, 1])**2)
                r2 = np.sqrt((water_c[i, j, 0] - water_cs[n, i, j, 0])**2 + (water_c[i, j, 1] - water_cs[n, i, j, 1])**2)
                # Select sign that maximizes water current
                if r1 > r2:
                    water_c[i, j] += water_cs[n, i, j]
                else:
                    water_c[i, j] -= water_cs[n, i, j]

    # Get maximum directional vector component of the water current
    cur_max = np.max([np.abs(np.max(water_c)), np.abs(np.min(water_c))])
    # Loop over each pixel in the chart
    for i in range(dim):
        for j in range(dim):
            # Recale and reverse water current magnitude
            water_c[i, j] = water_c[i, j] * max_cur * (cur_max - np.sqrt((water_c[i, j, 0])**2 + (water_c[i, j, 1])**2)) / (np.sqrt((water_c[i, j, 0])**2 + (water_c[i, j, 1])**2) * cur_max)

    # Create a copy of the existing water current
    water_c_old = deepcopy(water_c)
    # Loop over each pixel in the chart
    for i in range(dim):
        for j in range(dim):
            avg = 0.0
            # Loop over a 5x5 grid of nearest neighboors
            for k in range(5):
                for l in range(5):
                    avg += water_c_old[(i + k - 1) % dim, (j + l - 1) % dim]

            # Average over 25 nearest neighboors (including current pixel)
            water_c[i, j] = avg / 25
            if chart[i, j] > -0.1:
                water_c[i, j] = 0.0

    return chart, water_c

# Function to generate and save a chart using gen_chart
def save_chart(seed, n_islands, size, x_decay_min, x_decay_max, y_decay_min, y_decay_max, min_height, max_height, dim, max_cur, path):
    # Set NumPy random seed
    np.random.seed(seed)
    islands = []
    # Generate and store parameters for each island
    for i in range(n_islands):
        islands.append(islandParameter(size, x_decay_min, x_decay_max, y_decay_min, y_decay_max, min_height, max_height))

    # Generate a chart
    chart, water_c = gen_chart(dim, islands, size, max_cur)

    # Save previously generated chart
    np.savez(path + '/data/charts/charts_' + str(seed) + '.npz', chart=chart, water_c=water_c)
