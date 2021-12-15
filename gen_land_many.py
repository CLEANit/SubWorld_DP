import numpy as np
from copy import deepcopy
import multiprocessing

n_cpu = 48
maps_i = 0
maps_f = 50000

x_size = 10.0
y_size = 10.0
min_height = 0.9
max_height = 2.0
x_decay_min = 1.5
x_decay_max = 3.0
y_decay_min = 1.5
y_decay_max = 3.0
dim = 152
max_islands = 20
max_cur = 0.5

class islandParameter(object):
    def __init__(self, x_size, y_size, min_h, max_h):
        self.height = np.random.uniform(low=min_h, high=max_h) # Height parameter
        self.x0 = np.random.uniform(low=0.0, high=x_size) # x position parameter
        self.y0 = np.random.uniform(low=0.0, high=y_size) # y position parameter
        self.a = np.random.uniform(low=x_decay_min, high=x_decay_max) # x variance parameter
        self.c = np.random.uniform(low=y_decay_min, high=y_decay_max) # y variance parameter
        b_lim = np.sqrt(self.a*self.c) # Maximum absolute value for co-variance parameter
        self.b = np.random.uniform(low=-1.0*b_lim, high=1.0*b_lim) # Co-variance parameter

def get_height(x_pos, y_pos, island_list):
    n = len(island_list)
    height = -1.0
    dx_height = np.zeros(n, dtype=np.float32)
    dy_height = np.zeros(n, dtype=np.float32)
    for i in range(n):
        for j in range(3):
            for k in range(3):
                A = island_list[i].height
                x0 = island_list[i].x0
                y0 = island_list[i].y0
                a = island_list[i].a
                b = island_list[i].b
                c = island_list[i].c

                height += A * np.exp(
                    -1.0 * (
                        a * ( (x_pos + x_size * (j-1)) - x0) ** 2.0
                        + 2.0 * b * ((x_pos + x_size * (j-1))  - x0) * ( (y_pos + y_size * (k-1)) - y0)
                        + c * ( (y_pos + y_size * (k-1)) - y0) ** 2
                    )
                )

                dx_height[i] += A * -1.0 * (2.0 * a * (x_pos + x_size * (j-1) - x0) + b * (y_pos + y_size * (k-1) - y0)) * np.exp(-1.0 * (a * ( (x_pos + x_size * (j-1)) - x0) ** 2.0 + 2.0 * b * ((x_pos + x_size * (j-1))  - x0) * ((y_pos + y_size * (k-1)) - y0) + c * ((y_pos + y_size * (k-1)) - y0) ** 2))

                dy_height[i] += A * -1.0 * (2.0 * c * (y_pos + y_size * (k-1) - y0) + b * (x_pos + x_size * (j-1) - x0)) * np.exp(-1.0 * (a * ( (x_pos + x_size * (j-1)) - x0) ** 2.0 + 2.0 * b * ((x_pos + x_size * (j-1))  - x0) * ((y_pos + y_size * (k-1)) - y0) + c * ((y_pos + y_size * (k-1)) - y0) ** 2))

    return height, dx_height, dy_height

def gen_chart(seed, n_island):
    np.random.seed(seed)
    islands = []
    for i in range(n_island):
        islands.append(islandParameter(x_size, y_size, min_height, max_height))

    chart = np.zeros((dim, dim), dtype=np.float32)
    water_c = np.zeros((dim, dim, 2))
    water_cs = np.zeros((n_island, dim, dim, 2))
    for i in range(dim):
        for j in range(dim):
            height, dx_height, dy_height = get_height((i+0.5)*x_size/dim, (j+0.5)*y_size/dim, islands)
            chart[i, j] += height

            for n in range(n_island):
                water_cs[n, i, j, 0] -= dy_height[n]
                water_cs[n, i, j, 1] += dx_height[n]

    water_c += water_cs[0]
    for n in range(1, n_island):
        for i in range(dim):
            for j in range(dim):
                r1 = np.sqrt((water_c[i, j, 0] + water_cs[n, i, j, 0])**2 + (water_c[i, j, 1] + water_cs[n, i, j, 1])**2)
                r2 = np.sqrt((water_c[i, j, 0] - water_cs[n, i, j, 0])**2 + (water_c[i, j, 1] - water_cs[n, i, j, 1])**2)
                if r1 > r2:
                    water_c[i, j] += water_cs[n, i, j]
                else:
                    water_c[i, j] -= water_cs[n, i, j]

    cur_max = np.max([np.abs(np.max(water_c)), np.abs(np.min(water_c))])
    for i in range(dim):
        for j in range(dim):
            water_c[i, j] = water_c[i, j] * max_cur * (cur_max - np.sqrt((water_c[i, j, 0])**2 + (water_c[i, j, 1])**2)) / (np.sqrt((water_c[i, j, 0])**2 + (water_c[i, j, 1])**2) * cur_max)

    water_c_old = deepcopy(water_c)
    for i in range(dim):
        for j in range(dim):
            avg = 0.0
            for k in range(5):
                for l in range(5):
                    avg += water_c_old[(i + k - 1) % dim, (j + l - 1) % dim]

            water_c[i, j] = avg / 25
            if chart[i, j] > -0.1:
                water_c[i, j] = 0.0

    np.savez('./SubWorld_DP/data/charts/charts_' + str(seed) + '.npz', chart=chart, water_c=water_c)

def start():
    print('Starting', multiprocessing.current_process().name)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes = n_cpu, initializer=start)
    seeds = list(np.arange(maps_i, maps_f, 1, dtype=np.int32))
    n_islands = np.clip(np.random.normal(12, 5, maps_f - maps_i), 0, 20).astype(np.int32)
    pool.starmap(gen_chart, zip(seeds, n_islands))
    pool.close()
    pool.join()
