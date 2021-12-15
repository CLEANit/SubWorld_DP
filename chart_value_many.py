import numpy as np
from copy import deepcopy
import multiprocessing

n_cpu = 48
maps_i = 0
maps_f = 50000
size = 10.0
discount = 0.99

def gen_value(seed):
    data = np.load('./SubWorld_DP/data/charts/charts_' + str(seed) + '.npz')
    chart = data['chart']
    dim = chart.shape[0]
    water_c = np.zeros((dim, dim, 2), dtype=np.float32)
    steps = np.zeros((dim, dim), dtype=np.int32)

    target = np.random.rand(2)

    chart_value = np.zeros((dim, dim), dtype=np.float32) + 1
    for i in range(dim):
        for j in range(dim):
            if chart[i, j] >= -0.1:
                chart_value[i, j] = 0

            elif np.sqrt(((i + 0.5)/dim - target[0])**2 + ((j + 0.5)/dim - target[1])**2) < 0.3/size:
                chart_value[i, j] = 2

    tol = 1e-6
    dif_chart = np.zeros((dim, dim), dtype=np.float32) + 100
    dif = 100

    n_t = 5
    n_h = 16

    count = 0

    while dif > tol:
        count += 1
        old_value = deepcopy(chart_value)
        for i in range(dim):
            for j in range(dim):
                if old_value[i, j] < 2.0 - 1e-6 and old_value[i, j] > 1e-6 and dif_chart[i, j] > 1e-6:
                    values = np.zeros((n_h, n_t), dtype=np.float32)
                    for k in range(n_h):
                        for l in range(n_t):
                            ind1 = 0
                            ind2 = 0
                            for m in range(n_t):
                                current = water_c[int(i + ind1) % dim, int(j + ind2) % dim]
                                ind1 += dim * ((l * np.cos(2 * np.pi * k / n_h) + n_t * current[0]) / (size * n_t ** 2))
                                ind2 += dim * ((l * np.sin(2 * np.pi * k / n_h) + n_t * current[1]) / (size * n_t ** 2))
                                values[k, l] = old_value[int(i + ind1) % dim, int(j + ind2) % dim] - 0.01 * l / n_t
                                if values[k, l] < 1e-6:
                                    break
                    
                    chart_value[i, j] = discount * np.max(values)
                    steps[i, j] = count

        dif_chart = abs(old_value - chart_value)
        dif = np.mean(dif_chart)
        print(dif)

    chart_value -= 1

    np.savez('./SubWorld_DP/data/value/value_' + str(seed) + '.npz', value = chart_value, steps = steps, discount = discount)

def start():
    print('Starting', multiprocessing.current_process().name)

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes = n_cpu, initializer=start)
    seeds = list(np.arange(maps_i, maps_f, 1, dtype=np.int32))
    for _ in pool.imap_unordered(gen_value, seeds):
        pass
    pool.close()
    pool.join()
