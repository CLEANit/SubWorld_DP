import numpy as np
from copy import deepcopy

def gen_value(path, seed, dim, target_x, target_y, size, tol, n_h, n_t, discount):
    data = np.load(path + '/data/charts/charts_' + str(seed) + '.npz')
    chart = data['chart']
    water_c = np.zeros((dim, dim, 2), dtype=np.float32)

    try:
        target = np.array([target_x, target_y]) / dim
    except TypeError:
        place = False
        while not place:
            target = np.random.rand(2)
            if chart[int(target[0]*dim), int(target[1]*dim)] < -0.1:
                place = True

    chart_value = np.zeros((dim, dim), dtype=np.float32) + 1
    rel_chart_value = np.zeros((dim, dim), dtype=np.float32) + 1
    for i in range(dim):
        for j in range(dim):
            if chart[i, j] >= -0.1:
                chart_value[i, j] = 0
                rel_chart_value[i, j] = 0

            elif np.sqrt(((i + 0.5)/dim - target[0])**2 + ((j + 0.5)/dim - target[1])**2) < 0.3/size:
                chart_value[i, j] = 2
                rel_chart_value[i, j] = 2

    dif_chart = np.zeros((dim, dim), dtype=np.float32) + 100
    dif = 100

    count = 0

    while dif > tol:
        count += 1
        old_value = deepcopy(chart_value)
        old_rel_value = deepcopy(rel_chart_value)
        for i in range(dim):
            for j in range(dim):
                if old_value[i, j] < 2.0 - 1e-6 and old_value[i, j] > 1e-6 and dif_chart[i, j] > 1e-6:
                    values = np.zeros((n_h, n_t), dtype=np.float32)
                    rel_values = np.zeros((n_h, n_t), dtype=np.float32)
                    for k in range(n_h):
                        for l in range(n_t):
                            ind1 = 0
                            ind2 = 0
                            for m in range(n_t):
                                current = water_c[int(i + ind1) % dim, int(j + ind2) % dim]
                                ind1 += dim * ((l * np.cos(2 * np.pi * k / n_h) + n_t * current[0]) / (size * n_t ** 2))
                                ind2 += dim * ((l * np.sin(2 * np.pi * k / n_h) + n_t * current[1]) / (size * n_t ** 2))
                                values[k, l] = old_value[int(i + ind1) % dim, int(j + ind2) % dim] - 0.001 * l / n_t
                                rel_values[k, l] = old_rel_value[int(i + ind1) % dim, int(j + ind2) % dim] - 0.001 * l / n_t
                                if values[k, l] < 1e-6:
                                    break
                    
                    chart_value[i, j] = discount * np.max(values)
                    rel_chart_value[i, j] = rel_values[np.unravel_index(values.argmax(), values.shape)]

        dif_chart = abs(old_value - chart_value)
        dif = np.mean(dif_chart)
        print(dif, np.mean(abs(old_rel_value - rel_chart_value)))

    chart_value -= 1
    rel_chart_value -= 1

    return chart_value, rel_chart_value
