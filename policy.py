import numpy as np
from copy import deepcopy
from math import copysign

def sim_action(x, y, dim, n_h, n_t, action, current, size, chart_value, rel_chart_value, uncert_cur_x, uncert_cur_y, max_cur):
    ind1 = 0
    ind2 = 0
    k = action[0]
    l = action[1]
    for m in range(n_t):
        ind1 += dim * ((l * np.cos(2 * np.pi * k / n_h + np.pi/2) + n_t * np.clip(current[0] + m * uncert_cur_x, -1.0*max_cur, max_cur)) / (size * n_t ** 2))
        ind2 += dim * ((l * np.sin(2 * np.pi * k / n_h + np.pi/2) + n_t * np.clip(current[1] + m * uncert_cur_y, -1.0*max_cur, max_cur)) / (size * n_t ** 2))
        value = chart_value[int(x*dim - ind1) % dim, int(y*dim + ind2) % dim]
        rel_value = rel_chart_value[int(x*dim - ind1) % dim, int(y*dim + ind2) % dim]
        if value < -1 + 1e-6:
            return value, rel_value

    return value, rel_value

def est_value(n_h, n_t, unc_pos, unc_cur, pos_est, dim, current_e, size, chart_value, rel_chart_value, uncert_cur, max_cur):
    values = np.zeros((n_h, n_t), dtype=np.float32) - 1
    rel_values = np.zeros((n_h, n_t), dtype=np.float32) - 1
    uncert_pos_x = np.linspace(-1.0 * unc_pos, unc_pos, 3, endpoint = True)
    uncert_pos_y = np.linspace(-1.0 * unc_pos, unc_pos, 3, endpoint = True)
    uncert_cur_x = np.linspace(-1.0 * unc_cur, unc_cur, 3, endpoint = True)
    uncert_cur_y = np.linspace(-1.0 * unc_cur, unc_cur, 3, endpoint = True)
    for k in range(n_h):
        for l in range(n_t):
            value = np.zeros((uncert_pos_x.shape[0], uncert_pos_y.shape[0], uncert_cur_x.shape[0], uncert_cur_y.shape[0]), dtype=np.float32) - 1
            rel_value = np.zeros((uncert_pos_x.shape[0], uncert_pos_y.shape[0], uncert_cur_x.shape[0], uncert_cur_y.shape[0]), dtype=np.float32) - 1
            for unc_pos_x in range(uncert_pos_x.shape[0]):
                for unc_pos_y in range(uncert_pos_y.shape[0]):
                    for unc_cur_x in range(uncert_pos_x.shape[0]):
                        for unc_cur_y in range(uncert_pos_y.shape[0]):
                            value[unc_pos_x, unc_pos_y, unc_cur_x, unc_cur_y], rel_value[unc_pos_x, unc_pos_y, unc_cur_x, unc_cur_y] = sim_action(pos_est[0] + unc_pos_x, pos_est[1] + unc_pos_y, dim, n_h, n_t, np.array([k, l]), current_e - np.array([unc_cur_x, unc_cur_y]), size, chart_value, rel_chart_value, copysign(uncert_cur, unc_cur_x), copysign(uncert_cur, unc_cur_y), max_cur)

            values[k, l] = value.mean()
            rel_values[k, l] = rel_value.mean()

    return values, rel_values

def policy_gps(path, seed, sub_x, sub_y, n_steps, n_t, uncert_pos, n_h, size, gps_cost, cur_cost, uncert_cur, max_cur):
    data1 = np.load(path + '/data/value/value_' + str(seed) + '.npz')
    chart_value = data1['value']
    rel_chart_value = data1['rel_value']
    data = np.load(path + '/data/charts/charts_' + str(seed) + '.npz')
    chart = data['chart']
    dim = chart.shape[0]

    try:
        sub = np.array([sub_x, sub_y]) / dim
    except TypeError:
        place = False
        while not place:
            sub = np.random.rand(2)
            if chart_value[int(dim*sub[0]), int(dim*sub[1])] > -1 + 1e-6:
                place = True

    water = data['water_c']
    
    pos = np.zeros((n_steps*n_t+1, 2), dtype=np.float32)
    pos_est = np.zeros((n_steps+1, 2), dtype=np.float32)
    pos[0] = deepcopy(sub)
    current_e = water[int(sub[0]*dim), int(sub[1]*dim)]
    pos_est[0] = deepcopy(sub)
    unc_pos = 0.0
    unc_cur = 0.0
    no_gps = []
    no_cur = []
    last_gps = 0
    dead_rec = deepcopy(sub)
    action = (0.0, 0.0)
    done = False
    i = -1
    while not done:
        i += 1
        no_measure = True
        values, rel_values = est_value(n_h, n_t, unc_pos, unc_cur, pos_est[i], dim, current_e, size, chart_value, rel_chart_value, uncert_cur, max_cur)
        v_est = rel_values[np.unravel_index(values.argmax(), values.shape)]

        if (v_est < 1.0 - gps_cost and i != 0) or chart_value[int(pos_est[i, 0]*dim) % dim, int(pos_est[i, 1]*dim) % dim] > 1 - 1e-6:
            print('GPS')
            no_measure = False
            unc_pos = 0.0
            unc_cur = uncert_cur * (last_gps + 1)
            last_gps = 0
            current_e = sub - dead_rec
            pos_est[i] = deepcopy(sub)
            dead_rec = deepcopy(sub)
            if current_e[0] > 0.5:
                current_e[0] -= 1.0
            elif current_e[0] < -0.5:
                current_e[0] += 1.0
            if current_e[1] > 0.5:
                current_e[1] -= 1.0
            elif current_e[1] < -0.5:
                current_e[1] += 1.0

            current_e *= np.array([-1.0*size, size])
        
            values, rel_values = est_value(n_h, n_t, unc_pos, unc_cur, pos_est[i], dim, current_e, size, chart_value, rel_chart_value, uncert_cur, max_cur)
            v_est = rel_values[np.unravel_index(values.argmax(), values.shape)]
   
        if (v_est < 1.0 - cur_cost and i != 0):
            print('Current Profiler')
            no_measure = False
            current_e = water[int(sub[0]*dim), int(sub[1]*dim)]
            unc_cur = 0.0001

            values, rel_values = est_value(n_h, n_t, unc_pos, unc_cur, pos_est[i], dim, current_e, size, chart_value, rel_chart_value, uncert_cur, max_cur)

        if no_measure:
            print('No Measurement')
            last_gps += 1
            unc_pos += uncert_pos
            unc_cur += uncert_cur

        action = np.unravel_index(values.argmax(), values.shape)
        for m in range(n_t):
            current = water[int(sub[0]*dim), int(sub[1]*dim)]
            sub[0] -= ((action[1] * np.cos(2 * np.pi * action[0] / n_h + np.pi / 2)) + n_t * current[0]) / (size * n_t ** 2)
            sub[1] += ((action[1] * np.sin(2 * np.pi * action[0] / n_h + np.pi / 2)) + n_t * current[1]) / (size * n_t ** 2)
            sub %= 1
            pos[i * n_t + m + 1] = deepcopy(sub)
            if chart_value[int(sub[0]*dim), int(sub[1]*dim)] < -1 + 1e-6:
                print('Crashed after %d steps.' % (i+1))
                done = True
                break

        pos_est[i+1, 0] = pos_est[i, 0] - (action[1] * np.cos(2 * np.pi * action[0] / n_h + np.pi / 2) + n_t * current_e[0]) / (size * n_t)
        pos_est[i+1, 1] = pos_est[i, 1] + (action[1] * np.sin(2 * np.pi * action[0] / n_h + np.pi / 2) + n_t * current_e[1]) / (size * n_t)
        dead_rec[0] -= (action[1] * np.cos(2 * np.pi * action[0] / n_h + np.pi / 2)) / (size * n_t)
        dead_rec[1] += (action[1] * np.sin(2 * np.pi * action[0] / n_h + np.pi / 2)) / (size * n_t)
        pos_est %= 1

        if chart_value[int(sub[0]*dim), int(sub[1]*dim)] > 1 - 1e-6:
            print('Succeeded after %d steps.' % (i+1))
            done = True

        elif i+1 >= n_steps:
            print('Finished after %d steps.' % (i+1))
            done = True

    return pos, pos_est, i, no_gps, no_cur