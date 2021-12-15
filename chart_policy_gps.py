import numpy as np
from copy import deepcopy
import yaml
from os import getcwd

path = getcwd()

with open(path + '/params.yaml', 'r') as F:
    params = yaml.safe_load(F)

seed = params['seed']
size = params['size']
n_t = params['n_t']
n_h = params['n_h']
n_steps = params['n_steps']
gps_cost = params['gps_cost']
uncert_i = params['uncert_i']
uncert_inc = params['uncert_inc']
target = params['target']

data1 = np.load(path + '/data/value/value_' + str(seed) + '.npz')
chart_value = data1['value']
steps_est = data1['steps']
discount = float(data1['discount'])
data = np.load(path + '/data/charts/charts_' + str(seed) + '.npz')
chart = data['chart']
dim = chart.shape[0]

try:
    sub = np.array([params['sub_x'], params['sub_y']]) / dim
except TypeError:
    place = False
    while not place:
        sub = np.random.rand(2)
        if chart_value[int(dim*sub[0]), int(dim*sub[1])] > -1 + 1e-6:
            place = True

water = data['water_c']
water_p = np.meshgrid(np.arange(0, dim+1, 3), np.arange(0, dim+1, 3))
water_c = water[0::3, 0::3]

pos = np.zeros((n_steps*n_t+1, 2), dtype=np.float32)
pos_est = np.zeros((n_steps+1, 2), dtype=np.float32)
pos[0] = deepcopy(sub)
current_e = water[int(sub[0]*dim), int(sub[1]*dim)]
pos_est[0] = deepcopy(sub)
uncert = deepcopy(uncert_i)
no_gps = []
no_gps_line = []
action = (0.0, 0.0)
done = False
i = -1
while not done:
    i += 1
    values = np.zeros((n_h, n_t), dtype=np.float32) - 1
    uncert_x = np.linspace(-1.0 * uncert, uncert, 10, endpoint = True)
    uncert_y = np.linspace(-1.0 * uncert, uncert, 10, endpoint = True)
    for k in range(n_h):
        for l in range(n_t):
            value = np.zeros((uncert_x.shape[0], uncert_y.shape[0]), dtype=np.float32) - 1
            for unc_x in range(uncert_x.shape[0]):
                for unc_y in range(uncert_y.shape[0]):
                    ind1 = 0
                    ind2 = 0
                    for m in range(n_t):
                        ind1 += dim * ((l * np.cos(2 * np.pi * k / n_h + np.pi/2) + n_t * (current_e[0] + uncert_x[unc_x])) / (size * n_t ** 2))
                        ind2 += dim * ((l * np.sin(2 * np.pi * k / n_h + np.pi/2) + n_t * (current_e[1] + uncert_y[unc_y])) / (size * n_t ** 2))
                        value[unc_x, unc_y] = chart_value[int(pos_est[i, 0]*dim + ind1) % dim, int(pos_est[i, 1]*dim - ind2) % dim]
                        if value[unc_x, unc_y] < -1 + 1e-6:
                            break

            values[k, l] = value.mean()

    action = np.unravel_index(values.argmax(), values.shape)
    pos_look_x = pos_est[i, 0] + (action[1] * np.cos(2 * np.pi * action[0] / n_h + np.pi / 2) + n_t * current_e[0]) / (size * n_t)
    pos_look_y = pos_est[i, 1] - (action[1] * np.sin(2 * np.pi * action[0] / n_h + np.pi / 2) + n_t * current_e[1]) / (size * n_t)    
    v_est = (values.max() + 1.0) / (discount ** (np.max([steps_est[int(pos_look_x*dim) % dim, int(pos_look_y*dim) % dim] - 1, 0])))
    if (v_est < 2.0 - gps_cost and i != 0) or chart_value[int(pos_est[i, 0]*dim) % dim, int(pos_est[i, 1]*dim) % dim] > 1 - 1e-6:
        uncert = deepcopy(uncert_i)
        current_e = sub - pos_est[i]
        pos_est[i] = deepcopy(sub)
        if current_e[0] > 0.5:
            current_e[0] -= 1.0
        elif current_e[0] < -0.5:
            current_e[0] += 1.0
        if current_e[1] > 0.5:
            current_e[1] -= 1.0
        elif current_e[1] < -0.5:
            current_e[1] += 1.0

        current_e *= np.array([size, -1.0*size])
    
        values = np.zeros((n_h, n_t), dtype=np.float32) - 1
        uncert_x = np.linspace(-1.0 * uncert, uncert, 10, endpoint = True)
        uncert_y = np.linspace(-1.0 * uncert, uncert, 10, endpoint = True)
        for k in range(n_h):
            for l in range(n_t):
                value = np.zeros((uncert_x.shape[0], uncert_y.shape[0]), dtype=np.float32) - 1
                for unc_x in range(uncert_x.shape[0]):
                    for unc_y in range(uncert_y.shape[0]):
                        ind1 = 0
                        ind2 = 0
                        for m in range(n_t):
                            ind1 += dim * ((l * np.cos(2 * np.pi * k / n_h + np.pi/2) + n_t * (current_e[0] + uncert_x[unc_x])) / (size * n_t ** 2))
                            ind2 += dim * ((l * np.sin(2 * np.pi * k / n_h + np.pi/2) + n_t * (current_e[1] + uncert_y[unc_y])) / (size * n_t ** 2))
                            value[unc_x, unc_y] = chart_value[int(pos_est[i, 0]*dim + ind1) % dim, int(pos_est[i, 1]*dim - ind2) % dim]
                            if value[unc_x, unc_y] < -1 + 1e-6:
                                break

                values[k, l] = value.mean()
                    
    else:
        no_gps.append(i)
        if i-1 not in no_gps_line:
            no_gps_line.append(i-1)

        no_gps_line.append(i)
        uncert += uncert_inc

    action = np.unravel_index(values.argmax(), values.shape)
    for m in range(n_t):
        current = water[int(sub[0]*dim), int(sub[1]*dim)]
        sub[0] += (action[1] * np.cos(2 * np.pi * action[0] / n_h + np.pi / 2) + n_t * current[0]) / (size * n_t ** 2)
        sub[1] -= (action[1] * np.sin(2 * np.pi * action[0] / n_h + np.pi / 2) + n_t * current[1]) / (size * n_t ** 2)
        sub %= 1
        pos[i * n_t + m + 1] = deepcopy(sub)
        if chart_value[int(sub[0]*dim), int(sub[1]*dim)] < -1 + 1e-6:
            print('Crashed after %d steps.' % (i+1))
            done = True
            break

    pos_est[i+1, 0] = pos_est[i, 0] + (action[1] * np.cos(2 * np.pi * action[0] / n_h + np.pi / 2) + n_t * current_e[0]) / (size * n_t)
    pos_est[i+1, 1] = pos_est[i, 1] - (action[1] * np.sin(2 * np.pi * action[0] / n_h + np.pi / 2) + n_t * current_e[1]) / (size * n_t)
    pos_est %= 1

    if chart_value[int(sub[0]*dim), int(sub[1]*dim)] > 1 - 1e-6:
        print('Succeeded after %d steps.' % (i+1))
        done = True

    elif i+1 >= n_steps:
        print('Finished after %d steps.' % (i+1))
        done = True

np.savez(path + '/data/policy/policy_gps_' + str(seed) + '.npz', pos = pos[:(i+1)*n_t], pos_est = pos_est[:i+1], no_gps = no_gps[1:])
