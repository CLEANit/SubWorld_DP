import numpy as np
from copy import deepcopy
from math import copysign

# Function to simulate an action trajectory
def sim_action(x, y, dim, n_h, n_t, action, current, size, chart_value, rel_chart_value, uncert_cur_x, uncert_cur_y, max_cur):
    ind1 = 0
    ind2 = 0
    k = action[0]
    l = action[1]
    # Loop over multiple time steps
    for m in range(n_t):
        # Advance estimate of position based on action, water current, and uncertainty
        ind1 += dim * ((l * np.cos(2 * np.pi * k / n_h + np.pi/2) - n_t * np.clip(current[0] + m * uncert_cur_x, -1.0*max_cur, max_cur)) / (size * n_t ** 2))
        ind2 += dim * ((l * np.sin(2 * np.pi * k / n_h + np.pi/2) + n_t * np.clip(current[1] + m * uncert_cur_y, -1.0*max_cur, max_cur)) / (size * n_t ** 2))
        # Collect value and cost of the position
        value = chart_value[int(x*dim - ind1) % dim, int(y*dim + ind2) % dim]
        rel_value = rel_chart_value[int(x*dim - ind1) % dim, int(y*dim + ind2) % dim]
        # Check if the position results in a crash
        if value < -1 + 1e-6:
            return value, rel_value

    return value, rel_value

# Function to estimate the values and costs of each action using sim_action
def est_value(n_h, n_t, unc_pos, unc_cur, pos_est, dim, current_e, size, chart_value, rel_chart_value, uncert_cur, max_cur, uncert_res):
    values = np.zeros((n_h, n_t), dtype=np.float32) - 1
    rel_values = np.zeros((n_h, n_t), dtype=np.float32) - 1
    # Create a range of positions to simulate in uncertainty
    uncert_pos_x = np.linspace(-1.0 * unc_pos, unc_pos, uncert_res, endpoint = True)
    uncert_pos_y = np.linspace(-1.0 * unc_pos, unc_pos, uncert_res, endpoint = True)
    # Create a range of water currents to simulate in uncertainty
    uncert_cur_x = np.linspace(-1.0 * unc_cur, unc_cur, uncert_res, endpoint = True)
    uncert_cur_y = np.linspace(-1.0 * unc_cur, unc_cur, uncert_res, endpoint = True)
    # Loop over heading actions
    for k in range(n_h):
        # Loop over throttle actions
        for l in range(n_t):
            value = np.zeros((uncert_pos_x.shape[0], uncert_pos_y.shape[0], uncert_cur_x.shape[0], uncert_cur_y.shape[0]), dtype=np.float32) - 1
            rel_value = np.zeros((uncert_pos_x.shape[0], uncert_pos_y.shape[0], uncert_cur_x.shape[0], uncert_cur_y.shape[0]), dtype=np.float32) - 1
            # Loop over positions in uncertainty range
            for unc_pos_x in range(uncert_pos_x.shape[0]):
                for unc_pos_y in range(uncert_pos_y.shape[0]):
                    # Loop over water currents in uncertainty range
                    for unc_cur_x in range(uncert_pos_x.shape[0]):
                        for unc_cur_y in range(uncert_pos_y.shape[0]):
                            # Get value and cost for position, water current, and action 
                            value[unc_pos_x, unc_pos_y, unc_cur_x, unc_cur_y], rel_value[unc_pos_x, unc_pos_y, unc_cur_x, unc_cur_y] = sim_action(pos_est[0] + unc_pos_x, pos_est[1] + unc_pos_y, dim, n_h, n_t, np.array([k, l]), current_e - np.array([unc_cur_x, unc_cur_y]), size, chart_value, rel_chart_value, copysign(uncert_cur, unc_cur_x), copysign(uncert_cur, unc_cur_y), max_cur)

            # Take average value and cost over all estimates
            values[k, l] = value.mean()
            rel_values[k, l] = rel_value.mean()

    return values, rel_values

# Function to generate a policy trajectory using est_value
def policy_gps(path, seed, sub_x, sub_y, n_steps, n_t, uncert_pos, n_h, size, gps_cost, cur_cost, uncert_cur, max_cur, uncert_res):
    # Load the value and cost functions
    data1 = np.load(path + '/data/value/value_' + str(seed) + '.npz')
    chart_value = data1['value']
    rel_chart_value = data1['rel_value']
    # Load chart
    data = np.load(path + '/data/charts/charts_' + str(seed) + '.npz')
    chart = data['chart']
    dim = chart.shape[0]
    
    # Place the submarine either where specified or randomly not on an island
    try:
        sub = np.array([sub_x, sub_y]) / dim
    except TypeError:
        place = False
        while not place:
            sub = np.random.rand(2)
            if chart_value[int(dim*sub[0]), int(dim*sub[1])] > -1 + 1e-6:
                place = True

    # Load the water current
    water = data['water_c']
    
    pos = np.zeros((n_steps*n_t+1, 2), dtype=np.float32)
    pos_est = np.zeros((n_steps+1, 2), dtype=np.float32)
    pos[0] = deepcopy(sub)
    current_e = deepcopy(water[int(sub[0]*dim), int(sub[1]*dim)])
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
    # Loop until max time step, agent succeeds, or agent crashes
    while not done:
        i += 1
        # Get values and costs for all actions
        values, rel_values = est_value(n_h, n_t, unc_pos, unc_cur, pos_est[i], dim, current_e, size, chart_value, rel_chart_value, uncert_cur, max_cur, uncert_res)
        v_est = rel_values[np.unravel_index(values.argmax(), values.shape)] - unc_pos - unc_cur

        # Determine if GPS measurement should be taken
        if (v_est < 1.0 - gps_cost and i != 0) or chart_value[int(pos_est[i, 0]*dim) % dim, int(pos_est[i, 1]*dim) % dim] > 1 - 1e-6:
            # Update uncertainty values
            unc_pos = 0.0
            unc_cur = uncert_cur * (last_gps + 1)
            last_gps = 0
            # Calculate new estimate in water current
            current_e = sub - dead_rec
            # Update position estimate
            pos_est[i] = deepcopy(sub)
            dead_rec = deepcopy(sub)
            # Adjust for PBCs in water current estimate
            if current_e[0] > 0.5:
                current_e[0] -= 1.0
            elif current_e[0] < -0.5:
                current_e[0] += 1.0
            if current_e[1] > 0.5:
                current_e[1] -= 1.0
            elif current_e[1] < -0.5:
                current_e[1] += 1.0

            current_e *= np.array([-1.0*size, size])

            # Get values and costs for all actions after measurement
            values, rel_values = est_value(n_h, n_t, unc_pos, unc_cur, pos_est[i], dim, current_e, size, chart_value, rel_chart_value, uncert_cur, max_cur, uncert_res)
            v_est = rel_values[np.unravel_index(values.argmax(), values.shape)] - (unc_pos + unc_cur - uncert_cur) / 100.0

            unc_cur -= uncert_cur
        
        else:
            # Increase uncertainty if GPS measurement not taken
            no_gps.append(i)
            last_gps += 1
            unc_pos += uncert_pos

        # Determine if current profiler measurement should be taken
        if (v_est < 1.0 - cur_cost and i != 0):
            # Update water current estimate
            current_e = water[int(sub[0]*dim), int(sub[1]*dim)]
            # Update water current uncertainty
            unc_cur = 0.0001

            # Get values and costs for all actions after measurement
            values, rel_values = est_value(n_h, n_t, unc_pos, unc_cur, pos_est[i], dim, current_e, size, chart_value, rel_chart_value, uncert_cur, max_cur, uncert_res)
        
        else:
            # Increase uncertainty if current profiler measurement not taken
            no_cur.append(i)
            unc_cur += uncert_cur

        # Choose action with the highest estimated value
        action = np.unravel_index(values.argmax(), values.shape)
        # Loop over multiple time steps to update the true submarine position
        for m in range(n_t):
            # Get water current for true submarine position
            current = water[int(sub[0]*dim), int(sub[1]*dim)]
            # Advance position based on action and water current
            sub[0] -= ((action[1] * np.cos(2 * np.pi * action[0] / n_h + np.pi / 2)) - n_t * current[0]) / (size * n_t ** 2)
            sub[1] += ((action[1] * np.sin(2 * np.pi * action[0] / n_h + np.pi / 2)) + n_t * current[1]) / (size * n_t ** 2)
            sub %= 1
            pos[i * n_t + m + 1] = deepcopy(sub)
            # Check if submarine has crashed
            if chart_value[int(sub[0]*dim), int(sub[1]*dim)] < -1 + 1e-6:
                done = True
                status = 0
                break

        # Update estimated position based on action and water current estimate
        pos_est[i+1, 0] = pos_est[i, 0] - (action[1] * np.cos(2 * np.pi * action[0] / n_h + np.pi / 2) - n_t * current_e[0]) / (size * n_t)
        pos_est[i+1, 1] = pos_est[i, 1] + (action[1] * np.sin(2 * np.pi * action[0] / n_h + np.pi / 2) + n_t * current_e[1]) / (size * n_t)
        dead_rec[0] -= (action[1] * np.cos(2 * np.pi * action[0] / n_h + np.pi / 2)) / (size * n_t)
        dead_rec[1] += (action[1] * np.sin(2 * np.pi * action[0] / n_h + np.pi / 2)) / (size * n_t)
        pos_est %= 1

        # Check if the submarine is in the target
        if chart_value[int(sub[0]*dim), int(sub[1]*dim)] > 1 - 1e-6:
            done = True
            status = 1

        # Check if the maximum number of time steps has elapsed
        elif i+1 >= n_steps:
            done = True
            status = 2

    return pos, pos_est, i, no_gps, no_cur, status