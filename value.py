import numpy as np
from copy import deepcopy

# Function to generate a value function
def gen_value(path, seed, dim, target_x, target_y, size, tol, n_h, n_t, discount):
    # Load the chart
    data = np.load(path + '/data/charts/charts_' + str(seed) + '.npz')
    chart = data['chart']
    # Assume water current is unknown so it is not loaded
    # Can be replaced with true water current to get more accurate value function
    water_c = np.zeros((dim, dim, 2), dtype=np.float32)

    # Place the target either where specified or randomly not on an island
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
    # Loop of each pixel in the chart to initialize the value and cost functions
    for i in range(dim):
        for j in range(dim):
            # Check if pixel corresponds to an island
            if chart[i, j] >= -0.1:
                chart_value[i, j] = -99
                rel_chart_value[i, j] = -99

            # Check if pixel instead corresponds to the target
            elif np.sqrt(((i + 0.5)/dim - target[0])**2 + ((j + 0.5)/dim - target[1])**2) < 0.005*size:
                chart_value[i, j] = 2
                rel_chart_value[i, j] = 2

    dif_chart = np.zeros((dim, dim), dtype=np.float32) + 100
    dif = 100

    count = 0

    # Loop to iteratively update the value function until converged
    while dif > tol:
        count += 1
        old_value = deepcopy(chart_value)
        old_rel_value = deepcopy(rel_chart_value)
        # Loop over each pixel in the chart
        for i in range(dim):
            for j in range(dim):
                # Check if a value has been assigned yet
                if old_value[i, j] < 2.0 - 1e-6 and old_value[i, j] > -99 + 1e-6 and dif_chart[i, j] > 1e-6:
                    values = np.zeros((n_h, n_t), dtype=np.float32)
                    rel_values = np.zeros((n_h, n_t), dtype=np.float32)
                    # Loop over each action
                    for k in range(n_h):
                        for l in range(n_t):
                            ind1 = 0
                            ind2 = 0
                            # Loop over multiple time steps
                            for m in range(n_t):
                                # Get water current for true submarine position
                                current = water_c[int(i + ind1) % dim, int(j + ind2) % dim]
                                # Advance estimate of position based on action and water current
                                ind1 += dim * ((l * np.cos(2 * np.pi * k / n_h) + n_t * current[0]) / (size * n_t ** 2))
                                ind2 += dim * ((l * np.sin(2 * np.pi * k / n_h) + n_t * current[1]) / (size * n_t ** 2))
                                # Get value and cost for position and action 
                                values[k, l] = old_value[int(i + ind1) % dim, int(j + ind2) % dim] - 0.001 * l / n_t
                                rel_values[k, l] = old_rel_value[int(i + ind1) % dim, int(j + ind2) % dim] - 0.001 * l / n_t
                                # Check if submarine has crashed
                                if values[k, l] < 1e-6:
                                    break
                    
                    # Discount and store the maximum value all actions result in
                    chart_value[i, j] = discount * np.max(values)
                    rel_chart_value[i, j] = rel_values[np.unravel_index(values.argmax(), values.shape)]

        # Keep track of how the value function has changed for convergence purposes
        dif_chart = abs(old_value - chart_value)
        dif = np.mean(dif_chart)
        print(dif)

    # Shift the value function to be between -100 and +1
    chart_value -= 1
    rel_chart_value -= 1

    return chart_value, rel_chart_value
