# SubWorld Dynamic Programming

This repository conatins the code required to create SubWorld:
- Charts
- Water currents
- Value functions
- Policies

## Generating Charts

Running **gen_land.py** will generate a single chart with water currents stored in **data/charts/charts_SEED.npz**. Running **gen_land_many.py** will generate many charts with water currents stored in **data/charts/charts_SEED.npz**. These charts are generated in parallel using **multiprocessing**.

The charts and water currents can be visualized using **plot_chart.py**.

<img src="PNGs/charts/chart_2525.png" width="500"/>

## Generating Value Functions

Running **chart_value.py** will generate the value function assuming the chart has no water currents. The value function is generated using Bellman's dynamic programming algorithm and will be stored in **data/value/value_SEED.npz**. Running **chart_value_many.py** will generate the value functions for each chart from **gen_land_many.py** stored in **data/value/value_SEED.npz**.

The value functions can be visualized using **plot_value.py**.

<img src="PNGs/value/value_2525.png" width="500"/>

## Generating Policies

Running **chart_policy_gps.py** will generate a policy based on the value function. The policy's trajectectory will be stored in **data/policy/policy_gps_SEED.npz**. Running **chart_policy_many.py** will generate policies for each value function from **chart_value_many.py** stored in **data/policy/policy_gps_SEED.npz**.

The policy trajectories can be visualized using **plot_policy.py**.

## YAML Parameters

The parameters used in each task are stored in **params.yaml**. Some parameters may carry forward into other tasks.

### Generating Charts

|Parameter   |Type        |Description |
|------------|------------|------------|
|seed        |Non-negative **int** |The random seed used create the islands that define the chart. |
|dim         |Positive **int** |The dimension size of the chart. |
|n_islands   |Non-negative **int** or **None**| The number of islands that will be generated. Setting **None** will result in a random number of islands. |
|min_islands |Non-negative **int** |The minimum number of islands that will be generated if **n_islands** is **None**. |
|max_islands |Non-negative **int** |The maximum number of islands that will be generated if **n_islands** is **None**. |
|size        |Non-negative **float** |The size of the x and y dimensions of the chart. The Submarine can move up to 1 unit per action. |
|min_height  |Non-negative **float** |The minimum height of each island. |
|max_height  |Positive **float** > min_height |The maximum height of each island. |
|x_decay_min |Positive **float** |The minimum decay rate in the x direction of each island. |
|x_decay_max |Positive **float** > x_decay_min |The maximum decay rate in the x direction of each island. |
|y_decay_min |Positive **float** |The minimum decay rate in the y direction of each island. |
|y_decay_max |Positive **float** > y_decay_min |The maximum decay rate in the y direction of each island. |
|max_cur     |Non-negative **float** |The maximum water current magnitude. |

### Value Function Generation

|Parameter   |Type        |Description |
|------------|------------|------------|
|target_x    |Non-negative **float** < dim or **None**|The x coordinate of the target. |
|target_y    |Non-negative **float** < dim or **None**|The y coordinate of the target. |
|discount    |Non-negative **float** < 1 |The discount factor used in Bellman's dynamic programming algorithm. |
|n_t         |Positive **int** |The number of discretized throttle actions. |
|n_h         |Positive **int** |The number of discritized heading actions. |
|tol         |**float** |Log base 10 of the convergence tolerence used in Bellman's dynamic programming algorithm. I.e. tol=-6 -> a tolerence of 1e-6.|

### Extra Paramters for Parallelization

|Parameter   |Type        |Description |
|------------|------------|------------|
|n_cpu       |Positive **int** |The number of CPUs used in **multiprocessing.Pool** when generating many charts/value functions. |
|maps_i      |Non-negative **int** |The seed to start at when generating many charts. |
|maps_f      |Positive **int** > maps_i |The seed to end at when generating many charts.

### Policy Generation

|Parameter   |Type        |Description |
|------------|------------|------------|
|sub_x       |Non-negative **float** < dim or **None**|The x coordinate for the submarine's starting position. Setting **None** will result in a random coordinate. |
|sub_y       |Non-negative **float** < dim or **None**|The y coordinate for the submarine's starting position. Setting **None** will result in a random coordinate. |
|n_steps     |Positive **int** |The maximum number of steps the agent can take before the episode ending. |
|gps_cost    |Non-negative **float** |The cost required to use the GPS. If set to zero, the GPS will be used at every step. |
|cur_cost    |Non-negative **float** |The cost required to use the Current Profiler. If set to zero, the Current Profiler will be used at every step. |
|uncert_pos  |Non-negative **float** |The rate at which uncertainty in position increases. |
|uncert_cur  |Non-negative **float** |The rate at which uncertainty in water current increases. |
