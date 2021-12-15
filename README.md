# SubWorld Dynamic Programming

This repository conatins the code required to create SubWorld:
- Charts
- Water currents
- Value functions
- Policies

## Generating Charts

Running **gen_land_det.py** will generate a single chart with water currents stored in **data/charts/charts_SEED.npz**. Running **gen_land_many.py** will generate many charts with water currents stored in **data/charts/charts_SEED.npz**. These charts are generated in parallel using _multiprocessing_.

The charts and water currents can be visualized using **plot_chart.py**.

## Generating Value Functions

Running **chart_value.py** will generate the value function assuming the chart has no water currents. The value function will be stored in **data/value/value_SEED.npz**. Running **chart_value_many.py** will generate the value functions for each chart from **gen_land_many.py** stored in **data/value/value_SEED.npz**.

The value functions can be visualized using **plot_value.py**.

## Generating Policies

Running **chart_policy_gps.py** will generate a policy based on the value function. The policy's trajectectory will be stored in **data/policy/policy_GPS_SEED.npz**.

The policy trajectories can be visualized using **plot_policy.py**.

## YAML Parameters
