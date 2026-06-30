# Dynamic Influence Opinion Model Simulator
This repository contains a Python simulator for opinion dynamics in
social networks. The implementation is centered on DeGroot-type opinion updates
and on dynamic influence mechanisms where agents may change how much weight they
assign to others over time.

## Model Overview

The simulator represents a social network as a set of agents. Each agent has:

- an opinion value,
- an influence vector describing how much every agent affects its next opinion,
- optional functions that describe how influence weights change over time.

At each iteration, opinions are updated using the DeGroot model:

```text
x(t + 1) = A(t) x(t)
```

where `x(t)` is the opinion vector and `A(t)` is the influence matrix at time
`t`. In the standard case, influence changes are defined by explicit update
functions. In the dynamic case, influence weights are recomputed from opinion
differences, which allows simulations of homophily-driven influence.

The code uses `mpmath` matrices for the simulation state. This makes it possible
to run experiments that require more decimal precision than `float64` can
provide. Values are converted to `float64` only when required by visualization
libraries such as NetworkX and Matplotlib.

## MVC Structure

The simulator follows a simple Model-Controller-View organization. The model
owns the simulation state and numerical updates, the controller runs the
iteration process, and the view handles plots and graph visualizations.

## Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

## Running Simulations

Run the interactive launcher from the repository root:

```powershell
python main.py
```

The launcher asks for a simulation number and calls the corresponding function
from `simulations/`.

An individual simulation can also be executed directly:

```powershell
python -c "from simulations.simulation1 import simulation1; simulation1()"
```

When adding new simulations, use the following convention:

```text
simulations/simulationN.py
```

with an entry point named:

```python
def simulationN():
    ...
```

If the simulation should be available from `main.py`, expose it in
`simulations/__init__.py`.

## Precision

The decimal precision is configured in `main.py`:

```python
from model.precision import set_precision

set_precision(200)
```

Internally, opinions and influence weights are stored as `mpmath` values. This
is important for long simulations or cases where very small influence weights
must not underflow to zero. For readability, printed matrices can be formatted
with `mp.nstr`, which shows high-precision values without converting them to
standard Python floats.

## Visualization

The `view/` module provides:

- opinion history plots,
- minimum positive influence plots,
- directed influence graph rendering,
- graph animations controlled with the left and right arrow keys.

The influence matrix is transposed before graph construction because DeGroot
rows represent how an agent is influenced by others, while NetworkX edges point
from influencer to influenced agent.

## Research Notes

The standard influence-update path can be used to reproduce or inspect behavior
related to open-minded agents, such as the setting discussed by Chatterjee and
Seneta (1977). The dynamic update path is intended for experiments where
influence depends on opinion differences, including homophily-driven mechanisms (currently under study, as part as my research bachelor's thesis).

Reference material is stored in `docs/`.
