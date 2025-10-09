# MPPI Controller Project

This project implements a Model Predictive Path Integral (MPPI) controller using JAX for high-performance control of robotic systems.

## Getting Started

### Prerequisites

- Python 3.8+
- Pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/leonardjmihs/mppI_filter
    cd mppI_filter
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the MPPI simulation with random scenarios, execute the main script:

```bash
python3 main.py
```

You can also specify a different solver:

```bash
python3 main.py --solver <solver_name>
```

## Project Structure

- `main.py`: The main entry point for running simulations.
- `requirements.txt`: A list of all Python packages required for the project.
- `jax_mppi/`: Contains the core MPPI implementation, including planners, grid utilities, and simulation scripts.
- `systems/`: Defines the dynamics models for the systems being controlled (e.g., Unicycle, HalfCar).
- `tests/`: Contains tests for the project.
# mppI_filter
