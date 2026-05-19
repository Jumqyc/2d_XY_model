# XY Model — Monte Carlo Simulation of 2D BKT Transition

A high-performance C++/Python project for simulating the classical 2D XY model
and studying the Berezinskii–Kosterlitz–Thouless (BKT) phase transition.

**Note:** The report is given in `Final report.pdf`.

## Features

- **Wolff cluster algorithm** — eliminates critical slowing down
- **C++ acceleration** via pybind11 — `float32` SoA layout, AVX2, `-O3 -ffast-math`
- **Multi-process parallel data generation** — one `(T, L)` per task
- **Automatic observable recording** — magnetization, energy, helicity modulus
- **Analysis & plotting** — finite-size scaling of thermodynamic quantities

## Project Structure

```
.
├── cpp/                       # C++ core (pybind11 module)
│   ├── xy.hpp                 # XY model header
│   ├── xy.cpp                 # Wolff cluster + observable computation
│   ├── bind.cpp               # Python bindings
│   └── CMakeLists.txt         # Build config (AVX2, LTO, OpenMP)
├── generate_data.py           # Parallel MC data generator
├── data/                      # Raw simulation output (pkl)
│   └── thermal_L{L}/T{T}.pkl  # One file per (L, T)
├── newdrawer/
│   ├── analysis.py            # Shared loader & derived-quantity computation
│   ├── plot_thermal.py        # M, χ, E, C_v vs T
│   ├── plot_binder_helicity.py# Binder ratio & helicity modulus
│   ├── plot_TKT.py            # Helicity FSS → T_KT
│   └── plot_eta_chi.py        # η(T) from χ finite-size scaling
└── run_all.sh                 # One-shot: generate → plot → save
```

## Quick Start

```bash
# 1. Build the C++ module
cd cpp/build && cmake .. && cmake --build .

# 2. Test run (small params, ~1 minute)
bash run_all.sh --test

# 3. Full run (~10^5 samples per temperature)
bash run_all.sh
```

Figures are saved to `newdrawer/fig/`.

## Data Format

Each `data/thermal_L{L}/T{T:.4f}.pkl` contains:

```python
{
    "L": 16, "T": 0.1,
    "m": np.array(Ntest * 1024),  # magnetization magnitude raw sequence
    "e": np.array(Ntest * 1024),  # energy per site raw sequence
    "h": np.array(Ntest * 1024),  # per-site sin(Δθ) average raw sequence
    "params": {"Ntest": 100, "spacing": 10, "hot_runs": 5, "flush_length": 1024}
}
```

All averaging, error estimation, and derived quantities (χ, C_v, Binder ratio,
helicity modulus) are computed on-the-fly by `analysis.py` — the pkl files
store raw measurements only.

## Derived Quantities

| Quantity | Formula |
|----------|---------|
| Magnetization | $$\langle|m|\rangle$$ |
| Susceptibility | $$\chi = \beta N(\langle|m|^2\rangle - \langle|m|\rangle^2)$$ |
| Energy | $$\langle e\rangle$$ |
| Heat capacity | $$C_v = \beta^2 N(\langle e^2\rangle - \langle e\rangle^2)$$ |
| Binder ratio | $$U_4 = 1 - \langle|m|^4\rangle / (3\langle|m|^2\rangle^2)$$ |
| Helicity modulus | $$\Upsilon = -\langle e\rangle/2 - (L^2/T)\langle h^2\rangle$$ |

## Dependencies

- CMake ≥ 3.14, C++17 compiler
- Python 3.12, pybind11 (bundled in `cpp/extern/`)
- numpy, scipy, matplotlib (in `.venv`, managed by uv)
