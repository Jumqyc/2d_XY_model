#!/bin/bash
# ============================================================
# XY Model automation: generate data -> plot -> save to newdrawer/fig
#
# Usage:
#   bash run_all.sh           # Full run (Ntest=100, ~1e5 samples/temperature)
#   bash run_all.sh --test    # Test run (small params, quick validation)
#
# generate_data.py auto-skips existing pkl files; resume-safe.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── Configuration ──────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"
FIG_DIR="$SCRIPT_DIR/newdrawer/fig"
PYTHON_CMD="uv run python"

# Full-run parameters
N_POINTS=100          # Ntest: independent runs (x1024 ~ 1e5 samples)
SPACING=10            # cluster update spacing
HOT_RUNS=5            # thermalization runs
DT=0.01                # temperature step

# Test-mode overrides
TEST_NTEST=3
TEST_NT=3
TEST_HOT=2

# ─── Mode switch ────────────────────────────────────────
MODE="full"
if [[ "${1:-}" == "--test" ]]; then
    MODE="test"
    echo ">>> TEST MODE: small parameters for quick validation <<<"
    echo ""
fi

# ─── Pre-flight checks ──────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Virtualenv not found at $VENV_DIR"
    echo "Please create it with: uv venv"
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/XY.cpython-312-x86_64-linux-gnu.so" ]]; then
    echo "ERROR: XY C++ module not found. Build it first: cd cpp/build && cmake --build ."
    exit 1
fi

# ─── Step 1: Generate data ──────────────────────────────
echo "========================================"
echo "Step 1: Generating Monte Carlo data"
echo "========================================"

if [[ "$MODE" == "test" ]]; then
    # ── Test run ──
    echo ">>> Thermal data (L=16, 32, few temps)"
    $PYTHON_CMD generate_data.py \
        --L 16 32 \
        --Tmin 0.5 --Tmax 1.2 --nT "$TEST_NT" \
        --Ntest "$TEST_NTEST" \
        --spacing "$SPACING" \
        --hot "$TEST_HOT" \
        --prefix thermal

    echo ""
    echo ">>> TKT data (L=16, 32, 64)"
    $PYTHON_CMD generate_data.py \
        --L 16 32 64 \
        --Tmin 0.7 --Tmax 0.9 --nT "$TEST_NT" \
        --Ntest "$TEST_NTEST" \
        --spacing "$SPACING" \
        --hot "$TEST_HOT" \
        --prefix tkt

else
    # ── Full run ──
    N_THERMAL=$(python3 -c "print(int((1.4 - 0.1) / $DT) + 1)")

    # 1a. Thermodynamic + eta data: single call covers all L needed
    #     by plot_thermal, plot_helicity, and plot_eta_chi
    echo ">>> Thermal: L=16,32,48,64,80,100,128,196,256,360  T in [0.1,1.4]  nT~${N_THERMAL}  Ntest=${N_POINTS}"
    $PYTHON_CMD generate_data.py \
        --L 16 32 48 64 80 100 128 196 256 360 \
        --Tmin 0.1 --Tmax 1.4 --nT "$N_THERMAL" \
        --Ntest "$N_POINTS" \
        --spacing "$SPACING" \
        --hot "$HOT_RUNS" \
        --prefix thermal

    echo ""

    # 1b. Dense TKT scan: many L + fine T steps (only near T_KT)
    N_TKT=$(python3 -c "print(int((1.0 - 0.7) / 0.001) + 1)")
    echo ">>> TKT: L=16,32,48,64,80,100,128,196,256,360  T in [0.7,0.9]  nT~${N_TKT}"
    $PYTHON_CMD generate_data.py \
        --L 16 32 48 64 80 100 128 196 256 360 \
        --Tmin 0.7 --Tmax 0.9 --nT "$N_TKT" \
        --Ntest "$N_POINTS" \
        --spacing "$SPACING" \
        --hot "$HOT_RUNS" \
        --prefix tkt
fi

# ─── Step 2: Plotting ───────────────────────────────────
echo ""
echo "========================================"
echo "Step 2: Plotting"
echo "========================================"

mkdir -p "$FIG_DIR"

if [[ "$MODE" == "test" ]]; then
    # Test mode: only plot L values that were generated
    echo ">>> plot_thermal"
    $PYTHON_CMD newdrawer/plot_thermal.py --save --figdir "$FIG_DIR" --L 16 32

    echo ">>> plot_helicity"
    $PYTHON_CMD newdrawer/plot_helicity.py --save --figdir "$FIG_DIR" --L 16 32

    echo ">>> plot_eta_chi"
    $PYTHON_CMD newdrawer/plot_eta_chi.py --save --figdir "$FIG_DIR" --L 16 32
else
    echo ">>> plot_thermal"
    $PYTHON_CMD newdrawer/plot_thermal.py --save --figdir "$FIG_DIR"

    echo ">>> plot_helicity"
    $PYTHON_CMD newdrawer/plot_helicity.py --save --figdir "$FIG_DIR"

    echo ">>> plot_eta_chi"
    $PYTHON_CMD newdrawer/plot_eta_chi.py --save --figdir "$FIG_DIR"
fi

echo ""
echo "========================================"
echo "Done! Figures saved to $FIG_DIR"
ls -lh "$FIG_DIR"
echo "========================================"
