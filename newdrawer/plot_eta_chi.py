#!/usr/bin/env python3
"""
Critical exponent eta(T) from finite-size scaling of chi.

For each temperature T, the per-site |m|^2 across L values obeys:
    <|m|^2> ~ L^{-eta}
A linear regression in log-log space yields eta.

Usage:
    python newdrawer/plot_eta_chi.py
    python newdrawer/plot_eta_chi.py --prefix thermal --L 16 32 48 64 80 100

Note: a wide range of L values is required for reliable fitting.
Generate data with:
    python generate_data.py --L 16 32 48 64 80 100 --nT 25 --prefix thermal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import load_raw_data


def fit_eta(L_arr, m2_arr):
    """
    Power-law fit of <|m|^2> vs L, returns (eta, eta_err).
    """
    log_L = np.log(L_arr)
    log_m2 = np.log(m2_arr)
    result = linregress(log_L, log_m2)
    eta = -result.slope
    eta_err = result.stderr
    return eta, eta_err


def main():
    parser = argparse.ArgumentParser(description="Fit eta(T) from chi finite-size scaling")
    parser.add_argument("--L", type=int, nargs="+",
                        default=[16, 32, 48, 64, 80, 100],
                        help="Lattice sizes (default: 16 32 48 64 80 100)")
    parser.add_argument("--prefix", type=str, default="thermal",
                        help="Prefix for pkl filenames (default: thermal)")
    parser.add_argument("--T_KT", type=float, default=0.887,
                        help="Theoretical T_KT (default: 0.887)")
    parser.add_argument("--xlim", type=float, nargs=2, default=[0, 1.0],
                        help="Temperature x-axis limits")
    parser.add_argument("--save", action="store_true",
                        help="Save figures instead of showing")
    parser.add_argument("--figdir", type=str, default=None,
                        help="Output directory for saved figures (default: newdrawer/fig)")
    args = parser.parse_args()

    # Load data and align temperatures across L
    raw_data = {}
    T_common = None
    L_list = []
    for L in args.L:
        raw = load_raw_data(L, prefix=args.prefix)
        if raw is None:
            continue
        L_list.append(L)
        raw_data[L] = raw
        if T_common is None:
            T_common = raw["temperatures"]
        else:
            T_common = np.array([t for t in T_common if t in raw["temperatures"]])

    if len(L_list) < 2:
        print(f"Error: need at least 2 L values with data, got {len(L_list)}")
        sys.exit(1)

    L_arr = np.array(L_list, dtype=float)

    if len(T_common) == 0:
        print("Error: No common temperatures across L values.")
        sys.exit(1)

    n_T = len(T_common)
    eta_arr = np.zeros(n_T)
    eta_err = np.zeros(n_T)

    print(f"Fitting eta at {n_T} temperatures...")
    for t_idx, T in enumerate(T_common):
        m2_vals = np.array([
            np.mean(raw_data[L]["M2"][t_idx, :]) for L in L_list
        ])
        eta_arr[t_idx], eta_err[t_idx] = fit_eta(L_arr, m2_vals)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(T_common, eta_arr, yerr=eta_err,
                color="orange", fmt="d", markersize=3, capsize=2, lw=0.6,
                label=r"$\eta$ from $\langle|m|^2\rangle \propto L^{-\eta}$")

    # Theory line: eta = T / (2*pi) for T < T_KT
    T_spinwave = np.linspace(0, min(args.xlim[1], args.T_KT), 100)
    ax.plot(T_spinwave, T_spinwave / (2 * np.pi), color="red", lw=0.8,
            label=r"$\eta = T/(2\pi)$ (spin-wave)")

    # eta(T_KT) = 1/4 reference
    ax.axhline(0.25, color="black", lw=0.6, linestyle=":",
               label=r"$\eta(T_{\mathrm{KT}}) = 1/4$")
    ax.axvline(args.T_KT, color="black", lw=0.6, linestyle="--",
               label=rf"$T_{{\mathrm{{KT}}}} \approx {args.T_KT}$")

    ax.set_xlabel("Temperature")
    ax.set_ylabel(r"$\eta$")
    ax.set_title(r"Critical exponent $\eta$ from susceptibility scaling")
    ax.set_xlim(args.xlim)
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=9)

    plt.tight_layout()

    if args.save:
        figdir = args.figdir or os.path.join(os.path.dirname(__file__), "fig")
        os.makedirs(figdir, exist_ok=True)
        base = os.path.splitext(os.path.basename(__file__))[0]
        for ext in ["pdf", "png"]:
            plt.savefig(os.path.join(figdir, f"{base}.{ext}"), dpi=150)
        print(f"Saved to {figdir}/{base}.pdf, .png")
    else:
        plt.show()


if __name__ == "__main__":
    main()
