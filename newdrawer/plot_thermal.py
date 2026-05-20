#!/usr/bin/env python3
"""
Thermodynamic quantity plots: magnetization M, susceptibility chi,
average energy E, and heat capacity C_v vs temperature.

Loads data from data/*.pkl and renders in newdrawer.

Usage:
    python newdrawer/plot_thermal.py
    python newdrawer/plot_thermal.py --L 16 32 64 --prefix thermal
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import load_thermal_data


def main():
    parser = argparse.ArgumentParser(description="Plot thermal quantities")
    parser.add_argument("--L", type=int, nargs="+", default=[16, 32, 64, 128],
                        help="Lattice sizes (default: 16 32 64 128)")
    parser.add_argument("--prefix", type=str, default="thermal",
                        help="Prefix for pkl filenames (default: thermal)")
    parser.add_argument("--xlim", type=float, nargs=2, default=[0, 1.4],
                        help="Temperature x-axis limits (default: 0 1.4)")
    parser.add_argument("--Cv_ylim", type=float, nargs=2, default=[0, 3.5],
                        help="C_v y-axis limits (default: 0 3.5)")
    parser.add_argument("--save", action="store_true",
                        help="Save figures instead of showing")
    parser.add_argument("--figdir", type=str, default=None,
                        help="Output directory for saved figures (default: newdrawer/fig)")
    args = parser.parse_args()

    data = load_thermal_data(args.L, prefix=args.prefix)
    colors = ["magenta", "blue", "red", "green", "brown"]
    markers = ["o", "^", "s", "D", "v"]

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    for idx, L in enumerate(sorted(data.keys())):
        td = data[L]
        T = td.temperatures
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        lbl = f"L = {L}"

        # Magnetization
        axs[0, 0].errorbar(T, td.M_mean, yerr=td.M_err, label=lbl,
                           color=c, fmt=m, markersize=3, capsize=2, lw=0.6)

        # Susceptibility (log scale)
        axs[0, 1].errorbar(T, L * td.chi, yerr= td.chi_err,
                           color=c, fmt=m, markersize=3, capsize=2, lw=0.6)

        # Energy
        axs[1, 0].errorbar(T, td.E_mean, yerr=td.E_err,
                           color=c, fmt=m, markersize=3, capsize=2, lw=0.6)

        # Heat capacity
        axs[1, 1].errorbar(T, td.Cv, yerr=td.Cv_err,
                           color=c, fmt=m, markersize=3, capsize=2, lw=0.6)

    axs[0, 0].set_title("Magnetization")
    axs[0, 0].set_ylabel(r"$\langle |m| \rangle$")
    axs[0, 0].set_xlim(args.xlim)

    axs[0, 1].set_title("Susceptibility")
    axs[0, 1].set_ylabel(r"$L\chi$")
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_xlim(args.xlim)

    axs[1, 0].set_title("Average energy per site")
    axs[1, 0].set_ylabel(r"$\langle e \rangle$")
    axs[1, 0].set_xlabel("Temperature")
    axs[1, 0].set_xlim(args.xlim)

    axs[1, 1].set_title("Heat capacity")
    axs[1, 1].set_ylabel(r"$C_v$")
    axs[1, 1].set_xlabel("Temperature")
    axs[1, 1].set_xlim(args.xlim)
    axs[1, 1].set_ylim(args.Cv_ylim)

    axs[0, 0].legend(fontsize=8)
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
