#!/usr/bin/env python3
"""
Binder ratio and Helicity modulus plots.

Usage:
    python newdrawer/plot_binder_helicity.py
    python newdrawer/plot_binder_helicity.py --L 16 32 64 128 --prefix thermal
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import load_thermal_data


def main():
    parser = argparse.ArgumentParser(description="Plot Binder ratio & Helicity modulus")
    parser.add_argument("--L", type=int, nargs="+", default=[16, 32, 64, 128],
                        help="Lattice sizes (default: 16 32 64 128)")
    parser.add_argument("--prefix", type=str, default="thermal",
                        help="Prefix for pkl filenames (default: thermal)")
    parser.add_argument("--xlim", type=float, nargs=2, default=[0, 1.15],
                        help="Temperature x-axis limits (default: 0 1.15)")
    parser.add_argument("--T_KT", type=float, default=0.887,
                        help="Theoretical T_KT reference line (default: 0.887)")
    parser.add_argument("--save", action="store_true",
                        help="Save figures instead of showing")
    parser.add_argument("--figdir", type=str, default=None,
                        help="Output directory for saved figures (default: newdrawer/fig)")
    args = parser.parse_args()

    data = load_thermal_data(args.L, prefix=args.prefix)
    colors = ["magenta", "blue", "red", "green", "brown"]
    markers = ["o", "^", "s", "D", "v"]

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    for idx, L in enumerate(sorted(data.keys())):
        td = data[L]
        T = td.temperatures
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        lbl = f"L = {L}"

        # Binder ratio
        axs[0].errorbar(T, td.binder, yerr=td.binder_err, label=lbl,
                        color=c, fmt=m, markersize=3, capsize=2, lw=0.6)

        # Helicity modulus
        axs[1].errorbar(T, td.helicity, yerr=td.helicity_err,
                        color=c, fmt=m, markersize=3, capsize=2, lw=0.6)

    # T_KT reference line
    axs[0].axvline(args.T_KT, color="black", lw=0.6, linestyle="--",
                   label=rf"$T_{{\mathrm{{KT}}}} \approx {args.T_KT}$")
    axs[1].axvline(args.T_KT, color="black", lw=0.6, linestyle="--")

    axs[0].set_title("Binder ratio $U_4$")
    axs[0].set_xlabel("Temperature")
    axs[0].set_ylabel("$U_4$")
    axs[0].set_xlim(args.xlim)
    axs[0].legend(fontsize=8)

    axs[1].set_title("Helicity modulus")
    axs[1].set_xlabel("Temperature")
    axs[1].set_ylabel(r"$\Upsilon$")
    axs[1].set_xlim(args.xlim)

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
