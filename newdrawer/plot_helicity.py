#!/usr/bin/env python3
"""
Helicity modulus 综合分析：原始测量值 + 插值曲线 + BKT 理论线 + Binder ratio。

对每个 L 的 helicity vs T 数据做三次样条插值，在同一张图上叠加
原始数据点（误差棒）和插值曲线（虚线），并加入 BKT 理论线 h = 2T/π。
通过插值曲线与理论线的交点确定有限尺寸临界温度 T_c(L)，
再用 1/(ln L)^2 外推得到热力学极限下的 T_KT。

左图：Binder cumulant U_4
右图：Helicity modulus（原始值 + 插值 + 理论线）

用法:
    python newdrawer/plot_helicity.py
    python newdrawer/plot_helicity.py --L 16 32 64 128 --prefix thermal
    python newdrawer/plot_helicity.py --save
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import bisect
from scipy.stats import linregress
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import load_thermal_data


def find_intersection(interp_fn, T_min, T_max):
    """
    用二分法找插值曲线与 h=2T/π 的交点。

    interp_fn(T) 返回 helicity，解方程 interp_fn(T) - 2T/π = 0。
    返回交点 T 值，若区间内无交点则返回 None。
    """
    def f(T):
        return interp_fn(T) - 2.0 * T / np.pi

    fa = f(T_min)
    fb = f(T_max)

    if fa * fb > 0:
        return None  # 区间内无零点

    try:
        return bisect(f, T_min, T_max, xtol=1e-8)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Helicity modulus analysis with interpolation and BKT theory line"
    )
    parser.add_argument("--L", type=int, nargs="+", default=[16, 32, 64, 128],
                        help="Lattice sizes (default: 16 32 64 128)")
    parser.add_argument("--prefix", type=str, default="thermal",
                        help="Prefix for data subdirectory (default: thermal)")
    parser.add_argument("--xlim", type=float, nargs=2, default=[0, 1.15],
                        help="Temperature x-axis limits (default: 0 1.15)")
    parser.add_argument("--ylim", type=float, nargs=2, default=None,
                        help="Helicity y-axis limits (default: auto)")
    parser.add_argument("--save", action="store_true",
                        help="Save figures instead of showing")
    parser.add_argument("--figdir", type=str, default=None,
                        help="Output directory for saved figures (default: newdrawer/fig)")
    args = parser.parse_args()

    # ── 加载数据 ──
    data = load_thermal_data(args.L, prefix=args.prefix)
    colors = ["magenta", "blue", "red", "green", "brown"]
    markers = ["o", "^", "s", "D", "v"]

    if not data:
        print("Error: no data loaded.")
        sys.exit(1)

    L_sorted = sorted(data.keys())
    L_arr = np.array(L_sorted, dtype=float)

    # ── 插值 & 交点计算 ──
    interp_info = {}  # {L: {"fn": interp1d, "T_min":, "T_max":}}
    T_cross = {}      # {L: T_c(L)} — 与 2T/π 的交点

    for L in L_sorted:
        td = data[L]
        T = td.temperatures
        h = td.helicity

        # 只对 helicity > 0 且至少有 4 个点做插值
        mask = h > 1e-8
        if mask.sum() < 4:
            print(f"WARNING: L={L} has < 4 valid helicity points, skipping interpolation")
            continue

        T_valid = T[mask]
        h_valid = h[mask]


    # ── T_c(L) 外推 → T_KT ──
    T_KT_fit = None
    T_KT_fit_err = None
    if len(T_cross) >= 3:
        # 根据 BKT 理论: T_c(L) = T_KT + A / (ln L)^2
        L_cross = np.array(sorted(T_cross.keys()), dtype=float)
        T_c = np.array([T_cross[int(l)] for l in L_cross])
        x = 1.0 / (np.log(L_cross)) ** 2
        result = linregress(x, T_c)
        T_KT_fit = float(result.intercept)
        T_KT_fit_err = float(result.intercept_stderr)
        print(f"\nFinite-size crossing temperatures T_c(L):")
        for L, Tc in sorted(T_cross.items()):
            print(f"  L={L:<4d}  T_c = {Tc:.6f}")
        print(f"Extrapolated T_KT = {T_KT_fit:.6f} ± {T_KT_fit_err:.6f}")
    else:
        print(f"\nWARNING: only {len(T_cross)} valid crossing points, "
              f"need >= 3 for T_KT extrapolation.")

    # ── 绘图 ──
    fig, (ax_binder, ax_hel) = plt.subplots(1, 2, figsize=(12, 5))

    for idx, L in enumerate(L_sorted):
        td = data[L]
        T = td.temperatures
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        lbl = f"L = {L}"

        # ── Binder ratio（左图）──
        ax_binder.errorbar(T, td.binder, yerr=td.binder_err, label=lbl,
                           color=c, fmt=m, markersize=3, capsize=2, lw=0.6)

        # ── Helicity modulus（右图）：原始数据点 ──
        ax_hel.errorbar(T, td.helicity, yerr=td.helicity_err,
                        color=c, fmt=m, markersize=3, capsize=2, lw=0.6,
                        label=lbl, zorder=5)

        # ── 插值曲线 ──
        if L in interp_info:
            info = interp_info[L]
            T_fine = np.linspace(info["T_min"], info["T_max"], 200)
            h_fine = info["fn"](T_fine)
            ax_hel.plot(T_fine, h_fine, color=c, lw=0.8, linestyle="--",
                        alpha=0.5, zorder=3)

    # ── BKT 理论线 h = 2T/π ──
    T_line = np.linspace(args.xlim[0], args.xlim[1], 300)
    h_line = 2.0 * T_line / np.pi
    ax_hel.plot(T_line, h_line, color="black", lw=1.0, linestyle="-.",
                label=r"$h = 2T/\pi$ (BKT)", zorder=4)

    # ── 标注 T_KT ──
    if T_KT_fit is not None:
        h_at_TKT = 2.0 * T_KT_fit / np.pi
        ax_hel.axvline(T_KT_fit, color="gray", lw=0.8, linestyle=":",
                       zorder=2)

    # ── 左图：Binder ratio ──
    ax_binder.set_title("Binder ratio $U_4$")
    ax_binder.set_xlabel("Temperature")
    ax_binder.set_ylabel("$U_4$")
    ax_binder.set_xlim(args.xlim)
    ax_binder.legend(fontsize=7)

    # ── 右图：Helicity ──
    ax_hel.set_title("Helicity modulus (raw + interpolation + BKT line)")
    ax_hel.set_xlabel("Temperature")
    ax_hel.set_ylabel(r"$\Upsilon$")
    ax_hel.set_xlim(args.xlim)
    if args.ylim is not None:
        ax_hel.set_ylim(args.ylim)
    ax_hel.legend(fontsize=7, loc="upper right")

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
