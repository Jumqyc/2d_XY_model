#!/usr/bin/env python3
"""
Helicity modulus 有限尺寸标度 → T_KT 精确值。

对每个温度 T，用多 L 的 helicity modulus 做外推：
    Υ(L) = Υ_∞ + a / (ln L + C)
通过最大化 R² 找最佳 C，得到 Υ_∞(T)。
T_KT 定义为 Υ_∞ → 0 的温度。

需要密集的 L 和 T 扫描数据。用 generate_data.py 生成：
    python generate_data.py --L 8 16 24 32 48 64 80 96 \\
        --Tmin 0.7 --Tmax 0.9 --nT 50 --prefix tkt

用法:
    python newdrawer/plot_TKT.py
    python newdrawer/plot_TKT.py --prefix tkt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import load_raw_data


def gradient_descent(f, x0=0.0, x1=1.0):
    """梯度下降找 f(x) 的最大值."""
    gamma = 1.0
    n = 0
    while True:
        n += 1
        df = (f(x1) - f(x0)) / (x1 - x0 + 1e-15)
        x0, x1 = x1, x1 + gamma * df
        if abs(x0 - x1) < 1e-12:
            break
        if n == 10:
            gamma /= 2
            n = 0
    return (x0 + x1) / 2


def fit_helicity_infty(L_arr, helicity_arr):
    """
    对给定温度下多 L 的 helicity 做外推拟合 Υ_∞。

    Υ(L) = Υ_∞ + a / (ln L + C)
    通过最大化 R² 找最佳 C，返回 (Υ_∞, Υ_∞_err).
    """

    def r_squared(C):
        x = 1.0 / (np.log(L_arr) + C)
        result = linregress(x, helicity_arr)
        return result.rvalue ** 2

    C_best = gradient_descent(r_squared)
    x_best = 1.0 / (np.log(L_arr) + C_best)
    result = linregress(x_best, helicity_arr)
    return result.intercept, result.intercept_stderr


def main():
    parser = argparse.ArgumentParser(description="Helicity finite-size scaling → T_KT")
    parser.add_argument("--prefix", type=str, default="tkt",
                        help="Prefix for data subdirectory (default: tkt)")
    parser.add_argument("--L", type=int, nargs="+",
                        default=[8, 16, 24, 32, 48, 64, 80, 96],
                        help="Lattice sizes used in scaling")
    parser.add_argument("--T_KT_theory", type=float, default=0.887,
                        help="Theoretical T_KT for reference")
    parser.add_argument("--save", action="store_true",
                        help="Save figures instead of showing")
    parser.add_argument("--figdir", type=str, default=None,
                        help="Output directory for saved figures (default: newdrawer/fig)")
    args = parser.parse_args()

    print("Loading data...")
    helicity_data = {}  # {L: helicity_mean array}
    T_common = None
    L_list = []
    for L in args.L:
        raw = load_raw_data(L, prefix=args.prefix)
        if raw is None:
            continue
        L_list.append(L)
        # raw["H"] shape (n_T, Ntest), 取每行平均 = helicity modulus
        helicity_data[L] = np.mean(raw["H"], axis=1)
        if T_common is None:
            T_common = raw["temperatures"]
        else:
            T_common = np.array([t for t in T_common if t in raw["temperatures"]])

    if len(L_list) < 2:
        print(f"Error: need at least 2 L values with data, got {len(L_list)}")
        sys.exit(1)

    if len(T_common) == 0:
        print("Error: No common temperatures across L values.")
        sys.exit(1)

    # 对每个温度做外推拟合
    helicity_infty = np.zeros(len(T_common))
    helicity_infty_err = np.zeros(len(T_common))

    L_arr = np.array(L_list, dtype=float)
    for t_idx, T in enumerate(T_common):
        h_vals = np.array([helicity_data[L][t_idx] for L in L_list])
        helicity_infty[t_idx], helicity_infty_err[t_idx] = fit_helicity_infty(L_arr, h_vals)
        print(f"T={T:.4f}  Y_infty = {helicity_infty[t_idx]:.6f} +/- {helicity_infty_err[t_idx]:.6f}")

    # 绘图
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.errorbar(T_common, helicity_infty, yerr=helicity_infty_err,
                color="red", fmt="^", markersize=3, capsize=2, lw=0.6,
                label=r"$\Upsilon_\infty$ (extrapolated)")

    ax.axvline(args.T_KT_theory, color="gray", lw=0.6, linestyle=":",
               label=rf"$T_{{\mathrm{{KT}}}} \approx {args.T_KT_theory}$")

    ax.set_xlabel("Temperature")
    ax.set_ylabel(r"Helicity modulus at $L=\infty$")
    ax.set_title("Finite-size scaling of helicity modulus")
    ax.legend(fontsize=9)
    ax.set_xlim(T_common[0], T_common[-1])

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
