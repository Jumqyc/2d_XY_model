#!/usr/bin/env python3
"""
Data generation script (multi-process parallel): Monte Carlo simulation
using the C++ accelerated XY module.

Storage layout: data/{prefix}_L{L}/T{T:.4f}.pkl  -- one file per (L, T).
Files contain only raw observable sequences; no averaging / variance
pre-processing is performed.

Usage:
    python generate_data.py
    python generate_data.py --L 16 32 64 --Tmin 0.1 --Tmax 1.4 --nT 30 --Ntest 100
    python generate_data.py --workers 4
"""

import numpy as np
import pickle
import os
import sys
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")


def thermalize(model, spacing, hot_runs):
    for _ in range(hot_runs):
        model.run(spacing)


def simulate_one_TL(args_tuple):
    """
    Run Ntest independent samples for a single (T, L) and save the raw
    sequences to a pkl file.

    Arguments: (T, L, Ntest, spacing, hot_runs, out_dir)
    """
    T, L, Ntest, spacing, hot_runs, out_dir = args_tuple

    import XY

    model = XY.XY(float(T), L)
    thermalize(model, spacing, hot_runs)

    # First run to obtain flush_length
    model.run(spacing)
    fl = len(model.get_m())

    # Pre-allocate
    m_all = np.zeros(Ntest * fl, dtype=np.float64)
    e_all = np.zeros(Ntest * fl, dtype=np.float64)
    h_all = np.zeros(Ntest * fl, dtype=np.float64)

    # Store first-run data
    m_all[:fl] = model.get_m()
    e_all[:fl] = model.get_e()
    h_all[:fl] = model.get_h()

    for n in range(1, Ntest):
        model.run(spacing)
        m_all[n * fl:(n + 1) * fl] = model.get_m()
        e_all[n * fl:(n + 1) * fl] = model.get_e()
        h_all[n * fl:(n + 1) * fl] = model.get_h()

    data = {
        "L": L,
        "T": float(T),
        "m": m_all,
        "e": e_all,
        "h": h_all,
        "params": {
            "Ntest": Ntest,
            "spacing": spacing,
            "hot_runs": hot_runs,
            "flush_length": fl,
        },
    }

    fname = os.path.join(out_dir, f"T{T:.4f}.pkl")
    with open(fname, "wb") as f:
        pickle.dump(data, f)

    return L, float(T)


def main():
    parser = argparse.ArgumentParser(description="XY model data generator (parallel)")
    parser.add_argument("--L", type=int, nargs="+", default=[16, 32, 64, 96,128,196, 256])
    parser.add_argument("--Tmin", type=float, default=0.1)
    parser.add_argument("--Tmax", type=float, default=1.4)
    parser.add_argument("--nT", type=int, default=28)
    parser.add_argument("--Ntest", type=int, default=30)
    parser.add_argument("--spacing", type=int, default=40)
    parser.add_argument("--hot", type=int, default=5)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--prefix", type=str, default="thermal")
    args = parser.parse_args()

    workers = args.workers or multiprocessing.cpu_count()
    temperatures = np.linspace(args.Tmin, args.Tmax, args.nT)

    print(f"T in [{args.Tmin}, {args.Tmax}], {args.nT} points")
    print(f"L = {args.L}")
    print(f"Ntest = {args.Ntest}, spacing = {args.spacing}, hot = {args.hot}")
    print(f"Workers = {workers}  (total tasks: {len(args.L) * args.nT})")
    print()

    # Pre-create all output directories
    for L in args.L:
        d = os.path.join(DATA_DIR, f"{args.prefix}_L{L}")
        os.makedirs(d, exist_ok=True)

    # Build task list (skip already existing files)
    tasks = []
    for L in args.L:
        out_dir = os.path.join(DATA_DIR, f"{args.prefix}_L{L}")
        for T in temperatures:
            fname = os.path.join(out_dir, f"T{T:.4f}.pkl")
            if os.path.exists(fname):
                continue
            tasks.append((float(T), L, args.Ntest, args.spacing, args.hot, out_dir))

    if not tasks:
        print("All files already exist. Nothing to do.")
        return

    total = len(tasks)
    cached = len(args.L) * args.nT - total
    print(f"Submitting {total} tasks ({cached} already cached)...")

    completed = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(simulate_one_TL, t): t for t in tasks}
        for future in as_completed(futures):
            L, T = future.result()
            completed += 1
            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            eta = (total - completed) / rate if rate > 0 else 0
            print(f"[{completed}/{total}] L={L:3d} T={T:.4f}  "
                  f"({rate:.1f} tasks/s, ETA {eta:.0f}s)")

    total_elapsed = time.time() - t_start
    print(f"\nDone in {total_elapsed:.0f}s ({total_elapsed / total:.1f}s/task)")


if __name__ == "__main__":
    main()
