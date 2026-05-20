#!/usr/bin/env python3
"""
Shared analysis module: loads raw observable sequences from
data/{prefix}_L{L}/T*.pkl, splits them into flush blocks, and
computes derived thermodynamic quantities.

pkl file format (one per (L, T), produced by generate_data.py):
    {
        "L": int,
        "T": float,
        "m": np.array (Ntest * flush_length,),   # magnetization magnitude raw seq
        "e": np.array (Ntest * flush_length,),   # energy per site raw seq
        "h": np.array (Ntest * flush_length,),   # per-site sin(dtheta) avg raw seq
        "params": {"Ntest", "spacing", "hot_runs", "flush_length"},
    }
"""

import numpy as np
import pickle
import os
import re
from collections import namedtuple

# Paths
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_MODULE_DIR)
_DATA_DIR = os.path.join(_PROJ_DIR, "data")

# Derived-quantity named tuple
ThermalData = namedtuple("ThermalData", [
    "L",
    "temperatures",
    "M_mean", "M_err",
    "chi", "chi_err",
    "E_mean", "E_err",
    "Cv", "Cv_err",
    "binder", "binder_err",
    "helicity", "helicity_err",
])


def confidence_interval(arr, axis=0):
    """95% confidence interval (normal approximation)."""
    std = np.std(arr, axis=axis)
    n = arr.shape[axis]
    return 1.96 * std / np.sqrt(max(n - 1, 1))


def _compute_from_one_file(raw):
    """
    Compute statistics for a single (L, T) from its raw dict.

    Returns: (M_mean, M_err, chi, chi_err, E_mean, E_err,
              Cv, Cv_err, binder, binder_err, helicity, helicity_err)
    """
    L = raw["L"]
    T = raw["T"]
    N = L * L
    fl = raw["params"]["flush_length"]
    Ntest = raw["params"]["Ntest"]

    m = raw["m"]
    e = raw["e"]
    h = raw["h"]

    # Split into flush blocks; each block is one independent run
    # reshape: (Ntest, fl)
    m_blocks = m.reshape(Ntest, fl)
    e_blocks = e.reshape(Ntest, fl)
    h_blocks = h.reshape(Ntest, fl)

    # Per-block statistics
    m1 = np.mean(m_blocks, axis=1)          # (Ntest,)
    m2 = np.mean(np.square(m_blocks), axis=1)
    m4 = np.mean(np.power(m_blocks, 4), axis=1)
    e1 = np.mean(e_blocks, axis=1)
    e2 = np.mean(np.square(e_blocks), axis=1)
    # helicity per block: -e/2 - (L^2/T)*h^2, averaged over the block
    h2_mean = np.mean(np.square(h_blocks), axis=1)
    hp = -e1 / 2.0 - (N / T) * h2_mean         # (Ntest,)

    # Aggregate: mean and error across the Ntest blocks
    M_mean = float(np.mean(m1))
    M_err = float(confidence_interval(m1))

    E_mean = float(np.mean(e1))
    E_err = float(confidence_interval(e1))

    chi_val = N * float(np.mean(m2 - m1 ** 2)) / T
    chi_err = N * float(confidence_interval(m2 - m1 ** 2)) / T

    Cv_val = N * float(np.mean(e2 - e1 ** 2)) / (T ** 2)
    Cv_err = N * float(confidence_interval(e2 - e1 ** 2)) / (T ** 2)

    binder_val = 1.0 - float(np.mean(m4)) / (3.0 * float(np.mean(m2)) ** 2)
    binder_err = float(a(m4)) / (3.0 * float(np.mean(m2)) ** 2)

    H_mean = float(np.mean(hp))
    H_err = float(confidence_interval(hp))

    return (M_mean, M_err, chi_val, chi_err,
            E_mean, E_err, Cv_val, Cv_err,
            binder_val, binder_err, H_mean, H_err)


def load_thermal_data(L_values, prefix="thermal"):
    """
    Load data for multiple L values, return {L: ThermalData}.

    L_values: list of int
    prefix: data subdirectory prefix
    """
    result = {}
    for L in L_values:
        d = os.path.join(_DATA_DIR, f"{prefix}_L{L}")
        if not os.path.isdir(d):
            print(f"WARNING: directory {d} not found, skipping L={L}")
            continue

        # Collect all T files and sort by temperature
        files = []
        for fname in os.listdir(d):
            m = re.match(r"T([\d.]+)\.pkl$", fname)
            if m:
                files.append((float(m.group(1)), fname))

        if not files:
            print(f"WARNING: no pkl files in {d}, skipping L={L}")
            continue

        files.sort(key=lambda x: x[0])

        n_T = len(files)
        T_arr = np.zeros(n_T)
        M_mean = np.zeros(n_T)
        M_err = np.zeros(n_T)
        chi = np.zeros(n_T)
        chi_err = np.zeros(n_T)
        E_mean = np.zeros(n_T)
        E_err = np.zeros(n_T)
        Cv = np.zeros(n_T)
        Cv_err = np.zeros(n_T)
        binder = np.zeros(n_T)
        binder_err = np.zeros(n_T)
        helicity = np.zeros(n_T)
        helicity_err = np.zeros(n_T)

        for t_idx, (T_val, fname) in enumerate(files):
            fpath = os.path.join(d, fname)
            with open(fpath, "rb") as f:
                raw = pickle.load(f)

            (M_mean[t_idx], M_err[t_idx],
             chi[t_idx], chi_err[t_idx],
             E_mean[t_idx], E_err[t_idx],
             Cv[t_idx], Cv_err[t_idx],
             binder[t_idx], binder_err[t_idx],
             helicity[t_idx], helicity_err[t_idx]) = _compute_from_one_file(raw)

            T_arr[t_idx] = T_val

        result[L] = ThermalData(
            L=L, temperatures=T_arr,
            M_mean=M_mean, M_err=M_err,
            chi=chi, chi_err=chi_err,
            E_mean=E_mean, E_err=E_err,
            Cv=Cv, Cv_err=Cv_err,
            binder=binder, binder_err=binder_err,
            helicity=helicity, helicity_err=helicity_err,
        )

    return result


def load_raw_data(L, prefix="thermal"):
    """
    Load raw per-block statistics for a single L from all T files.

    Returns a dict compatible with scripts that need the internal
    block-statistic arrays (e.g. plot_TKT.py, plot_eta_chi.py).
    """
    d = os.path.join(_DATA_DIR, f"{prefix}_L{L}")
    if not os.path.isdir(d):
        print(f"WARNING: directory {d} not found, skipping L={L}")
        return None

    files = []
    for fname in os.listdir(d):
        m = re.match(r"T([\d.]+)\.pkl$", fname)
        if m:
            files.append((float(m.group(1)), fname))
    files.sort(key=lambda x: x[0])

    if not files:
        print(f"WARNING: no pkl files in {d}, skipping L={L}")
        return None

    first = None
    with open(os.path.join(d, files[0][1]), "rb") as f:
        first = pickle.load(f)
    fl = first["params"]["flush_length"]
    Ntest = first["params"]["Ntest"]

    n_T = len(files)
    M_arr = np.zeros((n_T, Ntest))
    M2_arr = np.zeros((n_T, Ntest))
    M4_arr = np.zeros((n_T, Ntest))
    E_arr = np.zeros((n_T, Ntest))
    E2_arr = np.zeros((n_T, Ntest))
    H_arr = np.zeros((n_T, Ntest))

    T_arr = np.zeros(n_T)

    for t_idx, (T_val, fname) in enumerate(files):
        with open(os.path.join(d, fname), "rb") as f:
            raw = pickle.load(f)

        T_arr[t_idx] = T_val
        m_blocks = raw["m"].reshape(Ntest, fl)
        e_blocks = raw["e"].reshape(Ntest, fl)
        h_blocks = raw["h"].reshape(Ntest, fl)

        M_arr[t_idx, :] = np.mean(m_blocks, axis=1)
        M2_arr[t_idx, :] = np.mean(np.square(m_blocks), axis=1)
        M4_arr[t_idx, :] = np.mean(np.power(m_blocks, 4), axis=1)
        E_arr[t_idx, :] = np.mean(e_blocks, axis=1)
        E2_arr[t_idx, :] = np.mean(np.square(e_blocks), axis=1)

        h2_mean = np.mean(np.square(h_blocks), axis=1)
        H_arr[t_idx, :] = -E_arr[t_idx, :] / 2.0 - (raw["L"] ** 2 / T_val) * h2_mean

    return {
        "L": L,
        "temperatures": T_arr,
        "M": M_arr,
        "M2": M2_arr,
        "M4": M4_arr,
        "E": E_arr,
        "E2": E2_arr,
        "H": H_arr,
    }
