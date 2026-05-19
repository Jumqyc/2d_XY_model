from functools import wraps
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
import pickle as pkl
from typing import Callable

dumping_area = './data'

def jackknife_estimator(func)->Callable:
    @wraps(func)
    def wrapper(d):
        t:float = d['t']
        L:int = d['L']
        m:np.ndarray = d['m']
        e:np.ndarray = d['e']
        h:np.ndarray = d['h']

        length = len(m)
        binsize = 128
        nbin = length//binsize
        est = np.zeros(nbin)
        for i in range(nbin):
            mask = np.array(
                [True if i*binsize<n<(i+1)*binsize else False 
                 for n in range(length)]
            )
            est[i] = func(t,L,m[mask],e[mask],h[mask])
        avg = np.mean(est)
        err = np.sqrt((nbin-1)*np.mean((est-avg)**2))
        return avg,err
    return wrapper

@jackknife_estimator
def avg_m(t,L,m,e,h):
    return np.mean(m)
@jackknife_estimator
def avg_e(t,L,m,e,h):
    return np.mean(e)
@jackknife_estimator
def avg_h(t,L,m,e,h):
    return np.mean(h)
@jackknife_estimator
def susceptibility(t,L,m,e,h):
    return np.average(m**2)/t
@jackknife_estimator
def specific_heat(t,L,m,e,h):
    return L**2*np.var(e)/t**2

def helper(folder,f):
    t_vals = []
    avgs,errs = [],[]
    size = int(folder.split('=')[-1])
    for file in os.listdir(f'{dumping_area}/{folder}'):
        with open(f'{dumping_area}/{folder}/{file}','rb') as reader:
            d:dict[str,np.ndarray] = pkl.load(reader)
            avg,err = f(d)
        avgs.append(avg)
        errs.append(err)
        t_vals.append(d['t'])
    order = np.argsort(t_vals)
    t_vals = np.array(t_vals)[order]
    avgs = np.array(avgs)[order]
    errs = np.array(errs)[order]
    return t_vals,avgs,errs,size

def plot(f:Callable[...,tuple[float,float]]):
    with Pool() as p:
        results = p.starmap(helper, [(folder, f) for folder in os.listdir(dumping_area)])
    results.sort(key=lambda x:x[3])
    for t_vals,avgs,errs,size in results:
        plt.errorbar(t_vals,avgs,yerr=errs,label=f'L={size}',fmt = 'o',capsize=3,markersize=3,capthick=1,lw = 2)
    plt.legend()
    plt.xlabel('T')
    plt.yscale('log')
    plt.show()

plot(specific_heat)