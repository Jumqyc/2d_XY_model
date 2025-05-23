import numpy as np
from numba import njit

@njit() #Core function
def build_cluster(t,proj:np.ndarray,x0:np.int64,y0:np.int64,r:np.ndarray)->np.ndarray:
    stack:list = [(x0,y0)]
    n:int = 0
    a,b=np.shape(proj)

    in_cluster:np.ndarray = np.zeros((a,b),dtype=np.bool_) 
    in_cluster[x0,y0] = True

    while stack: #while the stack is not empty,
        x,y = stack.pop()
        for nx,ny in [((x-1)%a,y),((x+1)%a,y),(x,(y-1)%b),(x,(y+1)%b)]: # nn site
            n +=1
            if not in_cluster[nx, ny]:
                if r[n] < np.exp((-2/t)* proj[x, y] * proj[nx,ny]) : 
                    in_cluster[nx, ny] = True  
                    stack.append((nx, ny))
    return in_cluster

# use this to sweep faster
def sweep(t:float,spin,Nsweep:int=50)->np.ndarray:
    a,b = spin.shape
    r = np.random.uniform(size= 4*np.size(spin))
    for n in range(Nsweep):
        phi = np.random.uniform(-np.pi,np.pi)
        x = np.random.randint(a)
        y = np.random.randint(b)
        proj = np.cos(spin - phi)
        in_cluster = build_cluster(t,proj,x,y,r)
        spin = np.where(in_cluster,(np.pi+2*phi)-spin,spin)
    return spin

#######################################################################

def acf(arr,axis=0): #calculate C_auto(tau) along some axis.
        arr = np.moveaxis(arr, axis, 0)
        total_sampling = np.size(arr, axis =0)
        t_max = total_sampling//2
        arr = arr - np.mean(arr, axis=axis, keepdims=True) #convert it into variance

        variance = np.mean(arr**2, axis=axis)  
        acfarr = []
        for n in range(t_max):
            acfarr.append( np.mean(arr[:total_sampling-n] * arr[n:],axis=axis) / variance)
        return np.array(acfarr)

def Magnetization(spin:np.ndarray,vecform:bool = True): #returns Magnetization
    # if vecfrom = True, then return component. Else return modular
    avg_x:float = np.average(np.cos(spin))
    avg_y:float = np.average(np.sin(spin))
    if not vecform:
        return np.sqrt(avg_x**2+avg_y**2)
    else:
        return [avg_x,avg_y]

def Energy(spin): #returns energy
    roll_x = np.roll(spin,1,axis=0)
    roll_y = np.roll(spin,1,axis=1)
    return -(np.average(np.cos(roll_x-spin))+np.average(np.cos(roll_y-spin)))


def Confidence_interval(arr,axis):
    std = np.std(arr,axis=axis)
    return 1.96*std/np.sqrt(np.size(arr,axis=axis)-1)


def rtime(L:int,Observable,cluster:bool,t:float=1,vecform = True):
    #returns relaxation time for Observable. 
    O = np.zeros(2000)
    for n in range(2000):
        spin,num_flip = sweep(t,spin = np.zeros((L,L)),is_num_flip=True,cluster= cluster)
        O[n] = Observable(spin)
    
    if vecform:
        O_acf = np.sum(acf(O),axis=1)
    else:
        O_acf = acf(O)

    ind = 0
    for n in O_acf:
        ind+=1
        if n <0:
            break
    return np.sum(O_acf[:ind])*num_flip/2000


from statsmodels.tsa.stattools import acf as stacf
from scipy.stats import ks_2samp

def is_equilibrium(data: np.array, 
                     auto_corr_thresh: float = 0.1,
                     stat_threshold: float = 0.1,
                     p_threshold: float = 0.01,
                     min_data_points: int = 50) -> bool:
    # auto-corr analysis
    nlags = len(data)
    if nlags < 50:
        return False 
    #return false if it is too short
    #normally min-data-points = 500 
    #saves time since the program quits in the first part and do not do any time-consuming calculation
        
    acf = stacf(data, nlags=nlags, fft=False)
    acf_abs = np.abs(acf)
    lag_candidates = np.where(acf_abs <= auto_corr_thresh)[0]
    if len(lag_candidates) == 0:
        return False
    lag = lag_candidates[0]
    if lag < 1:
        return False
    # slicing
    sampled_data = data[::lag]
    
    # Make sure there is enough sample
    if len(sampled_data) < min_data_points:
        return False
    
    # separate into two parts
    split_idx = len(sampled_data) // 2
    part1 = sampled_data[:split_idx]
    part2 = sampled_data[split_idx:]
    
    # KS sampling
    stat, p_value = ks_2samp(part1, part2)
    
    # returns bool
    return p_value >= p_threshold or stat <= stat_threshold


def independent_spin(t,spin): #output a spin configuration that is independent of the input one
    M=[]
    for m in range(30000): #quit the loop and output the result if the loop is too long...
        spin = sweep(t,spin,Nsweep=50) 
        M.append(Magnetization(spin,vecform=False))
        if is_equilibrium(np.array(M)): 
                        #print('Quit the loop successfully')
                        #print(m)
            break
    return spin

def cutoff(arr,epsilon = 10*(-8)):
    return np.where(arr<epsilon,epsilon,arr)


def vort(spin):
    v = np.zeros_like(spin,dtype=int)
    ds_x = spin - np.roll(spin,axis=0,shift=1)
    ds_y = spin - np.roll(spin,axis=1,shift=1)
    def normalize(spin):
        return (spin / (2*np.pi)+1/2)%1
    ds_x = normalize(ds_x)
    ds_y = normalize(ds_y)

    v = ds_x - np.roll(ds_x,axis= 1,shift= 1) - ds_y + np.roll(ds_y,axis= 0,shift=1)
    return np.round(v,2)
