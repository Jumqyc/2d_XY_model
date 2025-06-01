import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from collections import namedtuple

class XY:
    def __init__(self,t:float,Lx:int, Ly:int):
        """
        This provides a simulation of the XY model on a 2D lattice.
        Parameters:
        t (float): The temperature of the system.
        Lx (int): The length of the lattice in the x-direction.
        Ly (int): The length of the lattice in the y-direction.
        The spin configuration is initialized to zero.
        Functions:
        sweep(Nsweep:int=50): Perform a number of sweeps to update the spin configuration, using Wolff's algorithm.
        get_spin: Returns the current spin configuration.
        Magnetization(vecform: bool = True): Returns the magnetization of the system. If vecform is True, returns the vector form; otherwise, returns the magnitude.
        Energy: Returns the energy of the system.
        vort: Returns the vortex number of the spin configuration.
        """
        self.Lx :int= Lx
        self.Ly :int= Ly
        self.t :float= t
        self.spin :np.ndarray= np.zeros((Lx, Ly), dtype=float)
        self.num_flip :int= 0

    def __build_cluster(self,proj:np.ndarray)->np.ndarray:
        x0 = np.random.randint(self.Lx)
        y0 = np.random.randint(self.Ly)
        stack:list = [(x0,y0)]
        n:int = 0

        in_cluster:np.ndarray = np.zeros((self.Lx,self.Ly),dtype=np.bool_) 
        in_cluster[x0,y0] = True

        while stack: #while the stack is not empty,
            x,y = stack.pop()
            for nx,ny in [((x-1)%self.Lx,y),((x+1)%self.Lx,y),(x,(y-1)%self.Ly),(x,(y+1)%self.Ly)]: # nn site
                if not in_cluster[nx, ny]:
                    if np.random.uniform() < np.exp((-2/self.t)* proj[x, y] * proj[nx,ny]) : 
                        in_cluster[nx, ny] = True
                        stack.append((nx, ny))
        self.num_flip += np.sum(in_cluster)
        return in_cluster

    def sweep(self,Nsweep:int=50)->None:
        for _ in range(Nsweep):
            phi = np.random.uniform(-np.pi,np.pi)
            proj = np.cos(self.spin - phi)
            in_cluster = self.__build_cluster(proj)
            self.spin = np.where(in_cluster,(np.pi+2*phi)-self.spin,self.spin)

    @property
    def get_spin(self)->np.ndarray:
        return self.spin
    
    @property
    @njit
    def Magnetization(self, vecform: bool = True):  
        """ 
        Returns Magnetization.
        If vecform = True, then return component. Else return its modular
        """
        avg_x: float = float(np.mean(np.cos(self.spin)))
        avg_y: float = float(np.mean(np.sin(self.spin)))
        if not vecform:
            return np.sqrt(avg_x**2+avg_y**2)
        else:
            return [avg_x,avg_y]
    @property
    def Energy(self):
        """
        Returns the energy of the system.
        """
        roll_x = np.roll(self.spin,1,axis=0)
        roll_y = np.roll(self.spin,1,axis=1)
        return -(np.mean(np.cos(roll_x-self.spin))+np.mean(np.cos(roll_y-self.spin)))

    @property
    def _vort(self) -> tuple:
        """
        Returns the vortex number of the spin configuration.
        The vortex number is defined as the number of times the spin winds around the origin.
        """
        v = np.zeros_like(self.spin,dtype=int)
        ds_x = self.spin - np.roll(self.spin,axis=0,shift=1)
        ds_y = self.spin - np.roll(self.spin,axis=1,shift=1)
        def normalize(spin):
            return (spin / (2*np.pi)+1/2)%1
        ds_x = normalize(ds_x)
        ds_y = normalize(ds_y)
        v = np.round(ds_x - np.roll(ds_x,axis= 1,shift= 1) - ds_y + np.roll(ds_y,axis= 0,shift=1),3)
        vortex = namedtuple('vortex', ['positive_vortex_number', "negative_vortex_number", 'vortex_locations'])
        output = vortex(np.sum(v == 1), np.sum(v==-1), self.spin)
        return output

    @property
    def plot_vort(self):
        modular = 0.5
        dx = modular*np.cos(self.spin)
        dy = modular*np.sin(self.spin)
        for i in range(self.Lx):
            for j in range(self.Ly):
                plt.arrow(i,j,dx[i,j],dy[i,j],color = 'black',head_width=0.3, head_length=0.5,lw = 0.2 )
        v = self._vort.vortex_locations
        for i in range(self.Lx):
            for j in range(self.Ly):
                if v[i,j] == 0:
                    continue
                elif v[i,j] == 1:
                    plt.plot(np.array([0,-1,-1,0,0])+i,np.array([0,0,-1,-1,0])+j,color = 'red')
                elif v[i,j] == -1:
                    plt.plot(np.array([0,-1,-1,0,0])+i,np.array([0,0,-1,-1,0])+j,color = 'blue')
        plt.axis('equal')
        plt.show()
    
    def correlation(self, Nsample: int):
        G = np.zeros((self.Lx, self.Ly, Nsample))
        for n in range(Nsample):
            self.sweep(Nsweep=50)
            for i in range(self.Lx):
                for j in range(self.Ly):
                    G[i, j, n] = np.mean(np.cos(self.spin - np.roll(np.roll(self.spin, axis=0, shift=i), axis=1, shift=j)))
        G = np.mean(G, axis=2)
        G = G / G[0, 0]
        return G

    
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

