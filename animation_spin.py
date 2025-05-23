import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import XY

temperature = 1.1 
L = 30          
spin = np.zeros((L, L)) 

fig, ax = plt.subplots(figsize=(8, 8))

quiver = None
vort_plot = []

def init():
    """initialize"""
    global quiver
    ax.clear()
    ax.set_title('XY Model, T='+str(temperature))
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('equal')
    
    dx = 0.5 * np.cos(spin)
    dy = 0.5 * np.sin(spin)
    quiver = ax.quiver(np.arange(L), np.arange(L), dx, dy, 
                      color='black', scale=15, width=0.005)
    return [quiver]

def update(frame):
    global spin, quiver, vort_plot
    
    spin = XY.sweep(temperature, spin, Nsweep=5,cluster=False)
    
    # update quiver
    dx = 0.5 * np.cos(spin)
    dy = 0.5 * np.sin(spin)
    quiver.set_UVC(dx, dy)
    
    # update vortices
    # remove old vortices
    for p in vort_plot:
        p.remove()
    vort_plot.clear()
    
    v = XY.vort(spin)
    for i in range(L):
        for j in range(L):
            if v[i,j] == 1: #positive vortex
                p, = ax.plot(np.array([0,-1,-1,0,0])+i, 
                           np.array([0,0,-1,-1,0])+j, 
                           color='red', lw=1)
                vort_plot.append(p)
            elif v[i,j] == -1: #negative vortex
                p, = ax.plot(np.array([0,-1,-1,0,0])+i,
                           np.array([0,0,-1,-1,0])+j,
                           color='blue', lw=1)
                vort_plot.append(p)
    
    return [quiver] + vort_plot

# Animate
ani = FuncAnimation(
    fig,
    update, 
    init_func=init,
    frames=20000,       
    interval=1,      
    blit=True
)

plt.show()