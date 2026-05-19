import numpy as np
import matplotlib.pyplot as plt
import XY

def plot_spin(spin):
    a,b = spin.shape
    modular = 0.5
    dx = modular*np.cos(spin)
    dy = modular*np.sin(spin)
    for i in range(a):
        for j in range(b):
            plt.arrow(i,j,dx[i,j],dy[i,j],color = 'black',head_width=0.3, head_length=0.5,lw = 0.2 )
def plot_vort(spin):
    a,b = spin.shape
    v = XY.vort(spin)
    for i in range(a):
        for j in range(b):
            if v[i,j] == 0:
                continue
            elif v[i,j] == 1:
                plt.plot(np.array([0,-1,-1,0,0])+i,np.array([0,0,-1,-1,0])+j,color = 'red')
            elif v[i,j] == -1:
                plt.plot(np.array([0,-1,-1,0,0])+i,np.array([0,0,-1,-1,0])+j,color = 'blue')


tempreture = [0.8,0.9,0.95,1,1.05,1.1]
spin = XY.sweep(tempreture[0],np.zeros((50,50)),Nsweep=8000)
a,b = spin.shape

for ind, t in enumerate(tempreture):
    spin = XY.sweep(t,np.zeros((30,30)),Nsweep=8000)

    plt.subplot(2,3,(ind+1))
    plot_spin(spin)
    plot_vort(spin)
    plt.title('Temperature='+str(t))
    plt.axis('equal')
    plt.xlim((0,30))
    plt.ylim((0,30))
    

plt.show()