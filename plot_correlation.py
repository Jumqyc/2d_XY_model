import XY
import numpy as np
import matplotlib.pyplot as plt

fig,axs = plt.subplots(1,3)
def fig(t,figloc):
    #generate a G(r) plot at temprture T locate at figloc
    def corr(t,L,color,fmt):
        Ntest = 50
        G = np.zeros((L//2,Ntest))
        spin = np.zeros((L,L),dtype=float)
        for n in range(Ntest):
            spin = XY.independent_spin(t,spin)
            for i in range(L//2):
               G[i,n] = np.mean(np.cos(spin-np.roll(spin,axis=0,shift=i)))+np.mean(np.cos(spin-np.roll(spin,axis=1,shift=i)))
            G[:,n] = G[:,n]/G[0,n]

        axs[figloc].errorbar(np.arange(1,L//2),np.mean(G,axis=1)[1:],yerr=XY.Confidence_interval(G,axis=1)[1:],label='Size ='+str(L),color = color,lw = 0.4,capsize = 1.2,marker = fmt,markersize = 2)

    color = ['red','green','blue','brown']
    fmt = ['o','^','x','2']
    for i,L in enumerate([16,32,64,128]):
        corr(t,L,color[i],fmt[i])

    axs[figloc].set_xscale('log')
    axs[figloc].set_yscale('log')
    axs[figloc].set_title('Correlation function at $T=$'+str(t))
    axs[figloc].set_xlabel('Distance $r$')

fig(0.1,0)
print('Finished 33%')
fig(0.7,1)
print('Finished 66%')
fig(1.3,2)
print('Finished 100%!')

axs[0].legend()
plt.tight_layout()
plt.show()