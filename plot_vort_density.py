import XY
import numpy as np
import matplotlib.pyplot as plt

def plot_vort_density(temperature,L,color,fmt,Ntest =30):
    # plot vortex density
    t_num = len(temperature)
    spin = np.zeros((L,L),dtype=float)
    spin = XY.independent_spin(temperature[0],spin)
    #generates the spin field
    n_v = np.zeros((t_num,Ntest),dtype=float)
    # first we initialize the C(r)
    
    for t_ind,t in enumerate(temperature):
        for n in range(Ntest):
            Magnetization = []
            for m in range(10000): #quit the loop and output the result if the loop is too long...
                spin = XY.sweep(t,spin,Nsweep=50) 
                Magnetization.append(XY.Magnetization(spin,vecform=False))
                n_v[t_ind,n] += np.sum(XY.vort(spin)==1)
                # if is equilibrium, then quit
                if XY.is_equilibrium(np.array(Magnetization)): 
                        #print('Quit the loop successfully')
                        #print(m)
                    n_v[t_ind,n] /= (m*L**2)
                    break
                        
        print('The calculation has go to','size =',L,'temprture=',t)
    
    axs[0].errorbar(temperature,np.mean(n_v,axis=1),yerr=XY.Confidence_interval(n_v,axis=1),fmt= fmt,color = color,lw = 0.4,capsize=1.2,markersize = 2,label = 'Size = '+str(L))
    mu = (-0.5)*np.log(n_v)
    axs[1].errorbar(temperature,np.mean(mu,axis=1)*temperature,yerr=XY.Confidence_interval(mu,axis=1)*temperature,fmt= fmt,color = color,lw = 0.4,capsize=1.2,markersize = 2)

fig,axs = plt.subplots(1,2)

plot_vort_density(np.linspace(0.7,1.5,17)+0.1/5,16,'m','1')
plot_vort_density(np.linspace(0.7,1.4,15)+0.05/5,32,'blue','*')
plot_vort_density(np.linspace(0.7,1.3,13),64,'red','^')
plot_vort_density(np.linspace(0.7,1.3,13)-0.05/5,128,'green','o')
plot_vort_density(np.linspace(0.7,1.3,13)-0.1/5,256,'brown','d')

axs[0].set_xlim(0.65,1.5)
axs[1].set_xlim(0.65,1.5)

axs[0].set_xlabel('Temperature')
axs[1].set_xlabel('Temperature')

axs[0].set_ylabel('Density of vortex pair')
axs[1].set_ylabel('Chemical potential')

axs[0].set_yscale('log')

axs[0].legend()
plt.show()