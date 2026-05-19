import numpy as np
import matplotlib.pyplot as plt
import XY


def T_critical(t,spin,Ntest):
    a,b = spin.shape

    Binder = np.zeros((Ntest),dtype=float)
    Helicity = np.zeros((Ntest),dtype=float)

    spin = XY.independent_spin(t,spin)

    def helicity(spin):
        A = np.mean(np.sin(spin - np.roll(spin,axis=1,shift= 1)))
        return -XY.Energy(spin)/(2)- a**2*A**2 /(t)

    for n in range(Ntest):
        Magnetization = []
        Energy = []
        h = []
        for m in range(10000): #quit the loop and output the result if the loop is too long...
            spin = XY.sweep_fast(t,spin,Nsweep=100) 
            Magnetization.append(XY.Magnetization(spin,vecform=False))
            Energy.append(XY.Energy(spin))
            h.append(helicity(spin))
            if XY.is_equilibrium(np.array(Magnetization)): 
                #print('Quit the loop successfully')
                #print(m)
                break
                    
        Magnetization = np.array(Magnetization)
        Binder[n] = 1-np.mean(Magnetization**4)/(3*np.mean(Magnetization**2))
        Helicity[n] = np.mean(np.array(h))
    
    return spin,Binder,Helicity


######Now for plotting
fig, axs = plt.subplots(1,2)

def plot_critical(temperature,L,Ntest,color,fmt):
    #plot the thermodynamical quantities at given temperature, 
    # with size L spin, and take Ntest for every datapoint. 
    # Can assign the plotting color and fmt can
    t_num = len(temperature)
    spin = np.zeros((L,L),dtype=float)
    spin = XY.sweep(temperature[0],spin,Nsweep=1000) 
    #generates the spin field
    Binder = np.zeros((t_num,Ntest),dtype=float)
    Helicity = np.zeros((t_num,Ntest),dtype=float)



    for tind,t in enumerate(temperature): 
        spin,Binder[tind,:],Helicity[tind,:] = T_critical(t,spin,Ntest = Ntest)
        print('The calculation has go to','size =',L,'temprture=',t)
#####First plotting, avg(M)
    axs[0].errorbar(temperature,np.mean(Binder,axis=1),yerr = XY.Confidence_interval(Binder,axis=1),label='Size ='+str(L),color = color,lw = 0.4,capsize = 1.2,fmt = fmt,markersize = 2)

##second plotting, chi
    axs[1].errorbar(temperature,np.mean(Helicity,axis=1),yerr = XY.Confidence_interval(Helicity,axis=1),color = color,lw = 0.4,capsize = 1.2,fmt = fmt,markersize = 2)
###Third plotting, Energy


plot_critical(np.linspace(0.05,1.1,22)+0.1/5,16,15,'m','1')
plot_critical(np.linspace(0.05,1.1,22)+0.05/5,32,15,'blue','*')
plot_critical(np.linspace(0.05,1.1,22),64,15,'red','^')
plot_critical(np.linspace(0.05,1.1,22)-0.05/5,128,15,'green','o')
plot_critical(np.linspace(0.05,1.1,22)-0.1/5,256,15,'brown','d')


axs[1].plot(np.linspace(0,1.1,3),2*np.linspace(0,1.1,3)/np.pi,color= 'black',lw = 0.6,linestyle = ':')

axs[0].set_xlim(0,1.13)
axs[1].set_xlim(0,1.13)


axs[0].plot([0.887])
axs[1].set_xlim(0,1.13)

axs[0].set_title('Binder ratio')
axs[1].set_title('Helicity modulus')

axs[0].set_xlabel('Temperature')
axs[1].set_xlabel('Temperature')

axs[0].legend()


plt.show()


