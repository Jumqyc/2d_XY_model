import numpy as np
import matplotlib.pyplot as plt
import XY

def thermal(t,spin,Ntest):
    a,b = spin.shape
    M = np.zeros((Ntest),dtype=float)
    chi = np.zeros((Ntest),dtype=float)
    E = np.zeros((Ntest),dtype=float)
    C_v = np.zeros((Ntest),dtype=float)
    spin = XY.sweep(t,spin,Nsweep=2000) 
    for n in range(Ntest):
        Magnetization = []
        Energy = []
        for m in range(10000): #quit the loop and output the result if the loop is too long...
            spin = XY.sweep(t,spin,Nsweep=50) 
            Magnetization.append(XY.Magnetization(spin,vecform=False))
            Energy.append(XY.Energy(spin))
            if XY.is_equilibrium(np.array(Energy)): 
                # if is equilibrium, then quit
                if XY.is_equilibrium(np.array(Magnetization)): 
                    #print('Quit the loop successfully')
                    #print(m)
                    break

        Magnetization = np.array(Magnetization)
        Energy = np.array(Energy)

        M[n] = np.mean(Magnetization)
        chi[n] = (np.mean(Magnetization**2))/(t)
        E[n] = np.mean(Energy)
        C_v[n] = (a*b)*(np.mean(Energy**2)-np.mean(Energy)**2)/(t**2)
    
    return spin,M,chi,E,C_v


######Now for plotting
fig, axs = plt.subplots(2,2)

def plot_thermal(temperature,L,Ntest,color,fmt):
    #plot the thermodynamical quantities at given temperature, 
    # with size L spin, and take Ntest for every datapoint. 
    # Can assign the plotting color and fmt can
    t_num = len(temperature)
    spin = np.zeros((L,L),dtype=float)
    spin = XY.sweep(temperature[0],spin,Nsweep=2000) 
    #generates the spin field
    M = np.zeros((t_num,Ntest),dtype=float)
    chi = np.zeros((t_num,Ntest),dtype=float)
    E = np.zeros((t_num,Ntest),dtype=float)
    C_v = np.zeros((t_num,Ntest),dtype=float)

    swtemp = np.linspace(0,0.8,300)

    for tind,t in enumerate(temperature): 
        spin,M[tind,:],chi[tind,:],E[tind,:],C_v[tind,:] = thermal(t,spin,Ntest = Ntest)
        print('The calculation has go to','size =',L,'temprture=',t)

#####First plotting, avg(M)
    axs[0,0].errorbar(temperature,np.mean(M,axis=1),yerr = XY.Confidence_interval(M,axis=1),label='Size ='+str(L),color = color,lw = 0.4,capsize = 1.2,fmt = fmt,markersize = 2)
    axs[0,0].plot(swtemp,1-swtemp*(np.log(2*2.2*L**2))/(8*np.pi),color = color,lw = 0.4)

##second plotting, chi
    axs[0,1].errorbar(temperature,L*np.mean(chi,axis=1),yerr = L*XY.Confidence_interval(chi,axis=1),color = color,lw = 0.4,capsize = 1.2,fmt = fmt,markersize = 2)
    axs[0,1].plot(swtemp[2:],L*((2.2*L)**(-swtemp[2:]/(2*np.pi))/swtemp[2:]),color=color,lw = 0.4,label = str(L//16)+'$\\times$'+'data')
    axs[0,1].set_yscale('log')
###Third plotting, Energy

    axs[1,0].errorbar(temperature,np.mean(E,axis=1),yerr = XY.Confidence_interval(E,axis=1),color = color,lw = 0.4,capsize = 1.2,fmt = fmt,markersize = 2)
    axs[1,0].plot(swtemp,(-2+swtemp/2),color=color,lw = 0.4)

#####Fourth plotting, C_v

    axs[1,1].errorbar(temperature,np.mean(C_v,axis=1),yerr = XY.Confidence_interval(C_v,axis=1),color = color,lw = 0.4,capsize = 1.2,fmt = fmt,markersize = 2)
    axs[1,1].plot(swtemp,np.array([1/2]*len(swtemp)),color=color,lw = 0.4)

plot_thermal(np.linspace(0.05,1.4,28)+0.1/5,16,10,'m','1')
plot_thermal(np.linspace(0.05,1.25,25)+0.05/5,32,10,'blue','*')
plot_thermal(np.linspace(0.05,1.2,24),64,10,'red','^')
plot_thermal(np.linspace(0.05,1.15,23)-0.05/5,128,10,'green','o')
plot_thermal(np.linspace(0.05,1.1,22)-0.1/5,256,10,'brown','d')



axs[0,0].set_xlim(0,1.4)
axs[1,0].set_xlim(0,1.4)
axs[0,1].set_xlim(0,1.4)
axs[1,1].set_xlim(0,1.4)

axs[1,1].set_ylim(0,3)


axs[0,0].set_title('Magnetization')
axs[0,1].set_title('Susceptibility')
axs[1,0].set_title('Average energy per site')
axs[1,1].set_title('Heat Capacitance')

axs[1,0].set_xlabel('Temperature')
axs[1,1].set_xlabel('Temperature')

axs[0,0].legend()
axs[0,1].legend()


plt.show()
