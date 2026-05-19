import XY
import numpy as np
import matplotlib.pyplot as plt

def plot_corr_eta(temperature,L,color,fmt,Ntest=50):
    # plot eta using correlation function, 
    # with size L spin, and take Ntest for every datapoint. 
    # Can assign the plotting color and fmt can
    t_num = len(temperature)
    spin = np.zeros((L,L),dtype=float)
    spin = XY.sweep(temperature[0],spin,Nsweep=2000)
    #generates the spin field
    eta = np.zeros((t_num,Ntest),dtype=float)
    # first we initialize the C(r)
    def C(x,y):
        kx_val = np.linspace(-np.pi,np.pi,L+1)[1:]
        ky_val = np.linspace(-np.pi,np.pi,L+1)[1:]
        exponent = (x*kx_val)[:,np.newaxis]+(y*ky_val)
        denominator = 4-2*np.cos(kx_val)[:,np.newaxis]-2*np.cos(ky_val)
        denominator = XY.cutoff(denominator)
        ### cutoff the denominator
        return np.real(np.nansum((np.exp(1j*exponent)-1)/denominator)/L**2)
    
    C_arr = np.zeros((L,L),dtype=float)
    for i in range(L):
        for j in range(L):
            C_arr[i,j] = C(i,j)    
    
    C_arr = XY.cutoff(C_arr)

    def estimate_eta(spin):
        corr_arr = np.zeros((L,L),dtype=float)
        for i in range(L):
            for j in range(L):
                corr_arr[i,j] = np.mean(np.cos(spin - 
                np.roll(
                    np.roll(spin,shift=-i,axis=0),shift=-j,axis =1)))
        return np.nanmean((np.log(corr_arr)/C_arr))/(2*np.pi)

    for t_ind,t in enumerate(temperature):
        n = 0
        while n<Ntest:
            spin = XY.independent_spin(t,spin)
            try_value = estimate_eta(spin)
            if type(try_value) != ValueError:
                eta[t_ind,n] = try_value
                n+=1
        print('The calculation has go to','size =',L,'temprture=',t,'step')

                
    
    plt.errorbar(temperature,np.mean(eta,axis=1),yerr=XY.Confidence_interval(eta,axis=1),fmt= fmt,color = color,lw = 0.4,capsize=1.2,markersize = 2,label = 'Size = '+str(L))

plt.plot([0.89,0.89],np.array([0,0.5]),lw = 0.4,linestyle = '-.',color = 'black')
plt.plot([0,1],np.array([0.25,0.25]),lw = 0.4,linestyle = '-.',color = 'black')

plot_corr_eta(np.linspace(0.05,1,20),16,'red','^')
plot_corr_eta(np.linspace(0.05,1,20)-0.05/5,32,'green','^')
plot_corr_eta(np.linspace(0.05,1,20)-0.1/5,64,'blue','^')
plot_corr_eta(np.linspace(0.05,1,20)-0.15/5,128,'brown','^')

plt.plot([0,0.89],np.array([0,0.89])/(2*np.pi),lw = 0.4,color = 'red')
plt.xlim(0,1)
plt.ylim(0,0.5)
plt.xticks([0,0.2,0.4,0.6,0.8,0.89,1],[0,0.2,0.4,0.6,0.8,'$T_{\\mathrm{KT}}$',1.0])
plt.yticks([0,0.1,0.2,0.25,0.3,0.4,0.5],[0,0.1,0.2,'$\\eta(T_{\\mathrm{KT}})=\\frac{1}{4}$',0.3,0.4,0.5])






def chi(t,spin,Ntest):
    #direct copy of the previous program
    chi = np.zeros((Ntest),dtype=float)
    spin = XY.sweep(t,spin,Nsweep=2000) 
    for n in range(Ntest):
        Magnetization = []
        Energy = []
        for m in range(10000): #quit the loop and output the result if the loop is too long...
            spin = XY.sweep_fast(t,spin,Nsweep=50) 
            Magnetization.append(XY.Magnetization(spin,vecform=False))
            Energy.append(XY.Energy(spin))
            if XY.is_equilibrium(np.array(Magnetization)): 
                break

        Magnetization = np.array(Magnetization)
        Energy = np.array(Energy)
        chi[n] = (np.mean(Magnetization**2))
    return np.mean(chi)
from scipy.stats import linregress
def fit_power_law(L, t_data):
    log_L = np.log(L)
    log_t = np.log(t_data)  # take average then take log
    # linear fit
    res = linregress(log_L, log_t)
    z = res.slope
    z_error = res.stderr  # standard error of slope
    return z, z_error

tempreture = np.linspace(0.05,1,20)-0.2/5
L = np.array([16,32,48,64,80,100])
chi_arr = np.zeros((len(tempreture),len(L)),dtype=float)
for l_ind, l in enumerate(L):
    spin = np.zeros((l,l),dtype=float)
    for t_ind , t in enumerate(tempreture):
        chi_arr[t_ind,l_ind] = chi(t,spin,Ntest=10)

eta = np.zeros((len(tempreture),2),dtype=float)
for t_ind,t in enumerate(tempreture):
    eta[t_ind,:] = fit_power_law(L,chi_arr[t_ind,:])

plt.errorbar(tempreture,-eta[:,0],yerr=eta[:,1],label='Estimation using $\\chi$',fmt = 'd',lw = 0.4, capsize= 1.2,markersize = 2,color = 'orange')
plt.legend()
plt.show()