import numpy as np
import matplotlib.pyplot as plt
import XY
from scipy.stats import linregress

tempreture = np.linspace(0.7,0.9,120)
size = np.arange(4,100,4)
helicity = np.zeros((len(tempreture),len(size)))


def decent(f,x0=0,x1=1): # gradient decent
    gamma = 1
    n = 0
    while True:
        n +=1
        x0,x1 = x1,x1-gamma*(f(x1)-f(x0))/(x1-x0)
        #print(x0)
        if np.abs(x0-x1)<10**-12:
            break

        if n == 10: # first quickly approach the minimum, then reduce the slope and make dedicated result
            gamma/= 2
            n = 0
    return (x1+x0)/2


def fit_helicity(data): #return the fitted helicity at L = infty    
    def r_2(C):
        _, _, r, _, _ = linregress(1/(np.log(size)+C), data)
        return r**2
    C_best = decent(r_2)
    result = linregress(1/(np.log(size)+C_best), data)
    return result.intercept, result.intercept_stderr #returns the helicity


for l_ind, l in enumerate(size): #set up different size 
    spin = np.zeros((l,l))
    
    for t_ind,t in enumerate(tempreture):#sweep temperature
        h = []
        spin = XY.independent_spin(t,spin) # thermalize
        Ntest = 0
        for n in range(30000):
            h.append(-XY.Energy(spin)/(2)
                    -(l*np.mean(np.sin(spin - np.roll(spin,axis=1,shift= 1))))**2/(t))
            spin = XY.sweep(t,spin,Nsweep= 10)
            if n%30:
                if XY.is_equilibrium(np.array(h)):
                    Ntest +=1
                    if Ntest == 10:
                        break
        print('The calculation has go to','size =',l,'temprture=',t,'step')
        helicity[t_ind,l_ind] = np.mean(np.array(h))

helicity_infty = np.zeros_like(tempreture)
helicity_infty_err = np.zeros_like(tempreture)

for t_ind,t in enumerate(tempreture):
    helicity_infty[t_ind],helicity_infty_err[t_ind] = fit_helicity(helicity[t_ind,:])
    print("we fitted the slope at tempreture = "+str(t))


plt.errorbar(tempreture,helicity_infty,yerr=helicity_infty_err,color = 'red',lw = 0.4,capsize = 1.2,fmt = '^',markersize = 2)
plt.plot(tempreture,2*tempreture/np.pi,lw = 0.6,color = 'black')

plt.xlabel('Temperature')
plt.ylabel('helicity modulus at $L=\\infty$')

plt.show()