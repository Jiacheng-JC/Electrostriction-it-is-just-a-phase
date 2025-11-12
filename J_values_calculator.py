# -*- coding: utf-8 -*-
"""
Created on Wed May 28 13:31:07 2025

@author: Jiacheng YU
"""


import warnings
import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt 
from lmfit import Minimizer, Parameters, report_fit

# deriv function is to calculate the second derivative of displacement in the potential dVdx, with a damping c and sinusoidal driving force
def deriv(X, t, F, c, omega):
    """Return the derivatives dx/dt and d2x/dt2."""

    x, xdot = X
    xdotdot = -dVdx(x) -c * xdot + F * np.sin(omega*t)
    return xdot, xdotdot

# power_sin function is to calculate the amplitude of the second-order term sin^2(x) by lmfit as a function of time
def power_sin(params, x, data):
    a1 = params['amp1']
    a2 = params['amp2']
    b2 = params['shift2']
    model = a2*(np.sin(x*omega - b2))**2  + a1 
    return model - data
# solve function is to iterate the derivative of X, a list storing x and xdot, using odeint 
def solve(tmax, dt_per_period, t_trans, x0, v0, F, c, omega):
    period = 2*np.pi/omega
    dt = 2*np.pi/omega / dt_per_period
    step = int(period / dt)
    t = np.arange(0, tmax, dt)
    # Initial conditions: x, xdot
    X0 = [x0, v0]
    X = odeint(deriv, X0, t, args=(F, c, omega))
    idx = int(t_trans / dt)
    return t[idx:], X[idx:], dt, step


# initial values

x0, v0 = 0, 0
X0 = [x0, v0]
dt_per_period = 100
asheet=list()
aJ_list=list()


# parameters a,k,F,c,omega can be varied

a = np.logspace(-1, 0, 20)
k = np.logspace(0, 2, 20)
F = np.linspace (0.4, 0.8, 1)


# sweeping all the parameters

for i in range (len(a)):
    for j in range(len(k)):
        for m in range (len(F)):
            
            
            # potential related
            dVdx = lambda x: a[i]*x**2 + k[j]*x
            Fr, c = F[m], 0.01
            Fl, c = -Fr, c
            
            
            # frequencies related
            omega = 1e-11*2*np.pi
            tmax, t_trans = 2*np.pi/omega, 0
            dt = 2*np.pi/omega / dt_per_period
            t = np.arange(0, 100*tmax, dt)
            E = np.sin(omega*t)
            
            
            # solve x and skip the numerically unstable solutions
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")  # catch all warnings
            
                    Xr = odeint(deriv, X0, t, args=(Fr, c, omega))
                    Xl = odeint(deriv, X0, t, args=(Fl, c, omega))
                    xr, xrdot = Xr.T
                    xl, xldot = Xl.T
                    x = xr + xl
                    # Check if any of the warnings were about excess work
                    for warning in w:
                        if "Excess work done on this call" or "Repeated error test failures (internal error)" in str(warning.message):
                            print(f"Skipping (a={a[i]}, k={k[j]}, F={F[m]}) due to ODEint warning")
                            raise RuntimeError("ODEint integration warning")
            
            except RuntimeError:
                continue
             
            # obtain x2
            params = Parameters()
            params.add('amp1', value=0)
            params.add('amp2', value=1e-2)
            params.add('shift2', value=0)
            minner = Minimizer(power_sin, params, fcn_args=(t[-200:],x[-200:]))
            result = minner.minimize()
            final = x[-200:] + result.residual
            x2 = result.last_internal_values[1]
            
            
            # r2 evaluation 
            rss = (result.residual**2).sum()
            tss = sum(np.power(final - np.mean(final), 2)) 
            r2 = 1 - rss/tss
            
            # Calculation of J
            J = -x2*k[j]**3/(a[i]*Fr**2)
            
            if k[j]<a[i] :
                print (k[j], a[i], F[m], "risks of the escape of the potential well")
            
            elif r2<0.9 or r2>1 :  
                print (k[j], a[i], F[m], "bad fit")
            
            elif abs(x2)>700:
                print (k[j], a[i], F[m], "x2 is in the wrong range")
                       
            elif omega/(2*np.pi)>1e-2*np.sqrt(k[j]):
                print (k[j], a[i], F[m], "risks of resonance or relaxation")
                continue            
            else:    
                asheet.append((k[j], a[i], F[m], x2, J))

                aJ_list.append(J)
