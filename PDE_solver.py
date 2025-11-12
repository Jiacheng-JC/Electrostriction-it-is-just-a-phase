# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 12:30:29 2025

@author: Jiacheng YU
"""

import numpy as np 
from scipy.integrate import odeint 
import matplotlib.pyplot as plt 
from lmfit import Minimizer, Parameters, report_fit
import warnings
import time


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
    #a4 = params['amp4']
    #b4 = params['shift4']
    #a6 = params['amp6']
    #b6 = params['shift6']

    #model = a2*(np.sin(x*omega - b2))**2 + a4*(np.sin(x*omega - b4))**4 +a6*(np.sin(x*omega - b6))**6 + a1 
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


# initial values of displacement, velocity, max time, angular frequency omega 
x0, v0 = 0, 0
#tmax, t_trans = 1000, 0
omega = 1e-11*2*np.pi
tmax, t_trans = 2*np.pi/omega, 0

dt_per_period = 100
dt = 2*np.pi/omega / dt_per_period
period = 2*np.pi/omega
step = int(period / dt)
t = np.arange(0, 100*tmax, dt)
X0 = [x0, v0]

amplitude_modified=list()
parameter=list()
phase=list()
predisplacement=list()
start = time.time()
#j = np.logspace(0,5,401)
E = np.sin(omega*t)


# initial values of potential a,k
a=-0.1
k=4

# initial values of driving F,c
dVdx = lambda x: a*x**2 + k*x
Fr, c = 0.8, 1
Fl, c = -Fr, c

# the electrostrictive displacement x=xr+xl
Xr = odeint(deriv, X0, t, args=(Fr, c, omega))
Xl = odeint(deriv, X0, t, args=(Fl, c, omega))
xr, xrdot = Xr.T
xl, xldot = Xl.T
x = xr + xl

# the combination of a,k,F,c may create a chaotic solution, mark it

if (
    np.any(np.isnan(x)) or          # NaN values
    np.any(np.isinf(x)) or          # Infinite values
    np.any(np.abs(x) >= 1e10)        # Exploding values
):
    chaotic = 1
else:
    chaotic = 0



# lmfit of x(t), to obtain the amplitude of x(t), x2

#parameter.append(j[i])
params = Parameters()
params.add('amp1', value=0)
params.add('amp2', value=0.35)
#params.add('amp4', value=0)
#params.add('amp6', value=0)
params.add('shift2', value=0)
#params.add('shift4', value=0)
#params.add('shift6', value=0)

minner = Minimizer(power_sin, params, fcn_args=(t[-200:],x[-200:]))
result = minner.minimize()
# calculate final result
final = x[-200:] + result.residual
x2 = result.last_internal_values[1]
rss = (result.residual**2).sum() # same as result.chisqr 

report_fit(result) 
print(f"RSS/absolute sum of squares (Chi-square) = {rss:3.1f}")

tss = sum(np.power(final - np.mean(final), 2)) 
print(f"TSS = {tss:.1f}")

print(f"R² = {1 - rss/tss:.3f}")

# J constant
J = -x2*k**3/(a*Fr**2)


print(x2, J)

# Plotting
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.yaxis.get_major_formatter().set_powerlimits((0,1))
ax1.plot(t[-200:],E[-200:])
ax2 = ax1.twinx()
ax2.plot(t[-200:],x[-200:], '.', c='r')
ax2.set_ylim(-max(abs(x[-200:]))*1.2, max(abs(x[-200:]))*1.2)
plt.show()


plt.plot(E[-200:],x[-200:])







