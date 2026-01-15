import numpy as np
from Diagnostics import velExact, pExact

def velIC(X, Y,nu,nProb=0):
    """
    Initial condition for velocity field
    
    :param X: x coordinates
    :param Y: y coordinates 
    :param nu: viscosity
    :param nProb: problem number
    """
    Nxb,Nyb=X.shape
    Nx=Nxb-2
    Ny=Nyb-2
    t=0.0
    u=np.zeros_like(X)
    v=np.zeros_like(Y)
    match nProb:
        case 0:  #  Exact solution in periodic box
            u,v=velExact(X, Y,t, nu, nProb)
        case 1:  #  Kelvin-Helmholtz type initial condition  from Minion and Brown (1997)
            rho = 30.0
            delta = 0.05
            NyOver2=int(Ny/2)
            u[:,:NyOver2] = np.tanh((Y[:,:NyOver2]-0.25)*rho)
            u[:,NyOver2:] = np.tanh((0.75-Y[:,NyOver2:])*rho)
            v = delta*np.sin(2.0*np.pi*X)
        case _:  # Default zero initial condition
            u = np.zeros_like(X)
            v = np.zeros_like(Y)
    return u,v

def pIC(X, Y,nu,nProb=0):
    """
    Initial condition for pressure field
    
    :param X: x coordinates
    :param Y: y coordinates 
    :param nu: viscosity
    :param nProb: problem number
    """
    freqX = 1.0*2.0*np.pi
    freqY = 0.25*2.0*np.pi
    t=0.0

    match nProb:
        case 0:  #  Exact solution in periodic box
            p = pExact(X, Y,t,nu,nProb)
        case _:  # Default zero initial condition
            p = np.zeros_like(X)
    return p
