import numpy as np
from Diagnostics import velExact, pExact, err_plots,velPlots, surfc
from Operators import timestep
from InitialConditions import velIC, pIC


# Create 2D coordinate grids on the unit box

N = 128 # grid resolution (change as needed)
dx = 1.0/N      # grid spacing
Nover2=int(N/2)
N2=int(2*N) 

# 1D arrays giving x and y coordinates in the unit box
x = np.linspace(0-dx/2, 1+dx/2, N+2)
y = np.linspace(0-dx/2, 1+dx/2, N+2)
xedge = np.linspace(0, 1, N+1)
xcent = np.linspace(dx/2, 1-dx/2, N)

X, Y = np.meshgrid(x, y, indexing='ij')

nProb=1
nu=0.002

# Initial conditions
u0,v0=velIC(X,Y,nu,nProb)
p0=pIC(X,Y,nu,nProb)

#  Choose "CFL" 
CFL=0.5

# Compute time step
dt = CFL*dx/np.max(np.sqrt(u0**2+v0**2))

# Do a nSteps time steps    
nSteps=256

un=u0.copy()
vn=v0.copy()
pn=p0.copy()
t=0.0
for n in range(nSteps):     
    unp,vnp,pnp=timestep(X,Y,un,vn,pn,nu,t,dt,nProb)
    t=t+dt
    un=unp.copy()
    vn=vnp.copy()
    pn=pnp.copy()

velPlots(X,Y,unp,vnp, tstr=' After time step ' +str(nSteps) + ' at t='+str(t))
surfc(X,Y,pn, stitle=' Pressure after time step ' +str(nSteps) + ' at t='+str(t))
if(nProb==0):
    pexnp=pExact(X,Y,t-dt/2,nu,nProb)
    perr=err_plots(X,Y,pn,pexnp,'Pressure after time step')
    uexp,vexp= velExact(X,Y,t,nu,nProb)
    uerr=err_plots(X,Y,unp,uexp,'u after time step')
    verr=err_plots(X,Y,vnp,vexp,'v after time step')
