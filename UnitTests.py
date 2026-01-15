from MultiGrid import interp, reflectbc
import numpy as np
from Diagnostics import *
from Operators import timestep

""""""

N = 256 # grid resolution (change as needed)
dx = 1.0/N      # grid spacing
Nover2=int(N/2)
N2=int(2*N) 

# 1D arrays giving x and y coordinates in the unit box
x = np.linspace(0-dx/2, 1+dx/2, N+2)
y = np.linspace(0-dx/2, 1+dx/2, N+2)
xedge = np.linspace(0, 1, N+1)
xcent = np.linspace(dx/2, 1-dx/2, N)

#  Two dimensional meshgrid arrays for the unit box
X, Y = np.meshgrid(x, y, indexing='ij')
Xu,Yu=np.meshgrid(xedge, xcent, indexing='ij')
Xv,Yv=np.meshgrid(xcent, xedge, indexing='ij')
Xcent,Ycent=np.meshgrid(xcent, xcent, indexing='ij')

#print(Xu.shape,Yu.shape,Xv.shape,Yv.shape)

# Xcoarse, Ycoarse are a coarsened grid version 
xcoarse = np.linspace(0-dx, 1+dx, Nover2+2)
ycoarse = np.linspace(0-dx, 1+dx, Nover2+2)
Xcoarse, Ycoarse = np.meshgrid(xcoarse, ycoarse, indexing='ij')
# Xfine, Yfine are a fine grid version 
xfine = np.linspace(0-dx/4, 1+dx/4, N2+2)
yfine = np.linspace(0-dx/4, 1+dx/4, N2+2)
Xfine, Yfine = np.meshgrid(xfine, yfine, indexing='ij')

u=uExact(X,Y)
#u = reflectbc(u,N,dx)
uex=uExact(X,Y)
uexc=uExact(Xcoarse,Ycoarse)
uexf=uExact(Xfine,Yfine)

# test the coarsening 
errCoarsening=test_coarsening(uex,uexc,Xcoarse,Ycoarse, do_plots=False)

# test the interpolation
ufine=interp(uex)
ufine = reflectbc(ufine,N2,dx/2)
print('Max difference after interp:', interior_norm(ufine-uexf))
#err_plots(Xfine,Yfine,ufine,uexf,'After Interp')

# test the coarsening and interpolation combo
#ucf = interp(ucoarse)
#ucf = reflectbc(ucf,N,dx)
#print('Max difference after coarsening and interp:', interior_norm(ucf-uex))
#err_plots(X,Y,ucf,uex, 'After coarsening and interp')

# Test the Laplacian
errLap=test_Laplacian(X,Y, nProb=0, doPlots=False)

#  Test a projection
errProj=test_proj(X,Y,Xu,Yu,Xv,Yv,N,doPlots=False)

# Test a diffusion solve
errDiffusion=test_diffusion(X,Y,Xu,Yu,Xv,Yv,N,doPlots=False)

nu=0.01
# Do a time step
nProb=0
u0,v0=velExact(X,Y,0.0,nu,nProb)
p0=pExact(X,Y,0.0,nu,nProb)
dt = 0.8/N
t=0.0
unp,vnp,pnp=timestep(X,Y,u0,v0,p0,nu,0.0,dt,nProb)

pexnp=pExact(X,Y,t-dt/2,nu,nProb)
perr=err_plots(X,Y,pnp,pexnp,'Pressure after time step')
uexp,vexp= velExact(X,Y,dt,nu,nProb)
uerr=err_plots(X,Y,unp,uexp,'u after time step')
verr=err_plots(X,Y,vnp,vexp,'v after time step')
