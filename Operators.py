import numpy as np
def Laplacian(u):
    """
    Docstring for Laplacian
    
    :param u: function values

    :return: 5-point second-order Laplacian of u

    Notes
        Boundary conditions on u should be set before calling
        This version assumes a box size of 1 with one ghost cell on each side and uniform grid spacing dx
    """

    Nxb, Nyb = u.shape
    Nx=Nxb-2
    Ny=Nyb-2  
    lapU = np.zeros((Nxb, Nyb))
    dx = 1.0/Nx  # grid spacing assuming a unit box
    dxdx=dx*dx  # denominator precomputed

    #  lets use index magic instead of loops
    ii = np.arange(1, Nx+1)
    jj = np.arange(1, Ny+1)
    lapU[np.ix_(ii, jj)] = (-4.0*u[np.ix_(ii, jj)] + u[np.ix_(ii+1, jj)] + u[np.ix_(ii-1, jj)] + u[np.ix_(ii, jj+1)] 
                            + u[np.ix_(ii, jj-1)])/dxdx
    return lapU  # return the Laplacian


def MAC_div(u, v):
    """
    Calculate discrete MAC divergence of a 2D vector field.
    This takes edge centered velocities and returns divergence (cell averaged)
    
    Args:
        u: 2D array of x-component velocities
        v: 2D array of y-component velocities
        dx: grid spacing in x direction
        dy: grid spacing in y direction
    
    Returns:
        div: 2D array of divergence values 
    """
    Nxp,Ny= u.shape
    Nx=Nxp-1
    #print('In MAC_div',Nx,Ny,' u shape ',u.shape,'  v shape ', v.shape)

    dx= 1.0/Nx
    dy = 1.0/Ny

    div = np.zeros((Nx+2,Ny+2))
  
    # Divergence 
    div[1:Nx+1,1:Ny+1] = (u[1:, :] - u[:-1, :]) /dx + (v[:,1:]-v[:,:-1])/dy
    
    return div
def centered_grad(phi):
    """
    Take the centered gradient of phi.  
    This creates cell-centered gradients from cell-averaged data
    
    :param phi: field to take gradient of

    return:  phi_x, Phi_y
    """
    Nxb,Nyb= phi.shape
    Nx=Nxb-2
    Ny=Nyb-2

    dx=1.0/Nx
    dy=1.0/Ny
    phi_x = np.zeros((Nx+2,Ny+2))
    phi_y = np.zeros((Nx+2,Ny+2))
    phi_x[1:-1,1:-1] = (phi[2:,1:-1] - phi[:-2,1:-1])/(2.0*dx)
    phi_y[1:-1,1:-1] = (phi[1:-1,2:] - phi[1:-1,:-2])/(2.0*dy)

    return phi_x, phi_y

def MAC_grad(phi):
    """
    Take the MAC gradient of phi.  
    This creates edge centered gradients from cell-averaged data
    
    :param phi: field to take gradient of

    return:  phi_x, Phi_y
    """
    Nxb,Nyb= phi.shape
    Nx=Nxb-2
    Ny=Nyb-2

    dx=1.0/Nx
    dy=1.0/Ny
    phi_x = np.zeros((Nx+1,Ny))
    phi_y = np.zeros((Nx,Ny+1))
    phi_x = (phi[1:,1:-1] - phi[:-1,1:-1])/dx
    phi_y = (phi[1:-1,1:] - phi[1:-1,:-1])/dy

    return phi_x, phi_y

def MAC_projection(X,Y,ustar,vstar,phi0):
    from MultiGrid import invLapacian
    from Diagnostics import surfc
    """
    Docstring for MAC_projection
    The projection is 
       U = Ustar - Grad(phi)
    where
       Lap(phi) = Div(U)
    with boundary conditions
        Grad(phi).n = Ustar.n
    
    :param ustar: u-velocity to be projected
    :param vstar: v-velocity to be projected
    :phi0:  initial guess for phi

    return: u,v,phi
       u,v: divergence free edge velocity
       phi: scalar potential (cell averaged)
    """

    Nxb,Nyb = phi0.shape
    Nx=Nxb-2
    Ny=Nyb-2

    divstar=MAC_div(ustar,vstar)  # form rhs of Poisson equation

    # solve for correction
    res=divstar-Laplacian(phi0)   #  Rhs is the residual
    cstar=np.zeros((Nx+2,Ny+2))     # Best correction guess is zero
    cstar=invLapacian(cstar, res, maxiter=300, tol=1e-9)

    phi=phi0+cstar
    # Project 
    phi_x,phi_y = MAC_grad(phi)

    u=ustar-phi_x
    v=vstar-phi_y

    return u,v,phi

# Take a time step
def centered_projection(X,Y,ustar,vstar,phi0,Nprob):
    from MultiGrid import invLapacian, periodicBC
    from Diagnostics import surfc
    """
    The projection is 
       U = Ustar - Grad(phi)
    where
       Lap(phi) = Div(U)
    with boundary conditions
        dphi/dn = 0 on all boundaries
    
    :param ustar: u-velocity to be projected
    :param vstar: v-velocity to be projected
    :phi0:  initial guess for phi
    :param Nprob: problem number (for BCs)

    return: u,v,phi
       u,v: divergence centered elocity
       phi: scalar potential (cell averaged)

    Note, this is an "approximate" projection since the velocities are not discretely divergence free after the projection
    """

    Nxb,Nyb = phi0.shape
    Nx=Nxb-2
    Ny=Nyb-2
    dx=1.0/Nx
    dy=1.0/Ny   

    # Take the divergence by averaging to edges and taking MAC divergence
    ustar=periodicBC(ustar)
    vstar=periodicBC(vstar)
    ux = averageToEdge(ustar,'x')
    vy = averageToEdge(vstar,'y')
    divstar= MAC_div(ux, vy)  # form rhs of Poisson equation
    divstar=periodicBC(divstar)
    
    # solve for correction
    #surfc(X,Y, Laplacian(phi0), 'Lap phi0 in centered projection')
    res=divstar-Laplacian(phi0)    #  Rhs is the residual
    #surfc(X,Y, res, 'res in centered projection')
    cstar=np.zeros((Nx+2,Ny+2))     # Best correction guess is zero
    cstar=invLapacian(cstar, res, maxiter=30, tol=1e-9,Nprob=Nprob)

    phi=phi0+cstar

    # Project 
    phi_x,phi_y = centered_grad(phi)

    u=ustar-phi_x
    v=vstar-phi_y

    return u,v,phi


def computeAdvectionMAC(X,Y,u,v,nProb=0):
    """
    Docstring for computeAdvectionMAC
    
    :param X: x coordinates
    :param Y: y coordinates
    :param ux: u velocity on x-edges
    :param vx: v velocity on x-edges
    :param uy: u velocity on y-edges
    :param vy: v velocity on y-edges
    :param Nprob:   problem number (for BCs)
    """
    Nxb,Nyb= X.shape
    Nx=Nxb-2
    Ny=Nyb-2
    advectU = np.zeros((Nx+2,Ny+2))
    advectV = np.zeros((Nx+2,Ny+2))

    #  Compute edge fluxes at time n
    ux,uy,vx,vy = computeEdges(X,Y,u,v,nProb)
    #print('ComputeAdvectionMAC',ux.shape,vx.shape,uy.shape,vy.shape)
    #  Project edge fluxes
    phi0=np.zeros(u.shape)
    uMACx,vMACy,phi = MAC_projection(X,Y,ux,vy,phi0)

    advectU = MAC_div(uMACx*uMACx, uy*vMACy)
    advectV = MAC_div(vx*uMACx, vMACy*vMACy)
    #print('In computeAdvectionMAC',advectU.shape,advectV.shape)
    return advectU, advectV

def averageToEdge(f,dir):
    """
    Docstring for averageToCellCenters
    
    :param f: field to be averaged
    :param dir: direction to average ('x' or 'y')

    return:  fedge: field averaged to edges

    """
    Nxb,Nyb=f.shape
    Nx=Nxb-2
    Ny=Nyb-2
    if dir=='x':
        fedge = np.zeros((Nx+1,Ny))
        fedge = 0.5*(f[:-1,1:-1]+f[1:, 1:-1])
        return fedge
    elif dir=='y':
        fedge = np.zeros((Nx,Ny+1))
        fedge =  0.5*(f[1:-1,:-1]+f[1:-1,1:])
        return fedge
    else:   
        print('Error: direction must be x or y')
        return None


def computeEdges(X,Y,u,v,Nprob=0):
    """
    Docstring for computeEdges
    
    :param X: x coordinates
    :param Y: y coordinates
    :param u: u velocity
    :param v: v velocity
    :param Nprob: problem number (for BCs)
    """
    Nxb,Nyb=u.shape
    Nx=Nxb-2
    Ny=Nyb-2
    ux = np.zeros((Nx+1,Ny))
    vx = np.zeros((Nx+1,Ny))
    uy = np.zeros((Nx,Ny+1))
    vy = np.zeros((Nx,Ny+1))

    #  Average to get edges
    ux = 0.5*(u[:-1,1:-1]+u[1:, 1:-1])
    vx = 0.5*(v[:-1,1:-1]+v[1:,1:-1])
    uy = 0.5*(u[1:-1,:-1]+u[1:-1,1:])
    vy = 0.5*(v[1:-1,:-1]+v[1:-1,1:])

    return ux,uy,vx,vy

def timestep(X,Y,un,vn,pn,nu,tn,dt,nProb):
    from Diagnostics import surfc,pExact, velExact, err_plots
    from Operators import MAC_projection, MAC_grad, Laplacian
    from MultiGrid import invDiffusion, periodicBC

    """
    Docstring for timestep
    
    :param X: Description
    :param Y: Description
    :param un: u velocity at time n
    :param vn: v velocity at time n
    :param pn: pressure at time n
    :param nu: viscosity
    :param tn: current time
    :param dt: time step size
    :param Nprob: problem number (for BCs)

    return:  unp: 

    """
    #print('In timestep',un.shape,vn.shape,pn.shape)

    thalf = tn + 0.5*dt
    tnp = tn + dt
    #  Set boundary conditions for un,vn
    un=periodicBC(un)
    vn=periodicBC(vn)
    pn=periodicBC(pn)   
    phi0=np.zeros(pn.shape)
    #  Compute advective term at time n
    advx, advy = computeAdvectionMAC(X,Y,un,vn,nProb)
    #surfc(X,Y, advx, 'Advective term U')
    #surfc(X,Y, advy, 'Advective term V')

    # Compute pressure gradient at time n
    p_x, p_y = centered_grad(pn)
    p_x=periodicBC(p_x)
    p_y=periodicBC(p_y)

    # Compute diffusive term at time n
    LapUn = Laplacian(un)
    LapVn = Laplacian(vn)
    LapUn=periodicBC(LapUn)
    LapVn=periodicBC(LapVn)

    # Compute first order guess for U* and V*
    ustar = un + dt*(-advx - p_x + nu*LapUn)
    vstar = vn + dt*(-advy - p_y + nu*LapVn)
    ustar=periodicBC(ustar)
    vstar=periodicBC(vstar)

    #  Implicitly solve for U* at time n+1
    rhsu = ustar - 0.5*dt*nu*LapUn  #  Removing half the diffusive term to set up Trapezoidal rule
    rhsv = vstar - 0.5*dt*nu*LapVn  #  Removing half the diffusive term to set up Trapezoidal rule
    #set periodic BCs again
    rhsu=periodicBC(rhsu)
    rhsv=periodicBC(rhsv)

    ustar = invDiffusion(X,Y,ustar, rhsu, -0.5*dt*nu,maxiter=30, tol=1e-9,Nprob=nProb)
    vstar = invDiffusion(X,Y,vstar, rhsv, -0.5*dt*nu/2.0,maxiter=30, tol=1e-9,Nprob=nProb)
    #surfc(X,Y, ustar, 'U star after implicit diffusion')
    #surfc(X,Y, vstar, 'V star after implicit diffusion')    
    #  Project U*
    unp,vnp,phi = centered_projection(X,Y,ustar,vstar,phi0,nProb)
    #surfc(X,Y, unp, 'U  after centered projection')
    #surfc(X,Y, vnp, 'V  after centered projection')  
    # set periodic BCs again on velocity
    unp=periodicBC(unp)
    vnp=periodicBC(vnp)
    # Set boundary conditions on phi
    phi=periodicBC(phi)

    #  Update pressure at time n+1
    pnp = pn + phi/dt - nu*Laplacian(phi)
    pnp=periodicBC(pnp)

    # Now do a corrector step
    #  Compute advective term at time n+1
    advxnp, advynp = computeAdvectionMAC(X,Y,unp,vnp,nProb)
    #surfc(X,Y, advx, 'Advective term U n+1')
    #surfc(X,Y, advy, 'Advective term V n+1')

    # Compute pressure gradient at time n+1
    pnp_x, pnp_y = centered_grad(pnp)
    pnp_x=periodicBC(pnp_x)
    pnp_y=periodicBC(pnp_y) 

    # Compute diffusive term at time n+1
    LapUnp = Laplacian(unp)  
    LapVnp = Laplacian(vnp)
    LapUnp=periodicBC(LapUnp)
    LapVnp=periodicBC(LapVnp)

    # Compute second order guess for U* and V*
    ustar = un + 0.5*dt*(-advx -advxnp - p_x  - pnp_x + nu*(LapUn + LapUnp))
    vstar = vn + 0.5*dt*(-advy -advynp - p_y  - pnp_y + nu*(LapVn + LapVnp))
    ustar=periodicBC(ustar)
    vstar=periodicBC(vstar)

    # Implicitly solve for U* at time n+1
    rhsu = ustar - 0.5*dt*nu*LapUnp  #  Removing half the diffusive term to set up Trapezoidal rule
    rhsv = vstar - 0.5*dt*nu*LapVnp  #  Removing half the diffusive term to set up Trapezoidal rule
    #set periodic BCs again
    rhsu=periodicBC(rhsu)
    rhsv=periodicBC(rhsv)   
    ustar = invDiffusion(X,Y,ustar, rhsu, -0.5*dt*nu,maxiter=30, tol=1e-9,Nprob=nProb)
    vstar = invDiffusion(X,Y,vstar, rhsv, -0.5*dt*nu,maxiter=30, tol=1e-9,Nprob=nProb)
    #surfc(X,Y, ustar, 'U star after implicit diffusion corrector')
    #surfc(X,Y, vstar, 'V star after implicit diffusion corrector')    
    #  Project U*
    unp,vnp,phi = centered_projection(X,Y,ustar,vstar,phi0,nProb)
    #surfc(X,Y, unp, 'U  after centered projection corrector')
    #surfc(X,Y, vnp, 'V  after centered projection corrector')  
    # set periodic BCs again on velocity
    unp=periodicBC(unp)
    vnp=periodicBC(vnp)
    # Set boundary conditions on phi
    phi=periodicBC(phi)         
    #  Update pressure at time n+1
    #pnp = pn + phi/dt - nu*Laplacian(phi)
    pnp=periodicBC(pnp) 
    phalf = 0.5*(pn+pnp) + phi/dt - nu/2.0*Laplacian(phi)
    
    return unp,vnp,pnp