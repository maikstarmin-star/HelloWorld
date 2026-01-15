import numpy as np

def coarsen(phi):
    """Coarsen the grid by a factor of 2"""
    Nxb, Nyb = phi.shape
    Nx=Nxb-2
    Ny=Nyb-2  
    Nc = int(Nx/2)
    phicoarse = np.zeros((Nc+2, Nc+2))

    #  lets use index magic instead of loops
    ii = np.arange(1, Nx+1,2)
    jj = np.arange(1, Ny+1,2)

    phicoarse[1:-1, 1:-1] = 0.25*(phi[np.ix_(ii, jj)] + phi[np.ix_(ii, jj+1)] + phi[np.ix_(ii+1, jj)] 
                                + phi[np.ix_(ii+1, jj+1)])
    return phicoarse


def interp(phi):
    """
    Interpolate the grid by a factor of 2
    
    :param phi: function to interpolate

    If the interpolation is 2nd order, boundary conditions must be set
    """

    Nxb, Nyb = phi.shape
    Nx=Nxb-2
    Ny=Nyb-2  
    Nf = int(2*Nx)
    phifine = np.zeros((Nf+2, Nf+2))
    order = 2
    ii = np.arange(1, Nf+1,2)
    jj = np.arange(1, Nf+1,2)
    ic = np.arange(1, Nx+1)
    jc = np.arange(1, Ny+1)

    if order == 1:
        phifine[np.ix_(ii, jj)] = phi[np.ix_(ic, jc)]
        phifine[np.ix_(ii, jj+1)] = phi[np.ix_(ic, jc)]
        phifine[np.ix_(ii+1, jj)] = phi[np.ix_(ic, jc)]
        phifine[np.ix_(ii+1, jj+1)] = phi[np.ix_(ic, jc)]
    else:


        phifine[np.ix_(ii, jj)] = (9.0*phi[np.ix_(ic, jc)] + 3.0*(phi[np.ix_(ic, jc-1)] + phi[np.ix_(ic-1, jc)])
                                     + phi[np.ix_(ic-1, jc-1)])*0.0625
        phifine[np.ix_(ii, jj+1)] = (9.0*phi[np.ix_(ic, jc)] + 3.0*(phi[np.ix_(ic, jc+1)] + phi[np.ix_(ic-1, jc)])
                                     + phi[np.ix_(ic-1, jc+1)])*0.0625
        phifine[np.ix_(ii+1, jj)] = (9.0*phi[np.ix_(ic, jc)] + 3.0*(phi[np.ix_(ic, jc-1)] + phi[np.ix_(ic+1, jc)])
                                     + phi[np.ix_(ic+1, jc-1)])*0.0625
        phifine[np.ix_(ii+1, jj+1)] = (9.0*phi[np.ix_(ic, jc)] + 3.0*(phi[np.ix_(ic, jc+1)] + phi[np.ix_(ic+1, jc)])
                                     + phi[np.ix_(ic+1, jc+1)])*0.0625

    return phifine


def residual(u, rhs, alpha=0.0,beta=1.0):
    from Operators import Laplacian
    """
    Compute the residual res=rhs - (alpha*u + beta*Laplacian(u))

    :param u: function
    :param rhs: right hand side

    return:  residual at cell centers

    Notes
        boundary conditions on u should be set before calling
    """

    lapU = Laplacian(u)
    res = rhs - (alpha*u + beta*lapU)

    return res


def Jacobi(u, rhs, alpha=0.0, beta=1.0,bType=2):
    """
    Do one interation of weighted jacobi on the problem

    alpha*u+ beta*Laplacian(u)=rhs
    
    :param u: approximation of solution
    :param rhs: right hand side of equation

    :param alpha: equation paramter
    :param beta: sequation parameter

    return:  u 

    We are assuming that the ghost cells on u already updated


    """

    Nxb, Nyb = u.shape
    Nx=Nxb-2
    Ny=Nyb-2
    dx = 1.0/Nx
    cst = dx*dx/beta

    unew = np.zeros(u.shape)
    # Choose an overrelaxation parameter
    if alpha == 0.0:
        omega = 4/5
    else:
        omega = 1
    ii = np.arange(1, Nx+1)
    jj = np.arange(1, Ny+1)

    unew[np.ix_(ii, jj)] = (cst*rhs[np.ix_(ii, jj)] - u[np.ix_(ii+1, jj)] - u[np.ix_(ii-1, jj)]
                                - u[np.ix_(ii, jj+1)] - u[np.ix_(ii, jj-1)])/(alpha*cst -4.0)

    u = (1.0-omega)*u + omega*unew
    return u

def GSRB(phi,rhs):
    """
    Docstring for GSRB
    
    :param phi: Description
    :param rhs: Description
    """
        # for i in range(1,N+1):
    # for j in range(1+mod(i,2),N+1,2):
    # u[i,j] = (cst*rhs[i,j] + u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])/4.0

    # Loop over black interior points
    # for i in range(1,N+1):
    # for j in range(0+mod(i,2),N+1,2):
    # u[i,j] = (cst*rhs[i,j] + u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])/4.0
    return phi
# Plot the exact solution and the numerical solution


def zerobcDirichlet(u):
    """
    Set zero boundary conditions explicitly for Dirichlet problem
    
    :param u: function to set boundary conditions on
    """
    #  
    u[0, :] = 0.0  # left boundary
    u[-1, :] = 0.0  # right boundary
    u[:, 0] = 0.0  # bottom boundary
    u[:, -1] = 0.0  # top boundary
    return u


def zerobcReflect(u):
    """
    Set zero boundary conditions by reflection
    This is used in Jacobi interations because we are always solving for the correction
    
    :param u: Grid function to set boundary conditions on

    """

    u[0, 1:-1] = -u[1, 1:-1]   # left boundary
    u[-1, 1:-1] = -u[-2, 1:-1]  # right boundary
    u[:, 0] = -u[:, 1]  # bottom boundary
    u[:, -1] = -u[:, -2]  # top boundary
    return u


def zerobcNeumann(u):
    """
     Set Homogeneous Neumann boundary conditions
    
    :param u: function to set boundaries on

    This is used in Jacobi interations because we are always solving for the correction
    """

    u[0, 1:-1] = u[1, 1:-1]   # left boundary
    u[-1, 1:-1] = u[-2, 1:-1]  # right boundary
    u[:, 0] = u[:, 1]  # bottom boundary
    u[:, -1] = u[:, -2]  # top boundary
    return u

def periodicBC(u):
    """
    Set periodic boundary conditions
    
    :param u: function to set boundaries on

    """

    u[0, 1:-1] = u[-2, 1:-1]   # left boundary
    u[-1, 1:-1] = u[1, 1:-1]  # right boundary
    u[:, 0] = u[:, -2]  # bottom boundary
    u[:, -1] = u[:, 1]  # top boundary
    return u
def reflectbc(u, N, dx, t=0):
    """
    Set reflective boundary conditions 
    This will make the walls and bottom mimic a zero boundary condition and the top 
    to match the forced flow

    
    :param u: velocity to set boundary condition on
    :param t: time (for time dependent forcing)
    """
    top = setTopBoundary(u, N, dx, t)
    u[0, 1:N+1] = -u[1, 1:N+1]   # left boundary
    u[N+1, 1:N+1] = -u[N, 1:N+1]  # right boundary
    u[0:N+2, N+1] = 2*top - u[0:N+2, N]  # top boundary (with corners)
    u[0:N+2, 0] = -u[0:N+2, 1]  # bottom boundary (with corners)

    return u


def setTopBoundary(u, N, dx, t=0.0,nProb=0 ):
    """
    Docstring for setTopBoundary
    
    :param u: Description
    :param N: Description
    :param dx: Grid spacing
    :param t: Time, for time dependent problems
    :param nProb: Defines which problem is being run
    """
    from Diagnostics import uExact
    
    freqX = 1
    x = np.linspace(-dx/2, 1+dx/2, N+2)
    y = 1+np.zeros(N+2)
    return uExact(x, y)

def SetZeroBC(c, bType: int):
    """
    Set ghost cells for homogeneous boundary conditions.  
    In multigrid, we are always solving for a correction, so 
    zero boundary conditions are appropriate
    
    :param c: Correction grid function
    :param bType: Determines boundary type, 
        0 Dirichlet
        1 Neumann
        2 Periodic

    """
    if(bType == 0):
        # Dirichlet boundary conditions
        #return zerobcDirichlet(c)
        return zerobcReflect(c)
    elif(bType == 1):
        # Neumann boundary conditions
        return zerobcNeumann(c)
    elif(bType == 2):
        # Periodic boundary conditions
        return periodicBC(c)
    else:
        print('Error:  Unknown boundary condition type in SetZeroBC')
    return c
    


def setbc(u, N, dx):
    from Diagnostics import uExact
    order = 2
    x = np.linspace(0-dx/2, 1+dx/2, N+2)
    y = np.linspace(0-dx/2, 1+dx/2, N+2)
    X, Y = np.meshgrid(x, y, indexing='ij')
    if (order == 2):
        #print('order is 2')
        a = 2
        b = -1
        # Set the boundary conditions (Dirichlet)
        u[0, :] = a*u[1, :] + b*u[2, :]  # left boundary
        u[N+1, :] = a*u[N, :] + b*u[N-1, :]  # right boundary
        u[:, 0] = a*u[:, 1] + b*u[:, 2]  # bottom boundary
        u[:, N+1] = a*u[:, N] + b*u[:, N-1]  # top boundary
    elif (order==0 ):
        for i in range(0, N+2):
            u[i, 0] = uExact(X[i, 0], Y[i, 0])  # left boundary
            u[i, N+1] = uExact(X[i, N+1], Y[i, N+1])  # top boundary
            u[0, i] = uExact(X[0, i], Y[0, i])  # left boundary
            u[N+1, i] = uExact(X[N+1, i], Y[N+1, i])  # top boundary
    else:   
        # print('order is 4')
        a = -3
        b = 1
        c = -0.2
        
        d = 16/5
        # Set the boundary conditions (Dirichlet)
        u[0, :] = a*u[1, :] + b*u[2, :] + c*u[3, :]  # left boundary
        u[N+1, :] = a*u[N, :] + b*u[N-1, :] + c*u[N-2, :]  # right boundary
        u[:, 0] = a*u[:, 1] + b*u[:, 2] + c*u[:, 3]  # bottom boundary
        u[:, N+1] = a*u[:, N] + b*u[:, N-1] + c*u[:, N-2]  # top boundary

    return u


def vcycle(u, rhs,alpha=0.0,beta=1.0,Nprob=0):
    """
    Do a multigrid v-cycle for 
    alpha*u+ beta*Laplacian(u)=rhs
    
    :param u: solution (when called is usually a correction)
    :param rhs: right hand side (when called is usually a residual)

    """

    # Set boundary type
    if(alpha != 0.0):
        bType = 0     # Dirichlet
    else:
        bType = 1 # Neumann

    if(Nprob < 10):
        bType = 2  # Periodic

    Nxb,Nyb=u.shape
    N = Nxb-2       #  Assuming for now a box with equal points in each direction
    dx = 1.0/N    #  Assuming a unit box with uniform grid spacing
    #print('vcycle with N=', N)
    # Relax a few times
    Njacobi = 3
    u=SetZeroBC(u, bType)
    for kk in range(Njacobi):
        u = Jacobi(u, rhs, alpha, beta,bType)
        u=SetZeroBC(u, bType)
    
    if (N > 4):
        Nover2 = int(N/2)
        # Compute the residual  (boundary conditions are set coming out of Jacobi)
        res = residual(u, rhs,alpha,beta)

        # Coarsen the residual
        rescoarse = coarsen(res)

        correction = np.zeros((Nover2+2, Nover2+2))
        #  call the v-cycle on the coarsened residual
        correction = vcycle(correction, rescoarse,alpha,beta,Nprob)

        # Interpolate the correction back to the fine grid
        correction = SetZeroBC(correction,bType)
        correctionfine = interp(correction)

        # Apply the correction to the fine grid solution
        u = u+correctionfine

        # Relax a few more times
        u = SetZeroBC(u,bType)
        for kk in range(Njacobi):
            u = Jacobi(u, rhs, alpha, beta,bType)
            u = SetZeroBC(u,bType)

    return u

def invLapacian(phi0, rhs, maxiter=100, tol=1e-6,Nprob=0):

    """
    Solve Lap(phi) = rhs using multigrid
    
    :param phi0: Initial guess to solution
    :param rhs: Right hand side
    :param maxiter: Maximum Multigrid iterations
    :param tol: Residual tolerance stopping crieterion
    :param Nprob:  Problem number for boundary conditions

    return:  solution phi

    Note:  The boundary conditions here are Neummann.  Since multigrid is only called on the correction/residual problem,
    multigrid will use homegeneous Neumann boundary conditions.  The phi boundary conditions are the normal component of 
    Ustar here, since this is called for a projection.  Hence Ustar is passed in, but only to get boundary conditions

    """
    from Diagnostics import  interior_norm as interior_norm, surfc as surfc
    Nxb,Nyb=phi0.shape
    N=Nxb-2
    dx=1.0/N
    phi=np.zeros((Nxb,Nyb))


    #  For Laplacian, alpha + beta*Laplacian means 
    alpha=0.0
    beta=1.0

    k=0
    phi=phi0
    res=residual(phi,rhs)
    resNorm=interior_norm(res)
    verbose = 1  # Set to 2 to print iteration residual Set to 1 to print residuals, 0 to be quiet
    while resNorm > tol and k < maxiter: 
        c=np.zeros((N+2,N+2))
        c= vcycle(c,res,alpha,beta,Nprob)

        # Update the approximation with correction
        phi=phi + c
        # phi = zerobcNeumann(phi)  # Probably redundant
        res=residual(phi, rhs, alpha,beta)
        resNorm=interior_norm(res)
        k+=1
        if verbose==2:
            print('Vcycle ',k, '  Residual norm:', resNorm)
    if verbose==1:
        if k < maxiter:
            print('Inverse Laplcian MG Converged in ',k,' v-cycles,  Residual Norm = ',resNorm)
        else:
            print('Inverse Laplcian MG Did not converge in ',maxiter,'v-cycles,  Residual Norm = ',resNorm)
            
    return phi

def invDiffusion(X,Y,phi0, rhs, beta, maxiter=100, tol=1e-6, Nprob=0):

    """
    Solve alpha*phi + beta*Lap(phi) = rhs using multigrid
    Typically beta will be nu*dt or nu*dt/2 depending on the integrator
    and alpha = 1
    
    :param phi0: Initial guess to solution
    :param rhs: DRight hand side
    :param beta:  diffusion coefficietn
    :param maxiter: Maximum Multigrid iterations
    :param tol: Residual tolerance stopping crieterion
    :param Nprob:  Problem number for boundary conditions

    return:  solution phi

    Note:  The boundary conditions here are NDirichelet.  Since multigrid is only called on the correction/residual problem,
    multigrid will use homegeneous Dirichlet boundary conditions.  
    """
    from Diagnostics import  interior_norm as interior_norm, surfc as surfc
    Nxb,Nyb=phi0.shape
    N=Nxb-2
    dx=1.0/N
    phi=np.zeros((Nxb,Nyb))


    #  For Diffusion, alpha + beta*Laplacian means 
    alpha=1.0

    k=0
    phi=phi0
    res=residual(phi,rhs,alpha,beta)
    resNorm=interior_norm(res)
    verbose = 1  # Set to 2 to print iteration residual Set to 1 to print residuals, 0 to be quiet

    while resNorm > tol and k < maxiter: 
        c=np.zeros((N+2,N+2))
        c= vcycle(c,res,alpha,beta)

        # Update the approximation with correction
        phi=phi + c

        res=residual(phi, rhs, alpha, beta)
        resNorm=interior_norm(res)
        k+=1
        if verbose==2:
            print('Vcycle ',k, '  Residual norm:', resNorm)

    if verbose==1:
        if k < maxiter:
            print('Inverse Laplcian MG Converged in ',k,' v-cycles,  Residual Norm = ',resNorm)
        else:
            print('Inverse Laplcian MG Did not converge in ',maxiter,'v-cycles,  Residual Norm = ',resNorm)
            
    return phi
  
