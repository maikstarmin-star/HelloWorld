import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from MultiGrid import coarsen as coarsen, reflectbc, SetZeroBC
from Operators import MAC_div,MAC_grad,MAC_projection
def velExact(X, Y,t, nu, nProb=0):
    """
    Exact velocity field for testing on periodic box
    
    :param X: x coordinates
    :param Y: y coordinates 
    :param t: time
    :param nu: viscosity
    :param nProb: problem number
    """
    decay=np.exp(-8.0*np.pi**2*t*nu)
    c=0.75
    u = c+0.25*np.cos(2.0*np.pi*(X-c*t))*np.sin(2.0*np.pi*(Y-c*t))*decay
    v = c-0.25*np.sin(2.0*np.pi*(X-c*t))*np.cos(2.0*np.pi*(Y-c*t))*decay
    return u,v
def pExact(X, Y,t, nu,nProb=0):
    """
    Exact pressure field for testing on periodic box
    
    :param X: x coordinates
    :param Y: y coordinates 
    :param t: time
    :param nu: viscosity

    """
    c=0.75
    decay=np.exp(-16.0*np.pi**2*t*nu)
    p = (-1.0/64.0)*(np.cos(4.0*np.pi*(X-c*t)) + np.cos(4.0*np.pi*(Y-c*t)))*decay
    return p
def uExact(X, Y,nProb=0):
    """
    Exact solution for testing
    
    :param X: x coordinates
    :param Y: y coordinates 
    """
    freqX = 1.0*2.0*np.pi
    freqY = 0.25*2.0*np.pi
    # freqX=0.5
    # freqY=0.5
    # return X*(X-1)*Y
    return np.sin(freqX*X)*np.sin(freqY*Y)
    # return (X**2-X**4)*(Y**2-Y**4)

def lapExact(X, Y):
    """
    Return the exact Laplacian for testing
    
    :param X: x coordinates
    :param Y: y coordinates
    """
    freqX = 1.0*2.0*np.pi
    freqY = 0.25*2.0*np.pi
    return -(freqX**2+freqY**2)*np.sin(freqX*X)*np.sin(freqY*Y)
    # return 2*((1-6*X**2)*(Y**2)*(1-Y**2) + (1-6*Y**2)*(X**2)*(1-X**2))
    #
def velPlots(X, Y, u,v,  tstr=' '):
    """"
    Plot the velocity field and vorticity
    :param X: x coordinates
    :param Y: y coordinates
    :param u: x velocity
    :param v: y velocity
    :param tstr: title string
    """
    Nxb,Nyb=X.shape
    Nx=Nxb-2
    Ny=Nyb-2
    fig = plt.figure(figsize=(16, 16))

    ax1 = fig.add_subplot(221)
    #  Contour of u velocity
    c = ax1.contourf(X[1:-1,1:-1], Y[1:-1,1:-1], u[1:-1,1:-1], cmap='viridis')
    fig.colorbar(c, ax=ax1)
    ax1.set_title('u Velocity at '+ tstr)

    ax2 = fig.add_subplot(222)
    # Contour of v velocity
    c = ax2.contourf(X[1:-1,1:-1], Y[1:-1,1:-1], v[1:-1,1:-1], cmap='viridis')
    fig.colorbar(c, ax=ax2)
    ax2.set_title('v Velocity at '+ tstr)

    ax3 = fig.add_subplot(223)
    omega = np.zeros_like(u)
    dx=1.0/Nx
    dy=1.0/Ny
    omega[1:-1,1:-1] = (v[2:,1:-1]-v[0:-2,1:-1])/(2.0*dx) - (u[1:-1,2:]-u[1:-1,0:-2])/(2.0*dy)

    c = ax3.contour(X[1:-1,1:-1], Y[1:-1,1:-1], omega[1:-1,1:-1], cmap='viridis')
    fig.colorbar(c, ax=ax3)
    ax3.set_title('Vorticity at '+ tstr)
    # quiver of velocity field

    ax4 = fig.add_subplot(224)

    ax4.quiver(X[1:-1,1:-1], Y[1:-1,1:-1], u[1:-1,1:-1], v[1:-1,1:-1])
    ax4.set_title('Velocity Field at '+ tstr)

    plt.suptitle('Velocity Field and Vorticity at '+ tstr)

    plt.show()
def err_plots(X, Y, f, fExact, tstr=' '):
    Nxb,Nyb=X.shape
    Nx=Nxb-2
    Ny=Nyb-2
    fig = plt.figure(figsize=(16, 16))

    ax1 = fig.add_subplot(221, projection='3d')

    ax1.plot_surface(X, Y, f, cmap='viridis')
    ax1.set_title('Numerical Solution')

    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot_surface(X, Y, fExact, cmap='viridis')
    ax2.set_title('Exact Solution')

    ax3 = fig.add_subplot(223, projection='3d')
    ax3.plot_surface(X, Y, fExact-f, cmap='viridis')
    ax3.set_title('Difference: ' + str(np.max(np.fabs(f-fExact))))
    # print('Max difference:', np.max(np.fabs(f-fExact)))

    title = tstr + ': Max difference =' + str(np.max(np.fabs(f-fExact)))

    ax4 = fig.add_subplot(224, projection='3d')
    ax4.plot_surface(X[1:-1,1:-1], Y[1:-1,1:-1], fExact[1:-1,1:-1]-f[1:-1,1:-1], cmap='viridis')

    intNorm=interior_norm(f-fExact)
    ax4.set_title('Interior Difference: ' +str(intNorm))
    plt.suptitle(title)
    plt.show()


def surfc(X, Y, f,stitle=' '):

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')

    ax1.plot_surface(X, Y, f, cmap='viridis')
    ax1.set_title('With Boundary Conditions')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X[1:-1, 1:-1], Y[1:-1, 1:-1],
                     f[1:-1, 1:-1], cmap='viridis')
    ax2.set_title('Interior Only: max=' + str(np.max(np.fabs(f[1:-1, 1:-1]))))
    plt.suptitle(stitle)
    plt.show()


def interior_norm(f) -> float:

    """
    Compute the max norm over interior points
    
    :param f: A two dimensional numpy array representing the grid function.

    :return: The maximum absolute value of f over the interior points.
    """

    return np.max(np.fabs(f[1:-1, 1:-1]))



def my_contour(phi,X,Y,desc_str=' '):
    fig, ax = plt.subplots()
    CS = ax.contour(phi, X,Y)
    ax.clabel(CS, fontsize=10)
    ax.set_title('Contours of ',desc_str)


def test_coarsening(u,uexc,Xc,Yc, do_plots=False):
    """
    Test the coarsening routine in multigrid
    
    :param u: Function to be coarsened
    :param uexc: Exact coarse solution
    :param Xc: X location of cell centers
    :param Yc: Y location of cell centers

    """
    from MultiGrid import coarsen as coarsen, reflectbc
    Nxb,Nyb = u.shape
    N = Nxb-2
    dx=1/N
    Nover2 = int(N/2)

    uc = coarsen(u)
    uc = reflectbc(uc,Nover2,dx*2)
    errNorm = interior_norm(uc-uexc)

    if (do_plots):
        err_plots(Xc,Yc,uc,uexc, 'Difference after coarsening: ')
    else:
        print('Max difference after coarsening:', interior_norm(uc-uexc))

    return errNorm

def test_proj(X,Y,Xu,Yu,Xv,Yv,N, doPlots=False):

    """
    Test the projection operator
    """
    beta=0.001
    phiex=-1/64*(np.cos(4*np.pi*X)-np.cos(4*np.pi*Y))
    #phiex=SetZeroBC(phiex,bType=1)
 
    phi_x,phi_y = MAC_grad(phiex)
    #print('Shape of phi_x,phi_y', phi_x.shape,phi_y.shape)


    uex=np.cos(2*np.pi*Xu)*np.sin(2*np.pi*Yu)
    vex=-np.sin(2*np.pi*Xv)*np.cos(2*np.pi*Yv)

    ustar=uex+phi_x
    vstar=vex+phi_y

   
    #  start with a zero guess for phi
    phi0=np.random.rand(N+2,N+2)
    phi0=phi0-np.sum(phi0)/(N*N)
    #phi0=phiex*phiex
    #phi0=phiex*10
    phi0=-1/64*(np.cos(8*np.pi*X)-np.cos(2*np.pi*Y))
    phi0=SetZeroBC(phi0,bType=1)
    u,v,phi=MAC_projection(X,Y,ustar,vstar,phi0)

    divU=MAC_div(u,v)
    #surfc(X,Y, divU, 'Div after Projection')
    errNorm = interior_norm(phi-phiex)
    if (doPlots):  
        err_plots(X,Y,phi,phiex,'Mac Projection difference')  
    else:
        print('Max difference after MAC projection:', errNorm)
    return errNorm

def test_diffusion(X,Y,Xu,Yu,Xv,Yv,N,doPlots=False):
    from Operators import Laplacian
    from MultiGrid import invDiffusion

    """
    Test the inverse diffusion operator
    """
    
    phiex=-1/2*(np.sin(4*np.pi*X)*np.sin(4*np.pi*Y))

    beta=-0.001
    alpha = 1.0
    rhs = alpha*phiex + beta*Laplacian(phiex)
    #surfc(X,Y,rhs, 'rhs in test_diff')

   
    #  start with a zero guess for phi
    phi0=np.random.rand(N+2,N+2)
    phi0=phi0-np.sum(phi0)/(N*N)
    #phi0=phiex*phiex
    #phi0=phiex*10
    phi0=-1/64*(np.sin(8*np.pi*X)*np.sin(2*np.pi*Y))
    phi0=SetZeroBC(phi0,bType=0)
    
    phi=invDiffusion(X,Y,phi0, rhs, beta, maxiter=100, tol=1e-6)
    errNorm=interior_norm(phi-phiex)

    if (doPlots):  
        err_plots(X,Y,phi,phiex,'Inv diffusion difference')  
    else:
        print('Max difference after Inverse Diffusion:', errNorm)
  
    return errNorm

def test_Laplacian(X,Y, nProb=0, doPlots=False):
    from Operators import Laplacian
    """
    Test the interpolation routine in multigrid
    
    :param u: Function to be interpolated

    """
    uex=uExact(X,Y)
    lap=Laplacian(uex) 
    lapEx=lapExact(X,Y)
    errNorm=interior_norm(lap-lapEx)

    if (doPlots):  
        err_plots(X,Y,lap,lapEx,'Laplacian difference')  
    else:
        print('Max difference of Laplacian:', errNorm)  

    return errNorm

    