import numpy as np

def Eind(npixels):
    """Get indexers for the E matrix"""
    x = np.ones((npixels**2,))
    mask = np.arange(len(x)) % npixels == 0
    x[mask] = 0; x = np.roll(x,-1)
    A = np.diag(x,k=1) #horizontally adjacent

    x = np.ones((npixels**2-npixels,))
    B = np.diag(x,k=npixels) #vertically adjacent

    x = np.ones((npixels**2-npixels,))
    mask = np.arange(len(x)) % npixels == 0
    x[mask] = 0; x = np.roll(x,-1)
    C = np.diag(x,k=npixels+1) #right diagonal

    x = np.ones((npixels**2-npixels,))
    mask = np.arange(len(x)) % npixels == 0
    x[mask] = 0
    D = np.diag(x,k=npixels-1) #left diagonal

    F = np.eye(npixels**2) #autocorrelation

    Aind = np.where(A > 0); Bind = np.where(B > 0)
    Cind = np.where(C > 0); Dind = np.where(D > 0)
    Find = np.where(F > 0)
    return Aind,Bind,Cind,Dind,Find

def Sind(npixels):
    """Get indexers for the covariance map"""
    checker = np.indices((2*npixels-1,2*npixels-1)).sum(axis=0) % 2
    checker = 1-checker
    checker[::2,:] *= 2
    checker[::2,:] += 2
    Vind = np.where(checker == 0); RLind = np.where(checker == 1)
    Hind = np.where(checker == 2); Dind = np.where(checker == 4)
    return Vind, RLind, Hind, Dind
  
def linear_interpolation(image):
    rows, cols = image.shape
    interpolated_image = np.zeros_like(image)
    nx,ny = image.shape
    zero_pixels = np.argwhere(image == 0)
    for row, col in zero_pixels:
        if row > 0 and row < nx-1 and col > 0 and col < ny-1:
            image[row, col] = (image[row+1, col]+image[row-1, col]+image[row, col+1]+image[row, col-1])/4
        else:
            image[row,col] = 1.0
    image = np.pad(image,((0,1),(0,1)),mode='constant')
    image[-1,:] = 1.0; image[:,-1] = 1.0
    return image
    
def G2(adu):

    nt,nx,ny = adu.shape
    adu = adu.astype(np.float64)
    M = np.mean(adu,axis=0) #time average
    M = M.reshape((nx*ny,))
    _ExEy = np.outer(M,M)

    adur = adu.reshape((nt,nx*ny))
    adur = adur.astype(np.float64)
    fft = np.fft.fft(adur,axis=0)
    fftc = fft.conj()
    fft = fft[:,:,np.newaxis]
    fftc = fftc[:,np.newaxis,:]
    corr = np.fft.ifft(fft*fftc,axis=0)
    _Exy = np.real(corr)/nt #take zero lag

    Exy = np.zeros((nt,2*nx-1,2*nx-1),dtype=np.float64)
    ExEy = np.zeros((2*nx-1,2*nx-1),dtype=np.float64)

    Eh,Ev,Er,El,Ed = Eind(nx)
    Vind, RLind, Hind, Dind = Sind(nx)
    
    Exy[:,Vind[0],Vind[1]] = _Exy[:,Ev[0],Ev[1]]
    Exy[:,RLind[0],RLind[1]] = _Exy[:,Er[0],Er[1]] #or _Exy[:,El[0],El[1]]
    Exy[:,Hind[0],Hind[1]] = _Exy[:,Eh[0],Eh[1]]
    #Exy[:,Dind[0],Dind[1]] = _Exy[:,Ed[0],Ed[1]]
    Exy[:,Dind[0],Dind[1]] = 0.0 #diagonals are <x^2>/<x>^2 = 1 + 1/mu
    
    ExEy[Vind[0],Vind[1]] = _ExEy[Ev[0],Ev[1]]
    ExEy[RLind[0],RLind[1]] = _ExEy[Er[0],Er[1]]
    ExEy[Hind[0],Hind[1]] = _ExEy[Eh[0],Eh[1]]
    #ExEy[Dind[0],Dind[1]] = _ExEy[Ed[0],Ed[1]]
    ExEy[Dind[0],Dind[1]] = 0.0
    #g2 = linear_interpolation(Exy[0]/(ExEy+1e-14))
    g2 = Exy[0]
    
    return g2
